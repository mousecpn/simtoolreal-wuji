"""Compare FP32 and CUDA AMP training for the C4 actor and critic.

This is a small synthetic smoke test. It exercises the real C4 network,
escnn training-mode basis expansion, the SAPG coefficient conditioning, an
optimizer update, strict checkpoint save/load, finite gradients, and C4
equivariance after the update. It does not start Isaac Sim.
"""

from __future__ import annotations

import argparse
import copy
import io
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rl_games"))

import torch
import yaml

from scripts.verify_c4_equivariance import transform_action, transform_obs
from simtoolreal_shared.c4_rl import C4Builder, FIELDS


def field_size(fields):
    return sum(FIELDS[field][0] for field in fields)


def build_network(params, fields, *, device, hidden=12, input_channels=9):
    local = copy.deepcopy(params)
    local["hidden_channels"] = hidden
    local["input_channels"] = input_channels
    local["layers"] = 1
    local["mlp_units"] = [48, 48, 24, 24]
    builder = C4Builder()
    builder.load(local)
    raw = field_size(fields)
    ids = torch.linspace(50.0, 0.0, 6, device=device)
    network = builder.build(
        "a2c",
        actions_num=28,
        input_shape=(raw + 1,),
        num_seqs=8,
        value_size=1,
        normalize_input=False,
        type="extra_param",
        coef_ids=ids,
        coef_id_idx=raw,
    )
    return network.to(device=device, dtype=torch.float32), ids


def finite_parameters(module):
    return all(torch.isfinite(p).all().item() for p in module.parameters())


def make_batch(fields, ids, batch, device):
    raw = field_size(fields)
    obs = torch.randn(batch, raw, device=device)
    coeff = ids.repeat((batch + len(ids) - 1) // len(ids))[:batch]
    return torch.cat((obs, coeff[:, None]), dim=-1)


def run_update(actor, critic, actor_obs, critic_obs, *, amp, amp_dtype, steps):
    actor.train()
    critic.train()
    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    records = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        context = (
            torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp)
            if amp else torch.autocast(device_type="cpu", enabled=False)
        )
        with context:
            mean, log_std, value, _ = actor({"obs": actor_obs})
            critic_value, _ = critic({"obs": critic_obs})
            loss = (
                mean.float().square().mean()
                + log_std.float().square().mean()
                + value.float().square().mean()
                + critic_value.float().square().mean()
            )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [
            p.grad for p in list(actor.parameters()) + list(critic.parameters())
            if p.grad is not None
        ]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
            raise RuntimeError(f"non-finite or missing gradients at step {step}")
        scaler.step(optimizer)
        scaler.update()
        if not finite_parameters(actor) or not finite_parameters(critic):
            raise RuntimeError(f"non-finite parameters at step {step}")
        records.append(float(loss.detach().cpu()))
    return optimizer, scaler, records


@torch.no_grad()
def equivariance_error(actor, critic, actor_obs, critic_obs, fields, critic_fields):
    actor.eval()
    critic.eval()
    base_actor = actor({"obs": actor_obs})
    base_critic = critic({"obs": critic_obs})[0]
    error = 0.0
    for k in range(4):
        rotated_actor = torch.cat(
            (transform_obs(actor_obs[:, :-1], fields, k), actor_obs[:, -1:]), -1
        )
        rotated_critic = torch.cat(
            (transform_obs(critic_obs[:, :-1], critic_fields, k), critic_obs[:, -1:]), -1
        )
        actual_actor = actor({"obs": rotated_actor})
        actual_critic = critic({"obs": rotated_critic})[0]
        error = max(
            error,
            (actual_actor[0] - transform_action(base_actor[0], k)).abs().max().item(),
            (actual_actor[1] - base_actor[1]).abs().max().item(),
            (actual_actor[2] - base_actor[2]).abs().max().item(),
            (actual_critic - base_critic).abs().max().item(),
        )
    return error


def checkpoint_round_trip(actor, critic, actor_obs, critic_obs):
    buffer = io.BytesIO()
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict()}, buffer)
    buffer.seek(0)
    state = torch.load(buffer, map_location=actor_obs.device, weights_only=True)
    restored_actor = copy.deepcopy(actor)
    restored_critic = copy.deepcopy(critic)
    restored_actor.load_state_dict(state["actor"], strict=True)
    restored_critic.load_state_dict(state["critic"], strict=True)
    for before, after in zip(
        actor({"obs": actor_obs})[:3], restored_actor({"obs": actor_obs})[:3]
    ):
        torch.testing.assert_close(before, after, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        critic({"obs": critic_obs})[0], restored_critic({"obs": critic_obs})[0],
        atol=2e-5, rtol=2e-5,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="float16")
    args = parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("This experiment requires a CUDA device; run it on fang.")
    amp_dtype = getattr(torch, args.amp_dtype)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_float32_matmul_precision("high")

    config = yaml.safe_load(
        (ROOT / "isaacsimenvs/cfg/train/SimToolRealC4SAPG.yaml").read_text()
    )["params"]
    actor_params = config["network"]
    critic_params = config["config"]["central_value_config"]["network"]
    actor_fields = actor_params["obs_fields"]
    critic_fields = critic_params["obs_fields"]
    device = torch.device(args.device)

    fp_actor, ids = build_network(actor_params, actor_fields, device=device)
    fp_critic, _ = build_network(critic_params, critic_fields, device=device)
    amp_actor = copy.deepcopy(fp_actor)
    amp_critic = copy.deepcopy(fp_critic)
    actor_obs = make_batch(actor_fields, ids, args.batch, device)
    critic_obs = make_batch(critic_fields, ids, args.batch, device)
    # Ensure both runs see exactly the same inputs and initial weights.
    actor_state = copy.deepcopy(fp_actor.state_dict())
    critic_state = copy.deepcopy(fp_critic.state_dict())
    amp_actor.load_state_dict(actor_state, strict=True)
    amp_critic.load_state_dict(critic_state, strict=True)

    _, _, fp_losses = run_update(
        fp_actor, fp_critic, actor_obs, critic_obs,
        amp=False, amp_dtype=amp_dtype, steps=args.steps,
    )
    _, _, amp_losses = run_update(
        amp_actor, amp_critic, actor_obs, critic_obs,
        amp=True, amp_dtype=amp_dtype, steps=args.steps,
    )
    fp_error = equivariance_error(
        fp_actor, fp_critic, actor_obs, critic_obs, actor_fields, critic_fields
    )
    amp_error = equivariance_error(
        amp_actor, amp_critic, actor_obs, critic_obs, actor_fields, critic_fields
    )
    checkpoint_round_trip(amp_actor, amp_critic, actor_obs, critic_obs)
    print(f"FP32 losses: {fp_losses}")
    print(f"AMP {args.amp_dtype} losses: {amp_losses}")
    print(f"FP32 equivariance max error: {fp_error:.6g}")
    print(f"AMP equivariance max error: {amp_error:.6g}")
    print(
        f"PASS C4 AMP smoke: dtype={args.amp_dtype}, steps={args.steps}, "
        f"finite gradients/parameters, checkpoint round-trip, "
        f"equivariance error={amp_error:.6g}"
    )


if __name__ == "__main__":
    main()
