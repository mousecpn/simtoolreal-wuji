"""Verify privileged critic invariance, rl_games wrapper, update and checkpoints.

Run: .venv_isaacsim/bin/python scripts/verify_c4_critic.py --device cuda
"""

import argparse
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rl_games"))

import torch
import yaml
from rl_games.algos_torch.model_builder import ModelBuilder

from scripts.verify_c4_equivariance import transform_obs
from simtoolreal_shared.c4_rl import FIELDS


def check_critic(network, obs, atol=2e-5):
    value, state = network({"obs": obs})
    assert state is None and value.shape == (obs.shape[0], 1)
    assert torch.isfinite(value).all()
    error = 0.0
    for k in range(4):
        actual, _ = network({"obs": transform_obs(obs, network.encoder.fields, k)})
        torch.testing.assert_close(actual, value, atol=atol, rtol=atol)
        error = max(error, (actual - value).abs().max().item())
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(2)
    cfg = yaml.safe_load(
        (ROOT / "isaacsimenvs/cfg/train/SimToolRealC4PPO.yaml").read_text()
    )
    params = cfg["params"]["config"]["central_value_config"]
    fields = params["network"]["obs_fields"]
    task = yaml.safe_load(
        (ROOT / "isaacsimenvs/cfg/task/SimToolRealC4.yaml").read_text()
    )
    assert fields == task["obs"]["state_list"]
    assert params["network"]["mlp_units"] == [1024, 1024, 512, 512]
    n = sum(FIELDS[f][0] for f in fields)

    def build(dtype):
        return (
            ModelBuilder()
            .load(params)
            .build(
                {
                    "actions_num": 28,
                    "input_shape": (n,),
                    "num_seqs": 3,
                    "value_size": 1,
                    "normalize_input": False,
                    "normalize_value": True,
                }
            )
            .to(device=args.device, dtype=dtype)
        )

    for dtype, tol in ((torch.float64, 1e-10), (torch.float32, 2e-5)):
        model = build(dtype)
        network = model.a2c_network
        assert not model.is_rnn() and model.get_default_rnn_state() is None
        assert n == 196 and network.encoder.out_type.size == 202
        # Each hidden layer has exactly the baseline scalar width.
        assert [m.out_type.size for m in network.mlp if hasattr(m, "out_type")] == [
            1024,
            1024,
            512,
            512,
        ]
        obs = torch.randn(7, n, device=args.device, dtype=dtype)
        error = check_critic(network, obs, tol)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        before = [p.detach().clone() for p in model.parameters()]
        prediction = model({"obs": obs, "is_train": True})["values"]
        loss = (prediction - torch.randn_like(prediction)).square().mean()
        optimizer.zero_grad()
        loss.backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        optimizer.step()
        assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
        error = max(error, check_critic(network, obs, tol))
        restored = build(dtype)
        for source_train in (True, False):
            model.train(source_train)
            # Freeze running statistics while testing cross-mode weight loads.
            # rl_games updates these even on denorm=True if the module trains.
            model.value_mean_std.eval()
            with torch.no_grad():
                expected = model({"obs": obs, "is_train": False})["values"]
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            state = torch.load(buffer, weights_only=True, map_location=args.device)
            assert not any(k.endswith((".matrix", ".expanded_bias")) for k in state)
            for target_train in (True, False):
                restored.train(target_train)
                restored.value_mean_std.eval()
                with torch.no_grad():
                    for p in restored.parameters():
                        p.add_(0.1)
                restored.load_state_dict(state, strict=True)
                with torch.no_grad():
                    actual = restored({"obs": obs, "is_train": False})["values"]
                torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
                error = max(error, check_critic(restored.a2c_network, obs, tol))
        print(
            f"PASS {args.device} {dtype}: 196D -> 202D -> MLP[1024,1024,512,512] -> invariant value; optimizer/wrapper/checkpoints, max_error={error:.3g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
