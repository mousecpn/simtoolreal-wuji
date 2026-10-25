"""Check frozen Torch export, cache invalidation and recurrent state parity.

Run with --full-size --checkpoint <SAPG .pth> to check a production checkpoint.
This checks numerical correctness; timings on CPU do not predict GPU latency.
"""

import argparse
import copy
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rl_games"))

import torch
import yaml
from escnn import nn as enn

from simtoolreal_shared.c4 import (
    EquivariantLinear, TorchLSTM, mixed_type, regular_type,
)
from simtoolreal_shared.c4_rl import C4Builder, FIELDS
from scripts.verify_c4_equivariance import transform_action, transform_obs


def assert_tree(actual, expected, tol=2e-5):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    elif expected is None:
        assert actual is None
    else:
        for a, b in zip(actual, expected):
            assert_tree(a, b, tol)


def cache_check():
    linear = EquivariantLinear(mixed_type(3, 2), regular_type(4))
    x = torch.randn(5, 9)
    with patch.object(linear, "expand_parameters", wraps=linear.expand_parameters) as expand:
        with torch.no_grad():
            for _ in range(20):
                linear.eval()
            assert expand.call_count == 1
            reference = enn.Linear.forward(linear, enn.GeometricTensor(x, linear.in_type)).tensor
            torch.testing.assert_close(linear(x), reference, atol=0, rtol=0)
            linear.weights.add_(.01)
            linear.eval()
            assert expand.call_count == 2
        linear.train()
        linear(x).square().mean().backward()
        assert linear.weights.grad is not None
        assert torch.isfinite(linear.weights.grad).all()
        linear.eval()
        count = expand.call_count
        weights = copy.deepcopy(linear.state_dict())
        with torch.no_grad():
            reference = linear(x).clone()
            linear.weights.add_(.1)
        linear.load_state_dict(weights, strict=True)
        assert expand.call_count == count + 1
        torch.testing.assert_close(linear(x), reference)
        assert not any(k in weights for k in ("matrix", "expanded_bias"))
        count = expand.call_count
        linear.double().eval()
        assert expand.call_count == count + 1
        with torch.no_grad():
            matrix, bias = linear.expand_parameters()
            torch.testing.assert_close(linear(x.double()), torch.nn.functional.linear(x.double(), matrix, bias))
        # escnn shares basis samplers across layers; restore their dtype too.
        linear.float()
    print("PASS eval cache: 20 eval calls expand once; refresh after weight update, train/eval and checkpoint load", flush=True)


@torch.no_grad()
def check_network(network, device, critic=False):
    n = network.encoder.raw_size
    conditioning = network.conditioning
    obs = torch.randn(12, n, device=device)
    if conditioning is not None:
        obs = torch.cat((obs, conditioning.coef_ids.repeat(2)[:, None]), -1)
    network.train()
    keys_before = set(network.state_dict())
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device) if device.startswith("cuda") else None
    exported = network.export_for_rollout()
    assert network.training
    assert set(network.state_dict()) == keys_before
    assert torch.equal(rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state(device))
    assert all(not p.requires_grad for p in exported.parameters())
    assert not any(isinstance(m, enn.EquivariantModule) for m in exported.modules())
    if not critic:
        assert isinstance(exported.policy.rnn, TorchLSTM)
        assert isinstance(exported.policy.rnn.rnn, torch.nn.LSTM)
    network.eval()
    if critic:
        assert_tree(exported({"obs": obs}), network({"obs": obs}))
    else:
        for length in (1, 4):
            batch = len(obs) // length
            rnn = network.policy.rnn
            states = tuple(torch.randn(rnn.num_layers, batch, 4 * rnn.hidden_channels,
                                       device=device) for _ in range(2))
            for mask in (False, True):
                inputs = dict(obs=obs, seq_length=length, rnn_states=states)
                if mask:
                    dones = torch.zeros(len(obs), dtype=torch.bool, device=device)
                    dones[1::3] = True
                    inputs["dones"] = dones
                actual, expected = exported(inputs), network(inputs)
                assert_tree(actual, expected)
    reference = exported({"obs": obs})
    fields = network.encoder.fields
    for k in range(4):
        raw = transform_obs(obs[:, :n], fields, k)
        rotated = torch.cat((raw, obs[:, n:]), -1)
        actual = exported({"obs": rotated})
        if critic:
            assert_tree(actual, reference)
        else:
            assert_tree(actual[0], transform_action(reference[0], k))
            assert_tree(actual[1:3], reference[1:3])
    # Frozen snapshots remain fixed; refreshing after an update captures new weights.
    with torch.no_grad():
        next(network.parameters()).add_(.03)
    assert_tree(exported({"obs": obs}), reference)
    fresh = network.export_for_rollout()
    # Force a proper train/eval transition to refresh the reference's dense cache.
    network.train().eval()
    assert_tree(fresh({"obs": obs}), network({"obs": obs}))
    # Exercise the optional automatic sampler path used by rl_games.
    keys = set(network.state_dict())
    network.torch_rollout = True
    network.eval()
    expected = fresh({"obs": obs})
    assert_tree(network({"obs": obs}), expected)
    snapshot = network.__dict__["_torch_rollout_snapshot"]
    for _ in range(3):
        network.eval()
        assert_tree(network({"obs": obs}), expected)
        assert network.__dict__["_torch_rollout_snapshot"] is snapshot
    assert set(network.state_dict()) == keys
    network.to(device)
    assert network.__dict__["_torch_rollout_snapshot"] is None
    assert_tree(network({"obs": obs}), expected)
    weights = copy.deepcopy(network.state_dict())
    network.load_state_dict(weights, strict=True)
    assert network.__dict__["_torch_rollout_snapshot"] is None
    assert_tree(network({"obs": obs}), expected)
    network.train()
    assert network.__dict__["_torch_rollout_snapshot"] is None
    print(f"PASS {'critic' if critic else 'actor'} Torch export: outputs/states/reset, C4 transforms, frozen snapshot and refresh, RNG and source checkpoint keys", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--full-size", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cache_check()
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if 0 in checkpoint:
            checkpoint = checkpoint[0]
        assert args.full_size, "a production checkpoint needs --full-size"
    for sapg in (False, True):
        params = yaml.safe_load((ROOT / "isaacsimenvs/cfg/train" /
                                ("SimToolRealC4SAPG.yaml" if sapg else "SimToolRealC4PPO.yaml")).read_text())["params"]
        for critic in (False, True):
            p = params["config"]["central_value_config"]["network"] if critic else params["network"]
            if not args.full_size:
                p.update(hidden_channels=8, input_channels=9, layers=2,
                         mlp_units=[32, 32, 16, 16])
            n = sum(FIELDS[f][0] for f in p["obs_fields"])
            kwargs = dict(actions_num=28, input_shape=(n + int(sapg),), num_seqs=12,
                          value_size=1, normalize_input=False)
            if sapg:
                kwargs.update(type="extra_param", coef_ids=torch.linspace(50., 0., 6), coef_id_idx=n)
            builder = C4Builder()
            builder.load(p)
            network = builder.build("a2c", **kwargs)
            if checkpoint is not None and sapg:
                section = checkpoint["assymetric_vf_nets"] if critic else checkpoint["model"]
                prefix = (
                    "model.a2c_network." if any(
                        k.startswith("model.a2c_network.") for k in section
                    ) else "a2c_network."
                )
                weights = {k.removeprefix(prefix): v for k, v in section.items()
                           if k.startswith(prefix)}
                network.load_state_dict(weights, strict=True)
                print(f"Loaded real checkpoint {'critic' if critic else 'actor'}, epoch {checkpoint['epoch']}", flush=True)
            check_network(network.to(device=args.device, dtype=torch.float32), args.device, critic)


if __name__ == "__main__":
    main()
