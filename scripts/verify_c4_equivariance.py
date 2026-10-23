"""Verify escnn mixed-irrep encoding, C4 LSTM, SE3 heads and checkpoints.

Run: .venv_isaacsim/bin/python scripts/verify_c4_equivariance.py
"""

import argparse
import io
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from escnn import nn as enn

from simtoolreal_shared.c4 import C4, C4LSTM, pack_regular, rotations
from simtoolreal_shared.c4_rl import FIELDS, C4Network


def transform_obs(obs, fields, k):
    parts = []
    for name, part in zip(fields, obs.split([FIELDS[f][0] for f in fields], -1)):
        kind = FIELDS[name][1]
        if kind == "vector":
            part = (
                part.reshape(*part.shape[:-1], -1, 3) @ rotations(part)[k].T
            ).flatten(-2)
        elif kind == "quat":
            x, y, z, w = part.unbind(-1)
            c, s = math.cos(k * math.pi / 4), math.sin(k * math.pi / 4)
            part = torch.stack(
                (c * x - s * y, s * x + c * y, c * z + s * w, c * w - s * z), -1
            )
        parts.append(part)
    return torch.cat(parts, -1)


def transform_action(action, k):
    vectors = action[..., :6].reshape(*action.shape[:-1], 2, 3)
    return torch.cat(
        ((vectors @ rotations(action)[k].T).flatten(-2), action[..., 6:]), -1
    )


def check_policy(network, obs, seq_length=1, atol=2e-6):
    batch = obs.shape[0] // seq_length
    rnn = network.policy.rnn
    state = tuple(
        torch.randn(
            rnn.num_layers,
            batch,
            4,
            rnn.hidden_channels,
            device=obs.device,
            dtype=obs.dtype,
        ).flatten(-2)
        for _ in range(2)
    )
    dones = torch.zeros(batch, seq_length, device=obs.device, dtype=torch.bool)
    if seq_length > 1:
        dones[0, 1] = True
    base = network(
        {"obs": obs, "rnn_states": state, "seq_length": seq_length, "dones": dones}
    )
    max_error = 0.0
    for k in range(4):
        rotated_state = tuple(
            s.reshape(rnn.num_layers, batch, 4, -1).roll(k, -2).flatten(-2)
            for s in state
        )
        rotated_obs = transform_obs(obs, network.encoder.fields, k)
        actual = network(
            {
                "obs": rotated_obs,
                "rnn_states": rotated_state,
                "seq_length": seq_length,
                "dones": dones,
            }
        )
        expected_std = base[1].clone()
        if k % 2:
            expected_std = expected_std[..., [1, 0, 2, 4, 3, 5] + list(range(6, 28))]
        pairs = [
            (actual[0], transform_action(base[0], k)),
            (actual[1], expected_std),
            (actual[2], base[2]),
        ]
        pairs += [
            (
                a.reshape(rnn.num_layers, batch, 4, -1),
                b.reshape(rnn.num_layers, batch, 4, -1).roll(k, -2),
            )
            for a, b in zip(actual[3], base[3])
        ]
        # Test the actual Gaussian log density under the action transform.
        action = torch.randn_like(base[0])
        lp = torch.distributions.Normal(base[0], base[1].exp()).log_prob(action).sum(-1)
        rlp = (
            torch.distributions.Normal(actual[0], actual[1].exp())
            .log_prob(transform_action(action, k))
            .sum(-1)
        )
        pairs.append((lp, rlp))
        for a, b in pairs:
            torch.testing.assert_close(a, b, atol=atol, rtol=atol)
            max_error = max(max_error, (a - b).abs().max().item())
    return max_error


def check_checkpoints(network, params, obs, tolerance):
    """Strict train/eval saves/loads must agree and refresh inference caches."""
    restored = C4Network(
        params, actions_num=28, input_shape=(obs.shape[-1],), num_seqs=3
    ).to(device=obs.device, dtype=obs.dtype)
    for source_train in (True, False):
        network.train(source_train)
        with torch.no_grad():
            reference = network({"obs": obs})[:3]
        state = network.state_dict()
        assert not any(k.endswith((".matrix", ".expanded_bias")) for k in state)
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        state = torch.load(buffer, map_location=obs.device, weights_only=True)
        for target_train in (True, False):
            restored.train(target_train)
            # Perturb learned weights to make any stale eval cache observable.
            with torch.no_grad():
                for p in restored.parameters():
                    p.add_(0.1)
            restored.train(target_train)
            restored.load_state_dict(state, strict=True)
            with torch.no_grad():
                actual = restored({"obs": obs})[:3]
            for a, b in zip(actual, reference):
                torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
    network.train()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--full-size", action="store_true", help="verify configured production widths"
    )
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(2)
    config = yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "isaacsimenvs/cfg/train/SimToolRealC4PPO.yaml"
        ).read_text()
    )
    params = config["params"]["network"]
    if not args.full_size:
        params["hidden_channels"] = 12
        params["input_channels"] = 9
        params["mlp_units"] = [48, 48, 24, 24]
    params["layers"] = 2
    n = sum(FIELDS[f][0] for f in params["obs_fields"])
    for dtype, tol in [(torch.float64, 1e-10), (torch.float32, 2e-5)]:
        rnn = C4LSTM(7, 9, 2).to(device=args.device, dtype=dtype)
        assert all(
            isinstance(m.linear, enn.Linear) for m in (*rnn.inputs, *rnn.recurrents)
        )
        x = torch.randn(5, 3, 4, 7, device=args.device, dtype=dtype, requires_grad=True)
        state = tuple(
            torch.randn(2, 3, 4, 9, device=args.device, dtype=dtype) for _ in range(2)
        )
        dones = torch.zeros(5, 3, device=args.device, dtype=torch.bool)
        dones[2, 0] = True
        y, s = rnn(x, state, dones)
        for k in range(4):
            # Independently verify escnn field-major packing agrees with roll.
            rho = torch.as_tensor(
                rnn.state_type.representation(C4.element(k)),
                device=args.device,
                dtype=dtype,
            )
            torch.testing.assert_close(
                pack_regular(state[0]) @ rho.T,
                pack_regular(state[0].roll(k, -2)),
                atol=tol,
                rtol=tol,
            )
            ry, rs = rnn(x.roll(k, -2), tuple(v.roll(k, -2) for v in state), dones)
            torch.testing.assert_close(ry, y.roll(k, -2), atol=tol, rtol=tol)
            for a, b in zip(rs, s):
                torch.testing.assert_close(a, b.roll(k, -2), atol=tol, rtol=tol)
        y1, s1 = rnn(x[:2], state, dones[:2])
        y2, s2 = rnn(x[2:], s1, dones[2:])
        torch.testing.assert_close(torch.cat((y1, y2)), y)
        for a, b in zip(s2, s):
            torch.testing.assert_close(a, b)
        reset_y, _ = rnn(x[2:, 0:1])
        torch.testing.assert_close(y[2:, 0:1], reset_y, atol=tol, rtol=tol)
        y.square().sum().backward()
        assert torch.isfinite(x.grad).all() and all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in rnn.parameters()
        )
        network = C4Network(params, actions_num=28, input_shape=(n,), num_seqs=3).to(
            device=args.device, dtype=dtype
        )
        with torch.no_grad():
            network.policy.log_std.uniform_(-0.5, 0.5)
        obs = torch.randn(15, n, device=args.device, dtype=dtype)
        features = network.encoder(obs)
        assert features.shape == (15, 180)
        assert network.encoder.out_type == network.policy.input_projection.in_type
        assert network.policy.input_projection.out_type.size == 4 * params.get(
            "input_channels", params["hidden_channels"]
        )
        projected = network.policy.input_projection(features)
        for k in range(4):
            rotated = network.encoder(transform_obs(obs, params["obs_fields"], k))
            rho_in = torch.as_tensor(
                network.encoder.out_type.representation(C4.element(k)),
                device=args.device,
                dtype=dtype,
            )
            rho_out = torch.as_tensor(
                network.policy.latent_type.representation(C4.element(k)),
                device=args.device,
                dtype=dtype,
            )
            torch.testing.assert_close(rotated, features @ rho_in.T, atol=tol, rtol=tol)
            torch.testing.assert_close(
                network.policy.input_projection(rotated),
                projected @ rho_out.T,
                atol=tol,
                rtol=tol,
            )
        # Quaternion signs must not affect features.
        parts = list(obs.split(network.encoder.widths, -1))
        flipped = torch.cat(
            [
                -p if FIELDS[f][1] == "quat" else p
                for f, p in zip(params["obs_fields"], parts)
            ],
            -1,
        )
        torch.testing.assert_close(network.encoder(flipped), features)
        error = check_policy(network, obs, 5, tol)
        opt = torch.optim.Adam(network.parameters(), lr=1e-3)
        mu, std, value, _ = network({"obs": obs, "seq_length": 5})
        (mu.square().mean() + std.square().mean() + value.square().mean()).backward()
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in network.parameters()
        )
        opt.step()
        check_policy(network, obs, 5, tol)
        network.eval()
        check_policy(network, obs, 5, tol)
        check_checkpoints(network, params, obs, tol)
        print(
            f"PASS {dtype}: escnn 180D mixed irreps -> regular, 2-layer LSTM h/c, nonzero state, reset, chunking, quaternion signs, action/value/Gaussian, optimizer, train/eval checkpoints; max_error={error:.3g}"
        )


if __name__ == "__main__":
    main()
