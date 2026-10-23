"""Compare policy parameters and single-step dense matrix FLOPs (no simulator).

Run: .venv_isaacsim/bin/python scripts/compare_c4_policy_cost.py
One MAC = two FLOPs; excludes bias, activations, normalization, encoding,
sampling, backward and environment/IK. Eval escnn matrices are cached.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rl_games"))

import torch
import yaml
from rl_games.algos_torch.network_builder import A2CBuilder
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from simtoolreal_shared.c4 import EquivariantLinear
from simtoolreal_shared.c4_rl import FIELDS, C4Network


class DenseControl(nn.Module):
    """Same encoded inputs and scalar hidden width, without C4 constraints."""

    def __init__(self, encoder, params):
        super().__init__()
        self.encoder = encoder
        width = 4 * params["hidden_channels"]
        inputs = 4 * params.get("input_channels", params["hidden_channels"])
        self.input_projection = nn.Linear(encoder.out_type.size, inputs)
        self.rnn = nn.LSTM(inputs, width, params.get("layers", 1))
        self.layer_norm = (
            nn.LayerNorm(width) if params.get("layer_norm", False) else nn.Identity()
        )
        modules = []
        for out in params.get("mlp_units", ()):
            modules.extend((nn.Linear(width, out), nn.ELU()))
            width = out
        self.mlp = nn.Sequential(*modules)
        self.actor = nn.Linear(width, 28)
        self.critic = nn.Linear(width, 1)
        self.log_std = nn.Parameter(torch.zeros(28))

    def forward(self, obs):
        x = self.input_projection(self.encoder(obs)).unsqueeze(0)
        y, state = self.rnn(x)
        y = self.mlp(self.layer_norm(y))
        mean = self.actor(y)
        return mean, self.log_std + torch.zeros_like(mean), self.critic(y), state


def measure(model, call):
    model.eval()  # escnn expands and caches its matrices here, outside profiling.
    macs = {}
    handles = []

    def hook(name, module, args, output):
        if isinstance(module, EquivariantLinear):
            n = args[0].numel() // module.in_type.size
            count = n * module.in_type.size * module.out_type.size
        elif isinstance(module, nn.Linear):
            count = (args[0].numel() // module.in_features) * module.weight.numel()
        else:  # nn.LSTM: both input and recurrent matrix products, every step.
            count = 0
            steps = args[0].shape[0] * args[0].shape[1]
            for key, parameter in module.named_parameters():
                if key.startswith("weight_"):
                    count += steps * parameter.numel()
        macs[name] = macs.get(name, 0) + count

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.LSTM, EquivariantLinear)):
            handles.append(
                module.register_forward_hook(
                    lambda m, a, o, name=name: hook(name, m, a, o)
                )
            )
    try:
        # Unfused CPU LSTM exposes its GEMMs to the independent torch counter.
        with torch.no_grad(), torch.backends.mkldnn.flags(enabled=False):  # noqa: SIM117
            with FlopCounterMode(display=False) as counter:
                output = call()
        measured = counter.get_total_flops()
    finally:
        for handle in handles:
            handle.remove()
    expected = 2 * sum(macs.values())
    if measured != expected:
        raise AssertionError(
            f"matrix FLOP mismatch: hooks={expected}, torch={measured}"
        )

    def finite(tree):
        if isinstance(tree, torch.Tensor):
            assert torch.isfinite(tree).all()
        elif isinstance(tree, (tuple, list)):
            for item in tree:
                finite(item)

    finite(output)
    return {
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "parameter_groups": {
            name: sum(p.numel() for p in child.parameters() if p.requires_grad)
            for name, child in model.named_children()
        },
        "direct_parameters": {
            name: p.numel()
            for name, p in model.named_parameters(recurse=False)
            if p.requires_grad
        },
        "matrix_macs_per_step": sum(macs.values()),
        "matrix_flops_per_step": expected,
        "torch_flop_counter": measured,
        "macs_by_module": macs,
    }


def baseline(params, width):
    builder = A2CBuilder()
    builder.load(params)
    return builder.build(
        "a2c", actions_num=28, input_shape=(width,), num_seqs=1, value_size=1
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/c4_policy_cost.json"
    )
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(1)
    config = yaml.safe_load(
        (ROOT / "isaacsimenvs/cfg/train/SimToolRealC4PPO.yaml").read_text()
    )
    params = config["params"]["network"]
    raw_width = sum(FIELDS[f][0] for f in params["obs_fields"])
    c4 = C4Network(params, actions_num=28, input_shape=(raw_width,))
    dense = DenseControl(c4.encoder, params)
    original_cfg = yaml.safe_load(
        (ROOT / "isaacsimenvs/cfg/train/SimToolRealPPO.yaml").read_text()
    )["params"]
    task = yaml.safe_load((ROOT / "isaacsimenvs/cfg/task/SimToolReal.yaml").read_text())
    obs_cfg = task["obs"]
    original_width = sum(FIELDS[f][0] for f in obs_cfg["obs_list"])
    state_width = sum(FIELDS[f][0] for f in obs_cfg["state_list"])
    original = baseline(original_cfg["network"], original_width)
    central = baseline(
        original_cfg["config"]["central_value_config"]["network"], state_width
    )
    obs = torch.randn(1, raw_width)

    def run_original(net, width):
        return net(
            {
                "obs": torch.randn(1, width),
                "rnn_states": net.get_default_rnn_state(),
                "seq_length": 1,
            }
        )

    report = {
        "scope": "batch=1, sequence=1, FP32 eval forward, actor+value; dense matrix FLOPs only; 1 MAC=2 FLOPs",
        "excluded": [
            "bias and pointwise operations",
            "normalization",
            "observation encoding",
            "Gaussian sampling",
            "backward",
            "escnn train-mode basis expansion",
            "environment and IK",
        ],
        "dimensions": {
            "c4_raw_input": raw_width,
            "encoded_input": c4.encoder.out_type.size,
            "original_input": original_width,
            "original_central_input": state_width,
            "c4_scalar_hidden": 4 * params["hidden_channels"],
        },
        "c4_escnn": measure(c4, lambda: c4({"obs": obs})),
        "dense_same_width": measure(dense, lambda: dense(obs)),
        "original_policy": measure(
            original, lambda: run_original(original, original_width)
        ),
        "original_extra_central_critic": measure(
            central, lambda: run_original(central, state_width)
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
