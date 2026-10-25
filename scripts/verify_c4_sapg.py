"""Verify C4 SAPG block conditioning, equivariance, checkpoints and a CPU epoch.

The training smoke uses synthetic observations to exercise the actual SAPG
algorithm and independent critic without starting Isaac Sim.
"""

import argparse
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rl_games"))

import gym
import numpy as np
import torch
import yaml

from rl_games.algos_torch.model_builder import ModelBuilder
from rl_games.common import env_configurations, vecenv
from rl_games.common.ivecenv import IVecEnv
from rl_games.torch_runner import Runner
from scripts.verify_c4_equivariance import transform_action, transform_obs
from simtoolreal_shared.c4_rl import FIELDS


def size(fields):
    return sum(FIELDS[f][0] for f in fields)


def small(params):
    params = copy.deepcopy(params)
    params["network"].update(hidden_channels=8, input_channels=9,
                             mlp_units=[32, 32, 16, 16])
    params["config"]["central_value_config"]["network"]["mlp_units"] = [32, 32, 16, 16]
    return params


def verify(params):
    ids = torch.linspace(50., 0., 6)
    actor_fields = params["network"]["obs_fields"]
    critic_params = params["config"]["central_value_config"]
    critic_fields = critic_params["network"]["obs_fields"]

    def build(p, fields):
        return ModelBuilder().load(p).build(dict(
            actions_num=28, input_shape=(size(fields) + 1,), num_seqs=6,
            value_size=1, normalize_input=False, normalize_value=True,
            type="extra_param", coef_ids=ids, coef_id_idx=size(fields),
        ))

    actor, critic = build(params, actor_fields), build(critic_params, critic_fields)
    obs = torch.cat((torch.randn(6, size(actor_fields)), ids[:, None]), -1)
    states = torch.cat((torch.randn(6, size(critic_fields)), ids[:, None]), -1)
    net = actor.a2c_network
    # Make all six Gaussian blocks observably different.
    with torch.no_grad():
        net.policy.log_std.copy_(torch.arange(6)[:, None] * .03)
    for model, inputs, fields in ((actor, obs, actor_fields), (critic, states, critic_fields)):
        network = model.a2c_network
        model.eval()
        reference = network({"obs": inputs})
        for k in range(4):
            rotated = torch.cat((transform_obs(inputs[:, :-1], fields, k), inputs[:, -1:]), -1)
            actual = network({"obs": rotated})
            if model is actor:
                torch.testing.assert_close(actual[0], transform_action(reference[0], k), atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(actual[1], reference[1], atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(actual[2], reference[2], atol=2e-5, rtol=2e-5)
            else:
                torch.testing.assert_close(actual[0], reference[0], atol=2e-5, rtol=2e-5)
        # Check train/eval serialization, including learned embeddings and sigmas.
        weights = copy.deepcopy(model.state_dict())
        for training in (True, False):
            model.train(training)
            model.load_state_dict(weights, strict=True)
            actual = network({"obs": inputs})
            for a, b in zip(actual[:3] if model is actor else actual[:1],
                            reference[:3] if model is actor else reference[:1]):
                torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        result = model({"obs": inputs, "is_train": True,
                        "prev_actions": torch.zeros(inputs.shape[0], 28)})
        loss = result["values"].square().mean()
        if model is actor:
            loss = loss + result["mus"].square().mean() + result["sigmas"].mean()
        optimizer.zero_grad()
        loss.backward()
        grad = network.conditioning.extra_params.grad
        assert grad is not None and torch.isfinite(grad).all() and (grad.abs().sum(-1) > 0).all()
        if model is actor:
            assert (network.policy.log_std.grad.abs().sum(-1) > 0).all()
        optimizer.step()
        assert all(torch.isfinite(p).all() for p in model.parameters())
    log_std = net({"obs": obs})[1]
    torch.testing.assert_close(log_std[:, 0], log_std[:, 1])
    torch.testing.assert_close(log_std[:, 3], log_std[:, 4])
    assert not torch.equal(log_std[0], log_std[-1])
    print("PASS C4 SAPG: six conditioned embeddings/sigmas, actor equivariance, critic invariance, gradients and strict checkpoints", flush=True)


class SyntheticEnv(IVecEnv):
    def __init__(self, config_name, num_actors, **kwargs):
        self.n = num_actors
        self.t = 0
        self.env = SimpleNamespace(device="cpu")

    def get_env_info(self):
        return dict(observation_space=gym.spaces.Box(-np.inf, np.inf, (174,), dtype=np.float32),
                    state_space=gym.spaces.Box(-np.inf, np.inf, (196,), dtype=np.float32),
                    action_space=gym.spaces.Box(-1., 1., (28,), dtype=np.float32),
                    agents=1, value_size=1)

    def reset(self):
        return {"obs": torch.randn(self.n, 174), "states": torch.randn(self.n, 196)}

    def step(self, actions):
        assert actions.shape == (self.n, 28) and torch.isfinite(actions).all()
        self.t += 1
        return self.reset(), -actions.square().mean(-1), torch.full((self.n,), self.t % 3 == 0), {}

    def get_env_state(self):
        return None

    def set_env_state(self, state):
        pass


def train_epoch(params):
    vecenv.register("C4SAPGCheck", lambda config_name, num_actors, **kw: SyntheticEnv(config_name, num_actors, **kw))
    env_configurations.register("c4_sapg_check", dict(vecenv_type="C4SAPGCheck", env_creator=lambda **kw: None))
    params = small(params)
    c = params["config"]
    c.update(name="0_c4_sapg_check", full_experiment_name="0_c4_sapg_check",
             env_name="c4_sapg_check", device="cpu", device_name="cpu",
             num_actors=12, expl_coef_block_size=2, max_epochs=1,
             horizon_length=4, seq_length=4, minibatch_size=8, mini_epochs=1,
             save_frequency=1)
    c["central_value_config"].update(minibatch_size=8, mini_epochs=1)
    with tempfile.TemporaryDirectory(prefix="c4-sapg-check-") as directory:
        c["train_dir"] = directory
        runner = Runner()
        runner.load({"params": params})
        runner.run({"train": True, "play": False})
        checkpoints = list(Path(directory).rglob("*.pth"))
        assert checkpoints, "training did not save a checkpoint"
        checkpoint = torch.load(checkpoints[0], map_location="cpu", weights_only=False)[0]
        assert checkpoint["epoch"] >= 1
        assert "assymetric_vf_nets" in checkpoint
        for section in ("model", "assymetric_vf_nets"):
            assert all(torch.isfinite(v).all() for v in checkpoint[section].values() if isinstance(v, torch.Tensor))
    print("PASS actual SAPG epoch: six blocks, leader/follower sharing, actor and asymmetric critic updates, finite saved checkpoint", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-size", action="store_true")
    parser.add_argument("--torch-rollout", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(2)
    params = yaml.safe_load((ROOT / "isaacsimenvs/cfg/train/SimToolRealC4SAPG.yaml").read_text())["params"]
    if args.torch_rollout:
        params["network"]["torch_rollout"] = True
        params["config"]["central_value_config"]["network"]["torch_rollout"] = True
    verify(params if args.full_size else small(params))
    train_epoch(params)


if __name__ == "__main__":
    main()
