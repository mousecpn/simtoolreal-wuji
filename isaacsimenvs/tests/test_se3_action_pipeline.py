"""GPU smoke: 28-action SE3 policy, palm Jacobian, limits, and reset buffers."""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym
import torch
import yaml
import isaacsimenvs  # noqa: F401
from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
from isaacsimenvs.tasks.simtoolreal.utils.action_utils import damped_ik

cfg = SimToolRealEnvCfg()
with open(gym.spec('Isaacsimenvs-SimToolReal-SE3-Direct-v0').kwargs['env_cfg_yaml_entry_point']) as f:
    cfg.from_dict(yaml.safe_load(f))
cfg.scene.num_envs = 6
cfg.assets.num_assets_per_type = 1
cfg.domain_randomization.use_action_delay = False
env = gym.make('Isaacsimenvs-SimToolReal-SE3-Direct-v0', cfg=cfg)
inner = env.unwrapped
obs, _ = env.reset()
assert cfg.action_space == 28
assert inner._cur_targets.shape == (6, 29)
assert inner._action_queue.shape[-1] == 28
# Known analytic inverse and singularity behavior on the actual rollout device.
j = torch.eye(6, 7, device=inner.device).expand(6, -1, -1)
d = torch.randn(6, 6, device=inner.device)
assert torch.allclose(damped_ik(j, d, 0.05)[:, :6], d / 1.0025, atol=1e-6)
assert torch.isfinite(damped_ik(torch.zeros_like(j), d, 0.05)).all()
for step in range(32):
    action = torch.rand(6, 28, device=inner.device) * 2 - 1
    # Exercise each Cartesian axis independently across environments.
    action[:, :6] = torch.eye(6, device=inner.device) * (1 if step % 2 else -1)
    before = inner.robot.data.joint_pos[:, inner._arm_joint_ids].clone()
    obs, reward, terminated, truncated, _ = env.step(action)
    targets = inner._cur_targets[:, inner._arm_joint_ids]
    assert torch.isfinite(targets).all()
    assert (targets >= inner._arm_lower).all() and (targets <= inner._arm_upper).all()
    active = ~(terminated | truncated)
    if active.any():
        assert (targets[active] - before[active]).abs().max() <= cfg.action.ik_max_joint_delta + 1e-5
    assert torch.isfinite(reward).all()
    assert all(torch.isfinite(v).all() for v in obs.values())
    if step == 16:
        env.reset()
print('[SE3 smoke] PASS: nonzero 6-axis rollout, DLS, 28/29 buffers, limits, finite obs/rewards, reset', flush=True)
env.close()
app.close()
os._exit(0)
