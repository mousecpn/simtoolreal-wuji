"""Bimanual tool lifting: each Tianji/Wuji hand controls its own tool."""
import torch
from isaaclab.envs import DirectRLEnv

from ..simtoolreal.utils.logging_utils import log_step_metrics
from .rewards import compute_rewards, apply_wrench_dr
from .actions import apply_action_pipeline
from .config import TianjiWujiDualEnvCfg
from .observations import compute_obs_dim, compute_intermediate_values, build_observations, build_student_observations
from .reset import allocate_state_buffers, reset_env_state
from .scene import setup_scene, apply_physx_material_properties
from .termination import compute_terminations, update_tolerance_curriculum


class TianjiWujiDualEnv(DirectRLEnv):
    cfg: TianjiWujiDualEnvCfg

    def __init__(self, cfg: TianjiWujiDualEnvCfg, render_mode=None, **kwargs):
        if cfg.action_space != 54:
            raise ValueError("Dual Tianji/Wuji task requires 54 actions (left 7+20, right 7+20)")
        if cfg.reset.start_arm_higher:
            raise ValueError("start_arm_higher is a Kuka option; set robot.arm_default_joint_pos instead")
        cfg.observation_space = compute_obs_dim(cfg.obs.obs_list)
        cfg.state_space = compute_obs_dim(cfg.obs.state_list)
        super().__init__(cfg, render_mode, **kwargs)
        apply_physx_material_properties(self)
        allocate_state_buffers(self)

    def _setup_scene(self):
        setup_scene(self)

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        reset_env_state(self, torch.as_tensor(env_ids, device=self.device, dtype=torch.long))

    def _pre_physics_step(self, actions):
        apply_action_pipeline(self, actions)
        apply_wrench_dr(self)

    def _apply_action(self):
        self.robot.set_joint_position_target(self._cur_targets)

    def _get_dones(self):
        update_tolerance_curriculum(self)
        compute_intermediate_values(self)
        return compute_terminations(self)

    def _get_rewards(self):
        reward = compute_rewards(self)
        log_step_metrics(self)
        return reward

    def _get_observations(self):
        return build_observations(self)

    def get_student_obs(self):
        return build_student_observations(self)
