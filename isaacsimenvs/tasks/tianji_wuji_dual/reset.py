"""State allocation and reset helpers for SimToolReal."""

from __future__ import annotations

import torch

from isaaclab.utils.math import random_orientation

from ..simtoolreal.utils.action_utils import sample_log_uniform
from ..simtoolreal.utils.goal_sampling import sample_absolute_goal_pose, sample_delta_goal_pose
from .observations import KEYPOINT_CORNERS, NUM_FINGERTIPS
from .scene import (
    ARM_JOINT_REGEX,
    FINGERTIP_BODY_REGEX,
    HAND_JOINT_REGEX,
    JOINT_NAMES_CANONICAL,
    PALM_BODY_NAMES,
    FINGERTIP_LINK_NAMES,
)

def allocate_state_buffers(env) -> None:
    """Populate every per-env buffer + index cache used by the hooks.

    Called once from ``__init__`` after ``super().__init__`` (which runs
    ``_setup_scene`` and makes ``env.robot``/``env.object``/``env.goal_viz``
    available).
    """
    dr = env.cfg.domain_randomization
    rew = env.cfg.reward

    # --- Joint/body id caches ---
    env._arm_joint_ids = env.robot.find_joints(ARM_JOINT_REGEX)[0]      # 14
    env._hand_joint_ids = env.robot.find_joints(HAND_JOINT_REGEX)[0]     # 40
    env._palm_body_ids = [env.robot.find_bodies(name)[0][0] for name in PALM_BODY_NAMES]
    env._fingertip_body_ids = [env.robot.find_bodies(name)[0][0] for name in FINGERTIP_LINK_NAMES]  # 10
    assert len(env._fingertip_body_ids) == NUM_FINGERTIPS

    # Convert between Lab parser order and canonical policy order.
    lab_names = list(env.robot.data.joint_names)
    assert set(lab_names) == set(JOINT_NAMES_CANONICAL)
    env._perm_canon_to_lab = torch.tensor(
        [JOINT_NAMES_CANONICAL.index(n) for n in lab_names],
        device=env.device, dtype=torch.long,
    )
    env._perm_lab_to_canon = torch.tensor(
        [lab_names.index(n) for n in JOINT_NAMES_CANONICAL],
        device=env.device, dtype=torch.long,
    )

    limits = env.robot.data.joint_pos_limits  # (N, num_joints, 2), Lab order

    # Canonical-order limits for normalizing joint_pos observations.
    env._joint_lower_canon = limits[0, :, 0][env._perm_lab_to_canon]  # (54,)
    env._joint_upper_canon = limits[0, :, 1][env._perm_lab_to_canon]  # (54,)

    # Lab-order limits for action target clamping.
    env._arm_lower = limits[:, env._arm_joint_ids, 0]
    env._arm_upper = limits[:, env._arm_joint_ids, 1]
    env._hand_lower = limits[:, env._hand_joint_ids, 0]
    env._hand_upper = limits[:, env._hand_joint_ids, 1]

    # --- Action target buffers  ---
    action_space = env.cfg.action_space
    env._cur_targets = torch.zeros(env.num_envs, action_space, device=env.device)
    env._prev_targets = torch.zeros(env.num_envs, action_space, device=env.device)

    # --- Keypoint offsets in object local frame ---
    corners = torch.tensor(
        KEYPOINT_CORNERS, device=env.device, dtype=torch.float32
    )  # (4, 3)

    env._tool_scales = torch.stack((env._object_scale_per_env, env._object_scale_per_env2), dim=1)
    env._keypoint_offsets = (
        corners.reshape(1, 1, 4, 3)
        * (env._tool_scales * rew.object_base_size * rew.keypoint_scale * 0.5).unsqueeze(2)
    )  # (N, 2, 4, 3)

    # Fixed-size reward keypoints follow legacy: fixed_size * keypoint_scale / 2.
    env._keypoint_offsets_fixed = (
        corners
        * (0.5 * rew.keypoint_scale * torch.tensor(rew.fixed_size, device=env.device)).unsqueeze(0)
    ).reshape(1, 1, 4, 3).expand(env.num_envs, 2, -1, -1).contiguous()

    # --- Per-env DR priors (re-sampled on reset; seeded once here) ---
    lo, hi = dr.object_scale_noise_multiplier_range
    env._object_scale_multiplier = torch.empty(
        env.num_envs, 2, 3, device=env.device
    ).uniform_(lo, hi)

    # --- Reward / termination trackers  ---
    # whether the object is lifted
    env._lifted_object = torch.zeros(
        env.num_envs, 2, dtype=torch.bool, device=env.device
    )
    # the maximum distance between the object and the goal since the last goal reset
    env._closest_keypoint_max_dist = torch.full(
        (env.num_envs, 2), -1.0, device=env.device
    )
    # the minimum distance between a fingertip and the object since the last goal reset
    env._closest_fingertip_dist = torch.full(
        (env.num_envs, NUM_FINGERTIPS), -1.0, device=env.device
    )
    # the number of succeesses in the current episode
    env._successes = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    # the number of consecutive steps that the object is near the goal
    env._near_goal_steps = torch.zeros(
        env.num_envs, 2, dtype=torch.long, device=env.device
    )

    # --- Tolerance curriculum state ---
    env._current_success_tolerance: float = env.cfg.termination.success_tolerance
    env._prev_episode_successes = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    env._frame_counter: int = 0
    env._last_curriculum_update: int = 0

    # --- Object lifted-reward reference z (updated on each _reset_object_pose) ---
    init_z = env.cfg.reset.table_reset_z + env.cfg.reset.table_object_z_offset
    env._object_init_z = torch.full(
        (env.num_envs, 2), init_z, device=env.device
    )

    # --- Per-env table surface z (randomized in _reset_table_pose) ---
    env._table_z_per_env = torch.full(
        (env.num_envs,), env.cfg.reset.table_reset_z, device=env.device
    )

    # --- DR rolling buffers ---
    env._object_state_queue = torch.zeros(
        env.num_envs,
        max(1, dr.object_state_delay_max),
        26,  # two independent pos(3) + quat(4) + lin_vel(3) + ang_vel(3)
        device=env.device,
    )
    env._obs_queue = torch.zeros(
        env.num_envs,
        max(1, dr.obs_delay_max),
        env.cfg.observation_space,
        device=env.device,
    )
    env._action_queue = torch.zeros(
        env.num_envs,
        max(1, dr.action_delay_max),
        action_space,
        device=env.device,
    )

    # --- Wrench DR state (Phase C/D) ---
    env._random_force_prob = sample_log_uniform(
        dr.force_prob_range, env.num_envs
    ).to(env.device)
    env._random_torque_prob = sample_log_uniform(
        dr.torque_prob_range, env.num_envs
    ).to(env.device)
    env._object_forces = torch.zeros(env.num_envs, 2, 3, device=env.device)
    env._object_torques = torch.zeros(env.num_envs, 2, 3, device=env.device)
    env._object_mass = torch.stack([obj.data.default_mass[:, 0] for obj in env.objects], dim=1).to(env.device)  # (N, 2)

    # --- Step-shared caches populated by compute_intermediate_values (Phase F) ---
    env._keypoints_max_dist = torch.zeros(env.num_envs, 2, device=env.device)
    env._curr_fingertip_distances = torch.zeros(
        env.num_envs, NUM_FINGERTIPS, device=env.device
    )
    env._near_goal = torch.zeros(
        env.num_envs, 2, dtype=torch.bool, device=env.device
    )
    env._tool_success = torch.zeros(env.num_envs, 2, dtype=torch.bool, device=env.device)
    env._is_success = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
    )

    # A single-object fixed pose/trajectory has no unambiguous two-tool interpretation.
    for name in ("fixed_trajectory_file", "fixed_start_pose", "fixed_goal_pose"):
        if getattr(env.cfg.reset, name, None):
            raise ValueError(f"Dual two-tool task does not support reset.{name}; use independent sampled goals")

    # set the reward buffer to 0
    env.reward_buf = torch.zeros(env.num_envs, device=env.device)


def _randomize_robot_dof_state(env, env_ids: torch.Tensor) -> None:
    """Reset DOF state and seed previous targets from the reset pose."""
    cfg = env.cfg.reset
    default_pos = env.robot.data.default_joint_pos[env_ids]  # (n, num_dofs)
    lower = env.robot.data.joint_pos_limits[env_ids, :, 0]
    upper = env.robot.data.joint_pos_limits[env_ids, :, 1]

    reset_scale = torch.zeros_like(default_pos)
    reset_scale[:, env._arm_joint_ids] = cfg.reset_dof_pos_random_interval_arm
    reset_scale[:, env._hand_joint_ids] = cfg.reset_dof_pos_random_interval_fingers

    sampled_pos = lower + (upper - lower) * torch.rand_like(default_pos)
    joint_pos = torch.lerp(default_pos, sampled_pos, reset_scale).clamp(lower, upper)
    joint_vel = torch.empty_like(default_pos).uniform_(
        -cfg.reset_dof_vel_random_interval,
        cfg.reset_dof_vel_random_interval,
    )

    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    env._prev_targets[env_ids] = joint_pos
    env._cur_targets[env_ids] = joint_pos


def _reset_table_pose(env, env_ids: torch.Tensor) -> None:
    """Randomize the table's pose per env and write the new transform.

    Z position is required (the policy uses `_table_z_per_env` to set the
    object init height). XY position and yaw are gated on
    `table_reset_xy_range_m` / `table_reset_yaw_range_deg` — default ranges
    are zero so this is a no-op for runs that don't opt in.
    """
    cfg = env.cfg.reset
    n = env_ids.numel()
    env_origins = env.scene.env_origins[env_ids]

    dz = torch.empty(n, device=env.device).uniform_(
        -cfg.table_reset_z_range, cfg.table_reset_z_range
    )
    table_z = cfg.table_reset_z + dz
    env._table_z_per_env[env_ids] = table_z

    pos_local = torch.zeros(n, 3, device=env.device)
    pos_local[:, 2] = table_z

    # XY position noise — per-env independent uniform half-widths.
    xy_range = tuple(float(v) for v in cfg.table_reset_xy_range_m)
    if xy_range[0] > 0.0 or xy_range[1] > 0.0:
        rx = torch.empty(n, device=env.device).uniform_(-xy_range[0], xy_range[0])
        ry = torch.empty(n, device=env.device).uniform_(-xy_range[1], xy_range[1])
        pos_local[:, 0] = rx
        pos_local[:, 1] = ry

    # Yaw noise — sample uniform [-r, r] degrees, build a z-axis rotation quat.
    yaw_range_deg = float(cfg.table_reset_yaw_range_deg)
    if yaw_range_deg > 0.0:
        yaw_rad = (
            torch.empty(n, device=env.device).uniform_(-1.0, 1.0)
            * yaw_range_deg * (torch.pi / 180.0)
        )
        half = yaw_rad * 0.5
        w = torch.cos(half)
        z = torch.sin(half)
        quat = torch.stack([w, torch.zeros_like(w), torch.zeros_like(w), z], dim=-1)
    else:
        quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=env.device, dtype=torch.float32
        ).unsqueeze(0).expand(n, -1)

    pose = torch.cat([pos_local + env_origins, quat], dim=-1)
    env.table.write_root_pose_to_sim(pose, env_ids=env_ids)

def _reset_object_pose(env, env_ids: torch.Tensor) -> None:
    """Independently place each hand's tool in its own table region."""
    cfg = env.cfg.reset
    n = env_ids.numel()
    origins = env.scene.env_origins[env_ids]
    centers = env.cfg.tools.object_centers_xy
    noise_x, noise_y = env.cfg.tools.spawn_xy_noise
    for i, obj in enumerate(env.objects):
        noise = torch.empty(n, 3, device=env.device).uniform_(-1., 1.)
        pos = torch.stack((centers[i][0] + noise[:, 0] * noise_x,
                           centers[i][1] + noise[:, 1] * noise_y,
                           env._table_z_per_env[env_ids] + cfg.table_object_z_offset
                           + noise[:, 2] * cfg.reset_position_noise_z), dim=-1)
        quat = random_orientation(n, device=env.device)
        obj.write_root_pose_to_sim(torch.cat((pos + origins, quat), dim=-1), env_ids=env_ids)
        obj.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)
        env._object_init_z[env_ids, i] = pos[:, 2]


def _reset_goal_pose(env, env_ids: torch.Tensor, mode: str) -> None:
    """Sample independent goals in two non-overlapping lateral workspaces."""
    cfg = env.cfg.reset
    n = env_ids.numel()
    origins = env.scene.env_origins[env_ids]
    for i, goal in enumerate(env.goals):
        # Translate the original local goal volume into each hand's workspace.
        offset = float(env.cfg.tools.goal_y_offsets[i])
        mins = list(cfg.target_volume_mins)
        maxs = list(cfg.target_volume_maxs)
        mins[1] += offset
        maxs[1] += offset
        if mode == "delta":
            pos, quat = sample_delta_goal_pose(
                prev_pos=goal.data.root_pos_w[env_ids] - origins,
                prev_quat_wxyz=goal.data.root_quat_w[env_ids],
                delta_distance=cfg.delta_goal_distance,
                delta_rotation_degrees=cfg.delta_rotation_degrees,
                mins=mins, maxs=maxs, scale=cfg.target_volume_region_scale,
            )
        elif mode == "absolute":
            pos, quat = sample_absolute_goal_pose(mins=mins, maxs=maxs,
                scale=cfg.target_volume_region_scale, n_envs=n, device=env.device)
        else:
            raise ValueError(f"unknown goal sampling mode: {mode}")
        goal.write_root_pose_to_sim(torch.cat((pos + origins, quat), dim=-1), env_ids=env_ids)


def _clear_goal_trackers(env, env_ids: torch.Tensor) -> None:
    env._closest_keypoint_max_dist[env_ids] = -1.0
    env._closest_fingertip_dist[env_ids] = -1.0
    env._near_goal_steps[env_ids] = 0


def reset_goal_trackers(env, env_ids: torch.Tensor) -> None:
    """Clear per-goal trackers and sample the next goal."""
    _clear_goal_trackers(env, env_ids)
    _reset_goal_pose(env, env_ids, mode=env.cfg.reset.goal_sampling_type)


def reset_env_state(env, env_ids: torch.Tensor) -> None:
    """Full per-env reset after ``super()._reset_idx``."""
    n = env_ids.numel()

    _randomize_robot_dof_state(env, env_ids)
    _reset_table_pose(env, env_ids)
    _reset_object_pose(env, env_ids)
    _reset_goal_pose(env, env_ids, mode="absolute")  # full reset → always absolute

    env._prev_episode_successes[env_ids] = env._successes[env_ids]

    _clear_goal_trackers(env, env_ids)
    env._lifted_object[env_ids] = False
    env._successes[env_ids] = 0

    env._action_queue[env_ids] = 0.0
    env._obs_queue[env_ids] = 0.0
    env._object_state_queue[env_ids] = 0.0
    for queue_name in ("_student_camera_queue", "_student_obs_queue"):
        if hasattr(env, queue_name):
            getattr(env, queue_name)[env_ids] = 0.0
    env._object_forces[env_ids] = 0.0
    env._object_torques[env_ids] = 0.0

    dr = env.cfg.domain_randomization
    env._random_force_prob[env_ids] = sample_log_uniform(
        dr.force_prob_range, n
    ).to(env.device)
    env._random_torque_prob[env_ids] = sample_log_uniform(
        dr.torque_prob_range, n
    ).to(env.device)
    lo, hi = dr.object_scale_noise_multiplier_range
    env._object_scale_multiplier[env_ids] = torch.empty(
        n, 2, 3, device=env.device
    ).uniform_(lo, hi)

    # Camera pose randomization (no-op when cfg.student_obs.use_camera_pose_rand is False).
    from ..simtoolreal.utils.scene_utils import _apply_camera_pose_rand_at_reset
    _apply_camera_pose_rand_at_reset(env, env_ids)


__all__ = [
    "allocate_state_buffers",
    "reset_env_state",
    "reset_goal_trackers",
]
