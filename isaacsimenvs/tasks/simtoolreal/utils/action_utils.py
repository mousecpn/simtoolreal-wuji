"""Action processing and random object wrench helpers."""

from __future__ import annotations

import math

import torch


def sample_log_uniform(lo_hi: tuple[float, float], n: int) -> torch.Tensor:
    """Log-uniform sample on CPU (caller moves to device)."""
    lo, hi = lo_hi
    return torch.exp(
        torch.empty(n).uniform_(math.log(lo + 1e-12), math.log(hi + 1e-12))
    )


def apply_action_pipeline(env, actions: torch.Tensor) -> None:
    """Apply delay, canonical-to-Lab mapping, smoothing, and target clamps."""
    # Debug replay path accepts an already Lab-order target.
    replay_target = getattr(env, "_replay_target_lab_order", None)
    if replay_target is not None:
        env._cur_targets[:] = replay_target
        env._prev_targets = env._cur_targets.clone()
        return

    dr = env.cfg.domain_randomization
    act_cfg = env.cfg.action
    dt = env.step_dt  # policy step (= decimation * sim.dt = 1/60 s)

    actions = actions.clone().to(env.device)

    # Canonical policy order -> Lab parser order.
    se3 = act_cfg.arm_control_mode == "se3"
    if not se3:
        actions = actions[:, env._perm_canon_to_lab]

    # Action delay.
    if dr.use_action_delay and dr.action_delay_max > 0:
        episode_start = (env.episode_length_buf == 0) & (env._successes == 0)
        if episode_start.any():
            env._action_queue[episode_start] = actions[episode_start].unsqueeze(1)
        env._action_queue = torch.roll(env._action_queue, shifts=1, dims=1)
        env._action_queue[:, 0, :] = actions
        delay_idx = torch.randint(
            0, env._action_queue.shape[1], (env.num_envs,), device=env.device
        )
        actions = env._action_queue[
            torch.arange(env.num_envs, device=env.device), delay_idx
        ]

    if se3:
        arm_smoothed = compute_se3_arm_targets(env, actions[:, :6].clamp(-1.0, 1.0))
        # Hand inputs remain canonical; select in the parser's hand-joint order.
        hand_action = actions[:, 6:][:, env._perm_canon_to_lab[env._hand_joint_ids] - 7]
    else:
        # Legacy joint velocity-delta accumulator.
        arm_action = actions[:, :7]
        arm_raw = env._prev_targets[:, :7] + act_cfg.dof_speed_scale * dt * arm_action
        arm_raw = torch.clamp(arm_raw, env._arm_lower, env._arm_upper)
        arm_smoothed = (
            act_cfg.arm_moving_average * arm_raw
            + (1.0 - act_cfg.arm_moving_average) * env._prev_targets[:, :7]
        )
        arm_smoothed = torch.clamp(arm_smoothed, env._arm_lower, env._arm_upper)
        hand_action = actions[:, 7:]

    # Hand: absolute [-1, 1] scale.
    hand_raw = env._hand_lower + 0.5 * (hand_action + 1.0) * (
        env._hand_upper - env._hand_lower
    )
    hand_smoothed = (
        act_cfg.hand_moving_average * hand_raw
        + (1.0 - act_cfg.hand_moving_average) * env._prev_targets[:, 7:]
    )
    hand_smoothed = torch.clamp(hand_smoothed, env._hand_lower, env._hand_upper)

    # Write Lab-order targets and cache them for the next step.
    env._cur_targets[:, env._arm_joint_ids] = arm_smoothed
    env._cur_targets[:, env._hand_joint_ids] = hand_smoothed
    env._prev_targets = env._cur_targets.clone()


def damped_ik(jacobian: torch.Tensor, pose_delta: torch.Tensor, damping: float) -> torch.Tensor:
    """Batched DLS inverse: J.T (J J.T + lambda^2 I)^-1 delta_pose."""
    if damping <= 0:
        raise ValueError("ik_damping must be positive")
    jt = jacobian.transpose(-1, -2)
    identity = torch.eye(6, device=jacobian.device, dtype=jacobian.dtype)
    return (jt @ torch.linalg.solve(
        jacobian @ jt + damping**2 * identity, pose_delta.unsqueeze(-1)
    )).squeeze(-1)


def compute_se3_arm_targets(env, actions: torch.Tensor) -> torch.Tensor:
    """One differential IK step at the observed palm pose, in world axes.

    Actions specify a relative SE(3) target: p_des=p+dp and
    R_des=Exp(rotvec)R. Re-linearize at each rollout step. The Jacobian is
    shifted from the palm link origin to the same palm center as observations.
    """
    from isaaclab.utils.math import quat_apply
    from .obs_utils import PALM_CENTER_OFFSET

    cfg = env.cfg.action
    if cfg.ik_max_joint_delta <= 0:
        raise ValueError("ik_max_joint_delta must be positive")
    robot = env.robot
    # PhysX excludes the root body from a fixed-base articulation Jacobian.
    body_index = env._palm_body_id - int(robot.is_fixed_base)
    joint_columns = env._arm_joint_ids
    if not robot.is_fixed_base:
        joint_columns = [j + 6 for j in joint_columns]
    jacobian = robot.root_physx_view.get_jacobians()[:, body_index, :, joint_columns].clone()
    offset = torch.tensor(PALM_CENTER_OFFSET, device=env.device, dtype=actions.dtype)
    offset_w = quat_apply(robot.data.body_quat_w[:, env._palm_body_id],
                          offset.expand(env.num_envs, -1))
    # v_center = v_origin + omega cross offset.
    jacobian[:, :3] += torch.cross(
        jacobian[:, 3:].transpose(1, 2), offset_w.unsqueeze(1).expand(-1, 7, -1), dim=-1
    ).transpose(1, 2)
    delta_pose = torch.cat((actions[:, :3] * cfg.se3_position_scale,
                            actions[:, 3:] * cfg.se3_rotation_scale), dim=-1)
    delta_q = damped_ik(jacobian, delta_pose, cfg.ik_damping)
    delta_q = delta_q.clamp(-cfg.ik_max_joint_delta, cfg.ik_max_joint_delta)
    current = robot.data.joint_pos[:, env._arm_joint_ids]
    return (current + delta_q).clamp(env._arm_lower, env._arm_upper)


def apply_wrench_dr(env) -> None:
    """Apply decayed random force/torque impulses to the object."""
    dr = env.cfg.domain_randomization
    dt_pol = env.step_dt

    # Decay previous wrench.
    if dr.force_decay > 0.0:
        env._object_forces *= dr.force_decay ** (dt_pol / dr.force_decay_interval)
    else:
        env._object_forces.zero_()
    if dr.torque_decay > 0.0:
        env._object_torques *= dr.torque_decay ** (dt_pol / dr.torque_decay_interval)
    else:
        env._object_torques.zero_()

    # Sample new wrench with per-env probability.
    force_fire = (
        torch.rand(env.num_envs, device=env.device) < env._random_force_prob
    ).view(-1, 1, 1)
    torque_fire = (
        torch.rand(env.num_envs, device=env.device) < env._random_torque_prob
    ).view(-1, 1, 1)
    mass = env._object_mass.unsqueeze(-1)  # (N, 1, 1)
    new_force = (
        torch.randn(env.num_envs, 1, 3, device=env.device) * mass * dr.force_scale
    )
    new_torque = (
        torch.randn(env.num_envs, 1, 3, device=env.device) * mass * dr.torque_scale
    )
    env._object_forces = torch.where(force_fire, new_force, env._object_forces)
    env._object_torques = torch.where(torque_fire, new_torque, env._object_torques)

    # _lifted_object is from the previous step because rewards update later.
    if dr.force_only_when_lifted:
        env._object_forces *= env._lifted_object.float().view(-1, 1, 1)
    if dr.torque_only_when_lifted:
        env._object_torques *= env._lifted_object.float().view(-1, 1, 1)

    # Optional task-level gate: subclasses can disable wrench DR per env (e.g.
    # peg_in_hole zeros impulses once the final insert is achieved so the part
    # doesn't fly out of the hole during retract). Same opt-in convention as
    # _curriculum_eligible_mask in termination_utils.
    if hasattr(env, "_wrench_dr_active_mask"):
        active = env._wrench_dr_active_mask()
        if active is not None:
            active_f = active.float().view(-1, 1, 1)
            env._object_forces *= active_f
            env._object_torques *= active_f

    env.object.set_external_force_and_torque(
        forces=env._object_forces,
        torques=env._object_torques,
        is_global=True,
    )


__all__ = ["apply_action_pipeline", "apply_wrench_dr", "sample_log_uniform"]
