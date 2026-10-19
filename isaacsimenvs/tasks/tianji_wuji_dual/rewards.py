"""Two independent tool rewards, with unchanged bimanual joint penalties."""
import torch

from ..simtoolreal.utils.reward_utils import (
    lifting_reward, distance_delta_reward, keypoint_reward, action_penalty,
    reach_goal_bonus,
)


def compute_rewards(env):
    cfg = env.cfg.reward
    term = env.cfg.termination
    z = torch.stack([obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] for obj in env.objects], dim=1)
    lift, lift_bonus, lifted = lifting_reward(z, env._object_init_z, env._lifted_object,
        cfg.lifting_bonus_threshold, cfg.lifting_bonus, cfg.lifting_rew_scale)
    env._lifted_object = lifted
    n = env.num_envs
    ft, closest_ft = distance_delta_reward(
        env._curr_fingertip_distances.reshape(n * 2, 5),
        env._closest_fingertip_dist.reshape(n * 2, 5),
        lifted.reshape(-1), cfg.distance_delta_rew_scale)
    env._closest_fingertip_dist = closest_ft.reshape(n, 10)
    kp, env._closest_keypoint_max_dist = keypoint_reward(
        env._keypoints_max_dist, env._closest_keypoint_max_dist,
        lifted, cfg.keypoint_rew_scale)
    arm, hand = action_penalty(env.robot.data.joint_vel, env._arm_joint_ids, env._hand_joint_ids,
        cfg.kuka_actions_penalty_scale, cfg.hand_actions_penalty_scale)
    # One goal bonus per complete pair prevents repeatedly farming one easy tool.
    bonus = reach_goal_bonus(env._near_goal.all(dim=-1), env._is_success,
        cfg.reach_goal_bonus, term.success_steps, term.force_consecutive_near_goal_steps)
    terms = {"fingertip_delta_rew": ft.reshape(n, 2).sum(dim=-1),
             "lifting_rew": lift.sum(dim=-1), "lift_bonus_rew": lift_bonus.sum(dim=-1),
             "keypoint_rew": kp.sum(dim=-1), "kuka_actions_penalty": arm,
             "hand_actions_penalty": hand, "bonus_rew": bonus}
    reward = sum(terms.values())
    env._reward_terms = dict(terms, total_reward=reward)
    return reward


def apply_wrench_dr(env):
    """Independent external wrenches and lift gates for each physical tool."""
    dr = env.cfg.domain_randomization
    for state_name, decay, interval, probability, scale in (
        ("_object_forces", dr.force_decay, dr.force_decay_interval, env._random_force_prob, dr.force_scale),
        ("_object_torques", dr.torque_decay, dr.torque_decay_interval, env._random_torque_prob, dr.torque_scale),
    ):
        state = getattr(env, state_name)
        if decay > 0:
            state *= decay ** (env.step_dt / interval)
        else:
            state.zero_()
        fire = torch.rand(env.num_envs, 2, device=env.device) < probability.unsqueeze(1)
        impulse = torch.randn_like(state) * env._object_mass.unsqueeze(-1) * scale
        setattr(env, state_name, torch.where(fire.unsqueeze(-1), impulse, state))
    if dr.force_only_when_lifted:
        env._object_forces *= env._lifted_object.unsqueeze(-1)
    if dr.torque_only_when_lifted:
        env._object_torques *= env._lifted_object.unsqueeze(-1)
    for i, obj in enumerate(env.objects):
        obj.set_external_force_and_torque(forces=env._object_forces[:, i:i+1],
            torques=env._object_torques[:, i:i+1], is_global=True)
