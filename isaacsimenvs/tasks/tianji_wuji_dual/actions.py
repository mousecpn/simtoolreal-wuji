"""Position targets indexed by joint identity, independent of parser order."""
import torch


def apply_action_pipeline(env, actions: torch.Tensor) -> None:
    if actions.shape != (env.num_envs, 54):
        raise ValueError(f"Expected ({env.num_envs}, 54) actions, got {tuple(actions.shape)}")
    dr, cfg = env.cfg.domain_randomization, env.cfg.action
    actions = actions.to(env.device).clamp(-1., 1.)[:, env._perm_canon_to_lab]
    if dr.use_action_delay and dr.action_delay_max > 0:
        first = (env.episode_length_buf == 0) & (env._successes == 0)
        env._action_queue[first] = actions[first].unsqueeze(1)
        env._action_queue = torch.roll(env._action_queue, 1, 1)
        env._action_queue[:, 0] = actions
        delay = torch.randint(env._action_queue.shape[1], (env.num_envs,), device=env.device)
        actions = env._action_queue[torch.arange(env.num_envs, device=env.device), delay]
    arm, hand = env._arm_joint_ids, env._hand_joint_ids
    prev_arm = env._prev_targets[:, arm]
    raw_arm = (prev_arm + cfg.dof_speed_scale * env.step_dt * actions[:, arm]).clamp(
        env._arm_lower, env._arm_upper)
    env._cur_targets[:, arm] = torch.lerp(prev_arm, raw_arm, cfg.arm_moving_average).clamp(
        env._arm_lower, env._arm_upper)
    raw_hand = env._hand_lower + .5 * (actions[:, hand] + 1.) * (env._hand_upper - env._hand_lower)
    env._cur_targets[:, hand] = torch.lerp(
        env._prev_targets[:, hand], raw_hand, cfg.hand_moving_average
    ).clamp(env._hand_lower, env._hand_upper)
    env._prev_targets.copy_(env._cur_targets)
