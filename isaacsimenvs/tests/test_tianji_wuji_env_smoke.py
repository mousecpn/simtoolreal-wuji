"""Exercise the actual single-arm Tianji/Wuji simulator task.

    .venv_isaacsim/bin/python isaacsimenvs/tests/test_tianji_wuji_env_smoke.py \\
        --headless --num_envs 2 --steps 40

Requires Isaac Sim and the DexTele assets; CPU action tests live separately in
tests/test_tianji_wuji_actions.py and never import the simulator.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=40)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 2 or args.steps < 1:
        parser.error("--num_envs must be at least 2; --steps must be positive")
    launcher = AppLauncher(args)
    env = None
    try:
        # All simulator-dependent imports must follow AppLauncher.
        import gymnasium as gym
        import torch
        import isaacsimenvs  # noqa: F401
        from isaaclab.sim.utils import find_matching_prim_paths
        from isaacsimenvs.tasks.tianji_wuji.assets import (
            ARM_JOINT_NAMES, HAND_JOINT_NAMES, DOF_NAMES, PALM_LINK, FINGERTIP_LINKS,
        )
        from isaacsimenvs.tasks.tianji_wuji.config import TianjiWujiEnvCfg

        cfg = TianjiWujiEnvCfg()
        cfg.seed = 123
        cfg.scene.num_envs = args.num_envs
        cfg.assets.num_assets_per_type = 1
        env = gym.make("Isaacsimenvs-SimToolReal-TianjiWuji-Direct-v0", cfg=cfg)
        inner = env.unwrapped
        obs, _ = env.reset(seed=123)

        def finite_observations(observations):
            assert observations["policy"].shape == (args.num_envs, 134)
            assert observations["critic"].shape == (args.num_envs, 156)
            for key in ("policy", "critic"):
                assert torch.isfinite(observations[key]).all(), f"nonfinite {key}"

        def valid_targets():
            limits = inner.robot.data.joint_pos_limits
            assert torch.isfinite(limits).all()
            assert (limits[..., 0] < limits[..., 1]).all()
            for name in ("_cur_targets", "_prev_targets"):
                values = getattr(inner, name)
                assert torch.isfinite(values).all(), name
                assert (values >= limits[..., 0] - 1e-6).all(), name
                assert (values <= limits[..., 1] + 1e-6).all(), name
            assert torch.isfinite(inner.robot.data.joint_pos).all()
            assert torch.isfinite(inner.robot.data.joint_vel).all()

        names = list(inner.robot.data.joint_names)
        bodies = list(inner.robot.data.body_names)
        assert len(names) == 27 and set(names) == set(DOF_NAMES), names
        assert {names[i] for i in inner._arm_joint_ids} == set(ARM_JOINT_NAMES)
        assert {names[i] for i in inner._hand_joint_ids} == set(HAND_JOINT_NAMES)
        assert bodies[inner._palm_body_id] == PALM_LINK
        assert tuple(bodies[i] for i in inner._fingertip_body_ids) == FINGERTIP_LINKS
        assert not any("left" in name.lower() or name.endswith("_L") for name in bodies)
        assert [names[i] for i in inner._perm_lab_to_canon.tolist()] == list(DOF_NAMES)
        assert [DOF_NAMES[i] for i in inner._perm_canon_to_lab.tolist()] == names
        assert inner.cfg.action_space == 27
        assert inner.single_action_space.shape == (27,)
        # Verify effective imported limits and actuator gains, not only config.
        limits = inner.robot.data.joint_pos_limits[0]
        for finger in range(2, 6):
            joint_id = names.index(f"right_finger{finger}_joint2")
            torch.testing.assert_close(limits[joint_id], torch.tensor(
                [-0.03491, 0.03491], device=inner.device))
        for joint_names, expected, field in (
            (ARM_JOINT_NAMES, cfg.robot.arm_stiffness, "joint_stiffness"),
            (HAND_JOINT_NAMES, cfg.robot.hand_stiffness, "joint_stiffness"),
        ):
            joint_ids = [names.index(name) for name in joint_names]
            torch.testing.assert_close(getattr(inner.robot.data, field)[0, joint_ids],
                torch.tensor(expected, device=inner.device))
        table_top = inner.table.data.root_pos_w[:, 2] - inner.scene.env_origins[:, 2] + 0.15
        assert ((table_top >= 0.63 - 1e-6) & (table_top <= 0.65 + 1e-6)).all()
        for suffix in ("Object", "GoalViz", "Robot"):
            assert len(find_matching_prim_paths(f"/World/envs/env_.*/{suffix}")) == args.num_envs
        finite_observations(obs)
        valid_targets()

        # Verify actuators move both groups in the actual physics backend.
        # Test a small signed arm command and close the hand for one second.
        initial = inner.robot.data.joint_pos.clone()
        command = torch.ones(args.num_envs, 27, device=inner.device)
        command[:, :7] = .2
        for _ in range(60):
            obs, reward, _, _, _ = env.step(command)
            finite_observations(obs)
            valid_targets()
        motion = (inner.robot.data.joint_pos - initial).abs()
        assert motion[:, inner._arm_joint_ids].max() > 1e-3, "arm actuator did not move"
        assert motion[:, inner._hand_joint_ids].max() > 1e-2, "hand actuator did not move"
        env.reset()

        for step in range(args.steps):
            actions = torch.rand(args.num_envs, 27, device=inner.device) * 2 - 1
            obs, reward, terminated, truncated, _ = env.step(actions)
            finite_observations(obs)
            assert reward.shape == (args.num_envs,) and torch.isfinite(reward).all(), step
            assert terminated.shape == truncated.shape == (args.num_envs,)
            valid_targets()

        # Reset one clone only: no other clone's state or queues may be reset.
        selected = torch.tensor([0], device=inner.device)
        before = {name: getattr(inner, name)[1:].clone() for name in (
            "_cur_targets", "_prev_targets", "_successes", "episode_length_buf", "_action_queue",
        )}
        untouched_object = inner.object.data.root_state_w[1:].clone()
        inner._successes[0] = 3
        inner._action_queue[0] = .7
        inner._reset_idx(selected)
        for name, value in before.items():
            torch.testing.assert_close(getattr(inner, name)[1:], value, rtol=0, atol=0)
        torch.testing.assert_close(inner.object.data.root_state_w[1:], untouched_object, rtol=0, atol=0)
        assert inner.episode_length_buf[0] == inner._successes[0] == 0
        assert not inner._action_queue[0].any()
        valid_targets()

        # The normal env.step path must auto-reset an episode at its timeout.
        env.reset()
        inner.episode_length_buf[0] = inner.max_episode_length - 1
        inner._near_goal_steps.zero_()
        obs, reward, terminated, truncated, _ = env.step(torch.zeros(args.num_envs, 27, device=inner.device))
        assert truncated[0], "episode timeout did not propagate through step()"
        assert inner.episode_length_buf[0] == 0, "timed-out clone did not reset"
        finite_observations(obs)

        # Align the live goal to the live object, then exercise the actual done
        # hook without a physics tick moving the object away from that pose.
        env.reset()
        object_pose = inner.object.data.root_state_w[0:1, :7].clone()
        # A break in goal proximity must discard the preceding nine steps.
        far_pose = object_pose.clone()
        far_pose[:, 0] += 0.5
        inner.goal_viz.write_root_pose_to_sim(far_pose, env_ids=selected)
        inner._near_goal_steps.zero_()
        inner._near_goal_steps[0] = cfg.termination.success_steps - 1
        inner._get_dones()
        assert not inner._is_success[0] and inner._near_goal_steps[0] == 0
        inner.goal_viz.write_root_pose_to_sim(object_pose, env_ids=selected)
        inner._near_goal_steps.zero_()
        inner._near_goal_steps[0] = cfg.termination.success_steps - 1
        inner.episode_length_buf[0] = 7
        successes = inner._successes.clone()
        untouched_goal = inner.goal_viz.data.root_state_w[1:, :7].clone()
        terminated, truncated = inner._get_dones()
        assert inner._is_success[0], "matching object and goal did not count as success"
        assert inner._successes[0] == successes[0] + 1
        assert inner.episode_length_buf[0] == 0
        assert inner._near_goal_steps[0] == 0
        assert not terminated[0] and not truncated[0]
        assert not torch.allclose(inner.goal_viz.data.root_state_w[0:1, :7], object_pose), "goal was not resampled"
        torch.testing.assert_close(inner.goal_viz.data.root_state_w[1:, :7], untouched_goal, rtol=0, atol=0)
        finite_observations(inner._get_observations())
        print(f"[tianji-wuji smoke] PASS: {args.num_envs} clones, 27 joints/actions, "
              f"134/156 observations, {args.steps} random steps, subset reset, timeout and goal success",
              flush=True)
    except BaseException:
        # Kit's fast shutdown may bypass Python's uncaught-exception printer.
        import traceback
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
