"""Exercise the actual dual-arm Tianji/Wuji simulator task.

    .venv_isaacsim/bin/python isaacsimenvs/tests/test_tianji_wuji_dual_env_smoke.py \\
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
            DUAL_DOF_NAMES as DOF_NAMES, DUAL_PALM_LINKS, DUAL_FINGERTIP_LINKS,
        )
        from isaacsimenvs.tasks.tianji_wuji_dual.config import TianjiWujiDualEnvCfg

        cfg = TianjiWujiDualEnvCfg()
        cfg.seed = 123
        cfg.scene.num_envs = args.num_envs
        cfg.assets.num_assets_per_type = 1
        env = gym.make("Isaacsimenvs-SimToolReal-TianjiWujiDual-Direct-v0", cfg=cfg)
        inner = env.unwrapped
        obs, _ = env.reset(seed=123)

        def finite_observations(observations):
            assert observations["policy"].shape == (args.num_envs, cfg.observation_space)
            assert observations["critic"].shape == (args.num_envs, cfg.state_space)
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
        assert len(names) == 54 and set(names) == set(DOF_NAMES), names
        assert {names[i] for i in inner._arm_joint_ids} == {f"Joint{j}_{side}" for side in ("L", "R") for j in range(1, 8)}
        assert len(inner._hand_joint_ids) == 40
        assert set(DUAL_PALM_LINKS).issubset(bodies)
        assert set(DUAL_FINGERTIP_LINKS).issubset(bodies)
        assert len(inner._fingertip_body_ids) == 10
        assert [names[i] for i in inner._perm_lab_to_canon.tolist()] == list(DOF_NAMES)
        assert [DOF_NAMES[i] for i in inner._perm_canon_to_lab.tolist()] == names
        assert inner.cfg.action_space == 54
        assert inner.single_action_space.shape == (54,)
        for suffix in ("Object", "Object2", "GoalViz", "GoalViz2", "Robot"):
            assert len(find_matching_prim_paths(f"/World/envs/env_.*/{suffix}")) == args.num_envs
        assert len(inner.objects) == len(inner.goals) == 2
        local = torch.stack([obj.data.root_pos_w - inner.scene.env_origins for obj in inner.objects], dim=1)
        assert (local[:, 0, 1] > .2).all() and (local[:, 1, 1] < -.2).all()
        assert (torch.norm(local[:, 0, :2] - local[:, 1, :2], dim=-1) > .4).all()
        assert (inner._object_asset_index_per_env != inner._object_asset_index_per_env2).all()
        assert all(scale[1] == 3. for scale in inner._table_variant_scales)
        import xml.etree.ElementTree as ET
        for path in inner._table_urdf_paths:
            size = [float(x) for x in ET.parse(path).find(".//visual/geometry/box").get("size").split()]
            assert size[1] >= 1.19, size
        finite_observations(obs)
        valid_targets()

        # Verify actuators move both groups in the actual physics backend.
        # Test a small signed arm command and close the hand for one second.
        initial = inner.robot.data.joint_pos.clone()
        command = torch.ones(args.num_envs, 54, device=inner.device)
        command[:, :7] = .2
        command[:, 27:34] = .2
        for _ in range(60):
            obs, reward, _, _, _ = env.step(command)
            finite_observations(obs)
            valid_targets()
        motion = (inner.robot.data.joint_pos - initial).abs()
        assert motion[:, inner._arm_joint_ids].max() > 1e-3, "arm actuator did not move"
        assert motion[:, inner._hand_joint_ids].max() > 1e-2, "hand actuator did not move"
        for side in ("left", "right"):
            arm_ids = [i for i, name in enumerate(names) if name.endswith("_L" if side == "left" else "_R")]
            hand_ids = [i for i, name in enumerate(names) if name.startswith(side + "_finger")]
            assert motion[:, arm_ids].max() > 1e-3, f"{side} arm did not move"
            assert motion[:, hand_ids].max() > 1e-2, f"{side} hand did not move"
        env.reset()

        for step in range(args.steps):
            actions = torch.rand(args.num_envs, 54, device=inner.device) * 2 - 1
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
        untouched_objects = [obj.data.root_state_w[1:].clone() for obj in inner.objects]
        inner._successes[0] = 3
        inner._action_queue[0] = .7
        inner._reset_idx(selected)
        for name, value in before.items():
            torch.testing.assert_close(getattr(inner, name)[1:], value, rtol=0, atol=0)
        for obj, untouched in zip(inner.objects, untouched_objects):
            torch.testing.assert_close(obj.data.root_state_w[1:], untouched, rtol=0, atol=0)
        assert inner.episode_length_buf[0] == inner._successes[0] == 0
        assert not inner._action_queue[0].any()
        valid_targets()

        # The normal env.step path must auto-reset an episode at its timeout.
        env.reset()
        inner.episode_length_buf[0] = inner.max_episode_length - 1
        inner._near_goal_steps.zero_()
        obs, reward, terminated, truncated, _ = env.step(torch.zeros(args.num_envs, 54, device=inner.device))
        assert truncated[0], "episode timeout did not propagate through step()"
        assert inner.episode_length_buf[0] == 0, "timed-out clone did not reset"
        finite_observations(obs)

        # Reaching one tool's goal must not count as a complete paired success.
        env.reset()
        object_poses = [obj.data.root_state_w[0:1, :7].clone() for obj in inner.objects]
        inner.goals[0].write_root_pose_to_sim(object_poses[0], env_ids=selected)
        inner._near_goal_steps.zero_()
        inner._near_goal_steps[0] = cfg.termination.success_steps - 1
        successes = inner._successes.clone()
        before_goals = [goal.data.root_state_w[:, :7].clone() for goal in inner.goals]
        inner._get_dones()
        assert not inner._is_success[0], "one tool incorrectly counted as paired success"
        torch.testing.assert_close(inner._successes, successes)
        for goal, before in zip(inner.goals, before_goals):
            torch.testing.assert_close(goal.data.root_state_w[:, :7], before, rtol=0, atol=0)

        # Both tools reaching their own goals resamples both goals for only this clone.
        for goal, pose in zip(inner.goals, object_poses):
            goal.write_root_pose_to_sim(pose, env_ids=selected)
        inner._near_goal_steps[0] = cfg.termination.success_steps - 1
        inner.episode_length_buf[0] = 7
        untouched_goals = [goal.data.root_state_w[1:, :7].clone() for goal in inner.goals]
        terminated, truncated = inner._get_dones()
        assert inner._is_success[0]
        assert inner._successes[0] == successes[0] + 1
        assert inner.episode_length_buf[0] == 0
        assert not inner._near_goal_steps[0].any()
        assert not terminated[0] and not truncated[0]
        for goal, pose, untouched in zip(inner.goals, object_poses, untouched_goals):
            assert not torch.allclose(goal.data.root_state_w[0:1, :7], pose)
            torch.testing.assert_close(goal.data.root_state_w[1:, :7], untouched, rtol=0, atol=0)

        # The second tool participates in termination and full reset.
        fallen = inner.object2.data.root_state_w[0:1, :7].clone()
        fallen[:, 2] = inner.scene.env_origins[0, 2] - .2
        inner.object2.write_root_pose_to_sim(fallen, env_ids=selected)
        terminated, _ = inner._get_dones()
        assert terminated[0] and inner._termination_reasons["fall"][0]
        inner._reset_idx(selected)
        assert inner.object2.data.root_pos_w[0, 2] > inner.scene.env_origins[0, 2] + .3
        finite_observations(inner._get_observations())
        print(f"[tianji-wuji dual smoke] PASS: {args.num_envs} clones, 54 joints/actions, "
              f"{cfg.observation_space}/{cfg.state_space} observations, {args.steps} random steps, two tools, widened table, subset reset, timeout and paired goal success")
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
