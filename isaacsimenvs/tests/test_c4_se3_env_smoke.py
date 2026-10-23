"""Actual KUKA/Sharpa C4 policy -> SE3 IK -> PhysX smoke test.

.venv_isaacsim/bin/python isaacsimenvs/tests/test_c4_se3_env_smoke.py --headless
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=16)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs < 2 or args.steps < 2:
        parser.error("need at least two environments and two steps")
    launcher = AppLauncher(args)
    env = None
    try:
        import gymnasium as gym
        import torch
        import yaml
        from rl_games.algos_torch.model_builder import ModelBuilder

        import isaacsimenvs  # noqa: F401
        from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
        from scripts.verify_c4_equivariance import check_policy

        task = "Isaacsimenvs-SimToolReal-C4-SE3-Direct-v0"
        cfg = SimToolRealEnvCfg()
        spec = gym.spec(task)
        with open(spec.kwargs["env_cfg_yaml_entry_point"]) as f:
            cfg.from_dict(yaml.safe_load(f))
        cfg.scene.num_envs = args.num_envs
        cfg.assets.num_assets_per_type = 1
        cfg.domain_randomization.use_action_delay = False
        cfg.domain_randomization.use_obs_delay = False
        cfg.domain_randomization.use_object_state_delay_noise = False
        cfg.domain_randomization.joint_velocity_obs_noise_std = 0
        cfg.seed = 42
        env = gym.make(task, cfg=cfg)
        inner = env.unwrapped
        obs, _ = env.reset(seed=42)
        with open(spec.kwargs["rl_games_cfg_entry_point"]) as f:
            agent = yaml.safe_load(f)
        builder = ModelBuilder().load(agent["params"])
        model = builder.build(
            {
                "actions_num": 28,
                "input_shape": (obs["policy"].shape[-1],),
                "num_seqs": args.num_envs,
                "value_size": 1,
                "normalize_input": False,
                "normalize_value": True,
            }
        ).to(inner.device)
        state = model.get_default_rnn_state()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        max_error = check_policy(model.a2c_network, obs["policy"])
        assert obs["policy"].shape == (args.num_envs, 174)
        assert obs["critic"].shape == (args.num_envs, 196)
        arm_motion = 0.0
        for step in range(args.steps):
            model.eval()
            with torch.no_grad():
                result = model(
                    {"is_train": False, "obs": obs["policy"], "rnn_states": state}
                )
                state = result["rnn_states"]
                actions = result["mus"].clamp(-1, 1)
            assert (
                actions.shape == (args.num_envs, 28) and torch.isfinite(actions).all()
            )
            before = inner.robot.data.joint_pos[:, inner._arm_joint_ids].clone()
            obs, reward, terminated, truncated, _ = env.step(actions)
            targets = inner._cur_targets[:, inner._arm_joint_ids]
            arm_motion = max(
                arm_motion,
                (inner.robot.data.joint_pos[:, inner._arm_joint_ids] - before)
                .abs()
                .max()
                .item(),
            )
            active = ~(terminated | truncated)
            if active.any():
                assert (
                    targets[active] - before[active]
                ).abs().max() <= cfg.action.ik_max_joint_delta + 1e-5
            limits = inner.robot.data.joint_pos_limits
            assert (inner._cur_targets >= limits[..., 0] - 1e-5).all()
            assert (inner._cur_targets <= limits[..., 1] + 1e-5).all()
            assert all(torch.isfinite(x).all() for x in [reward, *obs.values(), *state])
            done = terminated | truncated
            state = tuple(s * (~done)[None, :, None] for s in state)
            if step == args.steps // 2:
                # Train through the rl_games distribution/value wrapper on live observations.
                model.train()
                train = model(
                    {
                        "is_train": True,
                        "obs": obs["policy"],
                        "rnn_states": tuple(s.detach() for s in state),
                        "prev_actions": actions,
                    }
                )
                loss = train["prev_neglogp"].mean() + train["values"].square().mean()
                optimizer.zero_grad()
                loss.backward()
                grads = [p.grad for p in model.parameters() if p.requires_grad]
                assert all(g is not None and torch.isfinite(g).all() for g in grads)
                optimizer.step()
                state = tuple(s.detach() for s in state)
                selected = torch.tensor([0], device=inner.device)
                untouched = inner.robot.data.joint_pos[1:].clone()
                inner._reset_idx(selected)
                torch.testing.assert_close(inner.robot.data.joint_pos[1:], untouched)
                state = tuple(s.clone() for s in state)
                for s in state:
                    s[:, 0] = 0
                obs = inner._get_observations()
        assert arm_motion > 1e-7, "IK commands did not move the arm"
        model.eval()
        max_error = max(max_error, check_policy(model.a2c_network, obs["policy"]))
        print(
            f"[C4 SE3 smoke] PASS: {args.num_envs} envs, {args.steps} policy steps, "
            f"174/196 obs, 28 actions, live IK/limits, subset reset, backward+optimizer, "
            f"arm_motion={arm_motion:.3g}, equivariance_max_error={max_error:.3g}",
            flush=True,
        )
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
