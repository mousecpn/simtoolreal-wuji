"""Record actual KUKA/Sharpa C4 checkpoint rollouts without RTX cameras."""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--sample-actions", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 1 or args.frame_stride < 1:
        parser.error("steps and frame-stride must be positive")
    launcher = AppLauncher(args)
    env = None
    try:
        import gymnasium as gym
        import torch
        from isaaclab.envs.utils.spaces import (
            replace_env_cfg_spaces_with_strings,
            replace_strings_with_env_cfg_spaces,
        )
        from isaaclab.utils import replace_strings_with_slices
        from omegaconf import OmegaConf
        from rl_games.algos_torch.model_builder import ModelBuilder

        import isaacsimenvs  # noqa: F401
        from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg

        saved = replace_strings_with_slices(
            OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
        )
        cfg = replace_env_cfg_spaces_with_strings(SimToolRealEnvCfg())
        cfg.from_dict(saved["env"])
        cfg = replace_strings_with_env_cfg_spaces(cfg)
        cfg.scene.num_envs = 6
        cfg.assets.num_assets_per_type = 1
        cfg.assets.handle_head_types = ("hammer", "screwdriver", "marker")
        cfg.assets.shuffle_assets = False
        cfg.seed = args.seed
        task = "Isaacsimenvs-SimToolReal-C4-SE3-Direct-v0"
        env = gym.make(task, cfg=cfg)
        inner = env.unwrapped
        obs, _ = env.reset(seed=args.seed)
        model = (
            ModelBuilder()
            .load(saved["agent"]["params"])
            .build(
                {
                    "actions_num": 28,
                    "input_shape": (obs["policy"].shape[-1],),
                    "num_seqs": inner.num_envs,
                    "value_size": 1,
                    "normalize_input": False,
                    "normalize_value": True,
                }
            )
            .to(inner.device)
        )
        ckpt = torch.load(
            args.checkpoint, map_location=inner.device, weights_only=False
        )
        ckpt = ckpt.get(0, ckpt)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        state = model.get_default_rnn_state()
        selected = {}
        for i in range(inner.num_envs):
            path = Path(
                inner._object_urdf_paths[int(inner._object_asset_index_per_env[i])]
            )
            for kind in cfg.assets.handle_head_types:
                if kind in path.name and kind not in selected:
                    selected[kind] = i
        assert len(selected) == 3, selected
        args.out.mkdir(parents=True, exist_ok=True)
        assets = args.out / "assets"
        assets.mkdir(exist_ok=True)
        robot_urdf = Path(cfg.assets.robot_urdf).resolve()
        table_urdf = Path(cfg.assets.table_urdf).resolve()
        records = {}
        returns = torch.zeros(inner.num_envs, device=inner.device)
        episode_counts = torch.zeros(
            inner.num_envs, device=inner.device, dtype=torch.long
        )
        for kind, i in selected.items():
            src = Path(
                inner._object_urdf_paths[int(inner._object_asset_index_per_env[i])]
            )
            target = assets / f"{kind}.urdf"
            shutil.copy2(src, target)
            records[kind] = {
                "title": "KUKA iiwa + SHARPA | C4 equivariant LSTM",
                "task": task,
                "tool": kind,
                "env_id": i,
                "seed": args.seed,
                "checkpoint": str(args.checkpoint.resolve()),
                "training_epochs": ckpt.get("epoch", 1),
                "action_mode": "sampled"
                if args.sample_actions
                else "deterministic mean",
                "robot_urdf": str(robot_urdf),
                "table_urdf": str(table_urdf),
                "object_urdf": str(target.resolve()),
                "step_dt": inner.step_dt,
                "fps": 1 / (inner.step_dt * args.frame_stride),
                "box_dimensions": (
                    inner._object_scale_per_env[i] * cfg.reward.object_base_size
                )
                .cpu()
                .tolist(),
                "frames": [],
                "episodes": [],
            }

        def frame(i, step, action=None, done=False):
            origin = inner.scene.env_origins[i]

            def pose(asset):
                p = asset.data.root_state_w[i, :7].clone()
                p[:3] -= origin
                return p.cpu().tolist()

            return {
                "step": step,
                "robot_root_pose": pose(inner.robot),
                "joint_positions": dict(
                    zip(
                        inner.robot.data.joint_names,
                        inner.robot.data.joint_pos[i].cpu().tolist(),
                    )
                ),
                "table_root_pose": pose(inner.table),
                "object_root_pose": pose(inner.object),
                "goal_root_pose": pose(inner.goal_viz),
                "action": action[i].cpu().tolist() if action is not None else None,
                "reset": done,
            }

        for kind, i in selected.items():
            records[kind]["frames"].append(frame(i, 0))
        with torch.inference_mode():
            for step in range(1, args.steps + 1):
                result = model(
                    {"is_train": False, "obs": obs["policy"], "rnn_states": state}
                )
                state = result["rnn_states"]
                action = (
                    result["actions"] if args.sample_actions else result["mus"]
                ).clamp(-1, 1)
                assert torch.isfinite(action).all()
                obs, reward, terminated, truncated, _ = env.step(action)
                assert (
                    torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                )
                returns += reward
                done = terminated | truncated
                episode_counts += done.long()
                for kind, i in selected.items():
                    if done[i]:
                        records[kind]["episodes"].append(
                            {"step": step, "return": returns[i].item()}
                        )
                    if step % args.frame_stride == 0:
                        records[kind]["frames"].append(
                            frame(i, step, action, bool(done[i]))
                        )
                returns[done] = 0
                state = tuple(s * (~done)[None, :, None] for s in state)
                if step % 120 == 0:
                    print(f"[C4 capture] {step}/{args.steps} policy steps", flush=True)
        for kind, i in selected.items():
            records[kind]["unfinished_return"] = returns[i].item()
            records[kind]["episode_count"] = episode_counts[i].item()
            path = args.out / f"{kind}.json"
            path.write_text(json.dumps(records[kind]))
            print(
                f"[C4 capture] SAVED {path}: {len(records[kind]['frames'])} frames",
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
