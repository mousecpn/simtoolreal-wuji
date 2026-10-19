"""Roll out the real simulator and export one scene for camera-free rendering."""
from pathlib import Path
import argparse
import json
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=45)
    parser.add_argument("--arms", choices=("single", "dual"), default="single",
                        help="Single right arm/hand (default), or both Tianji arms and Wuji hands")
    parser.add_argument("--out", type=Path, default=Path("outputs/tianji_wuji_snapshot/state.json"))
    parser.add_argument("--trajectory", action="store_true", help="Save a sampled rollout instead of a single state")
    parser.add_argument("--frame-stride", type=int, default=2)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.steps < 1 or args.frame_stride < 1:
        parser.error("steps and frame-stride must be positive")
    if args.trajectory and args.steps % args.frame_stride:
        parser.error("trajectory steps must be divisible by frame-stride")
    app = AppLauncher(args).app
    env = None
    try:
        import gymnasium as gym
        import torch
        import isaacsimenvs
        if args.arms == "dual":
            from isaacsimenvs.tasks.tianji_wuji_dual.config import TianjiWujiDualEnvCfg
            cfg = TianjiWujiDualEnvCfg()
            task = "Isaacsimenvs-SimToolReal-TianjiWujiDual-Direct-v0"
        else:
            from isaacsimenvs.tasks.tianji_wuji.config import TianjiWujiEnvCfg
            cfg = TianjiWujiEnvCfg()
            task = "Isaacsimenvs-SimToolReal-TianjiWuji-Direct-v0"
        cfg.seed = 42
        cfg.scene.num_envs = 1
        cfg.assets.handle_head_types = ("hammer", "screwdriver") if args.arms == "dual" else ("hammer",)
        cfg.assets.num_assets_per_type = 1
        env = gym.make(task, cfg=cfg)
        inner = env.unwrapped
        obs, _ = env.reset()
        origin = inner.scene.env_origins[0]

        def pose(asset):
            value = asset.data.root_state_w[0, :7].clone()
            value[:3] -= origin
            return value.cpu().tolist()

        def frame(step):
            result = {
                "step": step,
                "robot_root_pose": pose(inner.robot),
                "joint_positions": dict(zip(inner.robot.data.joint_names, inner.robot.data.joint_pos[0].cpu().tolist())),
                "table_root_pose": pose(inner.table),
                "object_root_pose": pose(inner.object),
                "goal_root_pose": pose(inner.goal_viz),
            }
            if hasattr(inner, "object2"):
                result["object2_root_pose"] = pose(inner.object2)
                result["goal2_root_pose"] = pose(inner.goal_viz2)
            return result

        frames = [frame(0)] if args.trajectory else []
        # A small random arm command and slowly varying random finger targets.
        # The scene is a physical rollout, not a posed model or trained policy.
        action = torch.zeros(1, cfg.action_space, device=inner.device)
        for step in range(args.steps):
            if step % 15 == 0:
                action = torch.rand_like(action) * 2 - 1
                action[:, :7] *= .6 if args.trajectory else .25
                if args.arms == "dual":
                    action[:, 27:34] *= .6 if args.trajectory else .25
            obs, reward, _, _, _ = env.step(action)
            assert torch.isfinite(obs["policy"]).all()
            if args.trajectory and (step + 1) % args.frame_stride == 0:
                frames.append(frame(step + 1))

        index = int(inner._object_asset_index_per_env[0])
        table_index = int(inner._table_asset_index_per_env[0]) if hasattr(inner, "_table_asset_index_per_env") else 0
        snapshot = {
            "arms": args.arms,
            "task": task,
            "action_dim": cfg.action_space,
            "step": args.steps,
            "robot_urdf": inner._robot_urdf_path,
            "robot_root_pose": pose(inner.robot),
            "joint_positions": dict(zip(inner.robot.data.joint_names, inner.robot.data.joint_pos[0].cpu().tolist())),
            "table_urdf": str(Path(inner._table_urdf_paths[table_index]).resolve())
                if hasattr(inner, "_table_urdf_paths") else str(Path(cfg.assets.table_urdf).resolve()),
            "table_root_pose": pose(inner.table),
            "object_urdf": inner._object_urdf_paths[index],
            "object_root_pose": pose(inner.object),
            "goal_root_pose": pose(inner.goal_viz),
        }
        if hasattr(inner, "object2"):
            index2 = int(inner._object_asset_index_per_env2[0])
            snapshot.update(
                object2_urdf=inner._object_urdf_paths[index2],
                object2_root_pose=pose(inner.object2),
                goal2_root_pose=pose(inner.goal_viz2),
            )
        if hasattr(inner, "_table_variant_scales"):
            snapshot["table_scale_xy"] = inner._table_variant_scales[table_index]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.trajectory:
            # Keep exported trajectories renderable after /tmp cleanup.
            asset_dir = args.out.parent / "assets"
            asset_dir.mkdir(exist_ok=True)
            robot_dir = asset_dir / "robot"
            shutil.copytree(Path(snapshot["robot_urdf"]).parent, robot_dir, dirs_exist_ok=True)
            snapshot["robot_urdf"] = str((robot_dir / Path(snapshot["robot_urdf"]).name).resolve())
            for name in ("table", "object", "object2"):
                if f"{name}_urdf" not in snapshot:
                    continue
                source = Path(snapshot[f"{name}_urdf"])
                target = asset_dir / f"{name}.urdf"
                shutil.copy2(source, target)
                snapshot[f"{name}_urdf"] = str(target.resolve())
            snapshot["frames"] = frames
            snapshot["step_dt"] = inner.step_dt
            snapshot["fps"] = 1.0 / (inner.step_dt * args.frame_stride)
        args.out.write_text(json.dumps(snapshot, indent=2))
        print(f"SNAPSHOT_STATE_SAVED: {args.out.resolve()}", flush=True)
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
