"""Lightweight env-creation smoke test for SimToolReal.

Skips Hydra + rl_games + tensorboard. Just spins up AppLauncher, instantiates
the env via `gym.make`, runs a few steps with random actions, and asserts
the per-env Object prim count matches num_envs (the original cloner-drop
guard rail). Useful when iterating on scene_utils.

    .venv_isaacsim/bin/python isaacsimenvs/tests/test_simtoolreal_env_smoke.py \\
      --num_envs 8 --num_assets_per_type 2 --steps 10
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


def _remove_video_arg_from_kit_argv() -> None:
    """Prevent Kit from interpreting this script's output path as its video flag."""
    filtered_argv = [sys.argv[0]]
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == "--video":
            skip_next = True
            continue
        if arg.startswith("--video="):
            continue
        filtered_argv.append(arg)
    sys.argv = filtered_argv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--num_assets_per_type", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    # Do not use ``dest="video"`` here.  AppLauncher treats any ``video``
    # argument as a request for its default viewport, which creates an extra
    # Hydra/RTX engine in addition to the rgb_array render product below.
    # Headless capture only needs the latter.
    parser.add_argument(
        "--video", dest="video_path", type=Path, help="Output MP4 path (enables headless rendering)"
    )
    parser.add_argument("--video_env", type=int, default=0)
    parser.add_argument("--video_width", type=int, default=640)
    parser.add_argument("--video_height", type=int, default=480)
    parser.add_argument(
        "--video_compat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable Fabric Scene Delegate and multi-GPU RTX startup for more robust single-GPU capture.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs <= 0 or args.steps <= 0:
        parser.error("--num_envs and --steps must be positive")
    if not 0 <= args.video_env < args.num_envs:
        parser.error("--video_env must be in [0, num_envs)")
    if args.video_width <= 0 or args.video_height <= 0:
        parser.error("video dimensions must be positive")
    if args.video_path and (args.video_width % 2 or args.video_height % 2):
        parser.error("MP4 video dimensions must be even")
    args.headless = True
    if args.video_path:
        args.enable_cameras = True
        if args.video_compat:
            # This task keeps every cloned environment in USD
            # (scene.clone_in_fabric=False), so rendering does not require the
            # Fabric Scene Delegate. Avoid its early RTX initialization path,
            # which crashes on some single-GPU Linux driver stacks.
            compatibility_args = (
                "--/app/useFabricSceneDelegate=false "
                "--/rtx/hydra/readTransformsFromFabricInRenderDelegate=false "
                "--/renderer/multiGpu/enabled=false "
                "--/renderer/multiGpu/autoEnable=false"
            )
            args.kit_args = f"{args.kit_args} {compatibility_args}".strip()

        # AppLauncher forwards raw sys.argv to Kit. Strip this script's
        # ``--video`` option after argparse consumes it; otherwise Kit creates
        # a default viewport before the intended offscreen render product.
        _remove_video_arg_from_kit_argv()

    app_launcher = AppLauncher(args)
    app = app_launcher.app

    if args.video_path:
        import imageio.v2 as imageio
        import imageio_ffmpeg

        imageio_ffmpeg.get_ffmpeg_exe()

    import gymnasium as gym
    import torch

    import isaacsimenvs  # noqa: F401  (registers gym envs)
    from isaaclab.sim.utils import find_matching_prim_paths
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg

    cfg = SimToolRealEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.assets.num_assets_per_type = args.num_assets_per_type

    if args.video_path:
        cfg.viewer.origin_type = "env"
        cfg.viewer.env_index = args.video_env
        cfg.viewer.resolution = (args.video_width, args.video_height)

    env = gym.make(
        "Isaacsimenvs-SimToolReal-Direct-v0",
        cfg=cfg,
        render_mode="rgb_array" if args.video_path else None,
    )
    inner = env.unwrapped

    object_prims = find_matching_prim_paths("/World/envs/env_.*/Object")
    goal_prims = find_matching_prim_paths("/World/envs/env_.*/GoalViz")
    assert len(object_prims) == args.num_envs, (
        f"Object prims: {len(object_prims)}, expected {args.num_envs}"
    )
    assert len(goal_prims) == args.num_envs, (
        f"GoalViz prims: {len(goal_prims)}, expected {args.num_envs}"
    )
    print(
        f"[smoke] OK — {len(object_prims)} Object + {len(goal_prims)} GoalViz prims"
    )

    # Diagnostic: all PhysX joint params that could dampen response —
    # stiffness, damping, armature (virtual inertia), friction, effort limit.
    from isaacsimenvs.tasks.simtoolreal.utils.scene_utils import JOINT_NAMES_CANONICAL
    stiffness = inner.robot.data.joint_stiffness[0].cpu().numpy()
    damping = inner.robot.data.joint_damping[0].cpu().numpy()
    armature = inner.robot.data.joint_armature[0].cpu().numpy()
    friction = inner.robot.data.joint_friction_coeff[0].cpu().numpy()
    effort_lim = inner.robot.data.joint_effort_limits[0].cpu().numpy()
    perm = inner._perm_lab_to_canon.cpu().numpy()
    # Dump per-link masses / inertias Isaac Lab sees from the URDF.
    # URDF declares thumb chain masses around 1e-3 kg per link — if
    # PhysX 5 shows e.g. 1 kg per link we've found the inertia inflation
    # that explains the 100x slow PD response.
    masses = inner.robot.data.default_mass[0].cpu().numpy()  # (num_bodies,)
    inertias = inner.robot.data.default_inertia[0].cpu().numpy()  # (num_bodies, 9) or similar
    body_names = inner.robot.data.body_names
    print(f"[smoke] Body masses / inertias (all {len(body_names)} bodies):")
    print(f"  {'idx':>3}  {'name':<25s}  {'mass (kg)':>10s}  {'Ixx':>10s}  {'Iyy':>10s}  {'Izz':>10s}")
    for i, name in enumerate(body_names):
        ixx = inertias[i, 0] if inertias.ndim == 2 else inertias[i]
        iyy = inertias[i, 4] if inertias.ndim == 2 and inertias.shape[1] >= 5 else 0
        izz = inertias[i, 8] if inertias.ndim == 2 and inertias.shape[1] >= 9 else 0
        print(f"  [{i:2d}] {name:<25s}  {masses[i]:>10.5f}  {ixx:>10.2e}  {iyy:>10.2e}  {izz:>10.2e}")
    print()
    print(f"[smoke] PhysX joint parameters (canonical order):")
    print(f"  {'idx':>3}  {'name':<22s}  {'K':>8s}  {'D':>8s}  {'armature':>8s}  {'friction':>8s}  {'effort':>8s}")
    for i in range(29):
        lab_j = perm[i]
        print(
            f"  [{i:2d}] {JOINT_NAMES_CANONICAL[i]:<22s}  "
            f"{stiffness[lab_j]:>8.4f}  {damping[lab_j]:>8.4f}  "
            f"{armature[lab_j]:>8.5f}  {friction[lab_j]:>8.5f}  {effort_lim[lab_j]:>8.4f}"
        )

    obs, _ = env.reset()
    print(
        f"[smoke] reset → policy obs {obs['policy'].shape}, "
        f"critic obs {obs['critic'].shape}"
    )

    action_dim = env.unwrapped.action_space.shape[-1] if hasattr(env.unwrapped.action_space, "shape") else cfg.action_space
    with ExitStack() as stack:
        writer = None
        if args.video_path:
            inner.render()
            for warmup_step in range(10):
                inner.sim.render()
            args.video_path.parent.mkdir(parents=True, exist_ok=True)
            writer = stack.enter_context(imageio.get_writer(
                str(args.video_path), format="FFMPEG", mode="I", fps=1.0 / inner.step_dt,
                codec="libx264", pixelformat="yuv420p", macro_block_size=1,
            ))
        for step in range(args.steps):
            action = torch.zeros(
                (args.num_envs, action_dim), device=inner.device, dtype=torch.float32
            )
            obs, reward, terminated, truncated, info = env.step(action)
            any_nan = (
                torch.isnan(obs["policy"]).any().item()
                or torch.isnan(obs["critic"]).any().item()
                or torch.isnan(reward).any().item()
            )
            if any_nan:
                raise RuntimeError(f"step {step}: NaN detected in obs/reward")
            if writer is not None:
                frame = inner.render()
                if frame is None or frame.size == 0:
                    raise RuntimeError(f"step {step}: renderer returned an empty frame")
                writer.append_data(frame)
    print(f"[smoke] OK — {args.steps} steps, no NaN")
    if args.video_path:
        print(f"[smoke] Video saved: {args.video_path.resolve()}")

    env.close()
    app.close()


if __name__ == "__main__":
    main()
