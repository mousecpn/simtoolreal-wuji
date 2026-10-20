"""Run a trained policy and record an mp4 of the rollout.

Unlike ``train.py --test`` (which calls rl_games' ``player.run()`` and gives us
no hook to capture frames), this script builds a manual rollout loop so we can
``camera.update(dt)`` and append rgb frames each step.

    python isaacsimenvs/play_video.py \
        --checkpoint runs/<experiment>/nn/last_<experiment>_ep_<...>_.pth \
        --task Isaacsimenvs-SimToolReal-Direct-v0 \
        --agent rl_games_cfg_entry_point \
        --num_envs 4 --steps 300

Output: ``isaacsimenvs/videos/<task>_rollout.mp4``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


VIDEO_DIR = Path(__file__).resolve().parent / "videos"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to rl_games .pth checkpoint")
    parser.add_argument("--config", help="Saved Hydra config containing env and agent")
    parser.add_argument("--metrics_out", help="Write rollout metrics as JSON")
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--success_tolerance", type=float, default=None)
    parser.add_argument("--zero_exploration", action="store_true")
    parser.add_argument("--task", default="Isaacsimenvs-SimToolReal-Direct-v0", help="Gym task id")
    parser.add_argument(
        "--agent",
        default="rl_games_cfg_entry_point",
        help="Key in gym.register kwargs for the rl_games YAML (PPO vs SAPG).",
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_assets_per_type", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--env_idx", type=int, default=0, help="Which env the camera follows (0..num_envs-1)")
    parser.add_argument("--steps", type=int, default=300, help="Physics steps to record")
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--out", default=None, help="Output mp4 path (default: videos/<task>_rollout.mp4)")
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy (mean)")
    parser.add_argument(
        "--goal_mode",
        default=None,
        help="Override env_cfg.peg_in_hole.goal_mode before instantiation (e.g. 'dense').",
    )
    parser.add_argument("--safe_tianji_start", action="store_true", help="Override an old single-arm Tianji saved config with the raised reset pose")
    parser.add_argument("--exploration_grid", action="store_true", help="Record all six SAPG exploration groups in a 3x2 grid")
    my_args = parser.parse_args()
    if my_args.exploration_grid and (my_args.num_envs != 6 or my_args.no_video or my_args.zero_exploration or my_args.deterministic):
        parser.error("exploration_grid requires six envs, video, stochastic actions, and original exploration conditions")

    from isaaclab.app import AppLauncher

    launcher_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args, _ = launcher_parser.parse_known_args([])
    launcher_args.headless = True
    launcher_args.enable_cameras = not my_args.no_video
    cpu_count = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpu_count:
        os.environ["PXR_WORK_THREAD_LIMIT"] = cpu_count
        os.environ["OPENBLAS_NUM_THREADS"] = cpu_count
        launcher_args.kit_args = f"--/plugins/carb.tasking.plugin/threadCount={int(cpu_count)}"
    app = AppLauncher(launcher_args).app

    import gymnasium as gym
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    import isaacsimenvs  # noqa: F401  triggers gym.register
    from isaacsimenvs.utils.rlgames_utils import register_rlgames_env

    # --- Env ---
    # Direct instantiation (not gym.make) — play loop is manual, no gym
    # step/reset semantics needed, and DirectRLEnv exposes .scene / .sim directly.
    env_cfg = load_cfg_from_registry(my_args.task, "env_cfg_entry_point")
    saved_cfg = None
    if my_args.config:
        from omegaconf import OmegaConf
        from isaaclab.envs.utils.spaces import (
            replace_env_cfg_spaces_with_strings,
            replace_strings_with_env_cfg_spaces,
        )
        from isaaclab.utils import replace_strings_with_slices

        saved_cfg = replace_strings_with_slices(
            OmegaConf.to_container(OmegaConf.load(my_args.config), resolve=True)
        )
        env_cfg = replace_env_cfg_spaces_with_strings(env_cfg)
        env_cfg.from_dict(saved_cfg["env"])
        env_cfg = replace_strings_with_env_cfg_spaces(env_cfg)
    if my_args.safe_tianji_start:
        if my_args.task != "Isaacsimenvs-SimToolReal-TianjiWuji-Direct-v0":
            raise ValueError("safe_tianji_start is only supported for the single-arm Tianji/Wuji task")
        from isaacsimenvs.tasks.tianji_wuji.assets import RAISED_ARM_POSE
        env_cfg.robot.arm_default_joint_pos = RAISED_ARM_POSE
    env_cfg.scene.num_envs = my_args.num_envs
    if my_args.num_assets_per_type is not None:
        env_cfg.assets.num_assets_per_type = my_args.num_assets_per_type
    if my_args.seed is not None:
        env_cfg.seed = my_args.seed
    if my_args.goal_mode is not None and hasattr(env_cfg, "peg_in_hole"):
        env_cfg.peg_in_hole.goal_mode = my_args.goal_mode
    if my_args.success_tolerance is not None:
        env_cfg.termination.eval_success_tolerance = my_args.success_tolerance

    spec = gym.spec(my_args.task)
    mod_name, cls_name = spec.entry_point.split(":")
    env_cls = getattr(importlib.import_module(mod_name), cls_name)
    env = env_cls(cfg=env_cfg)

    # --- Camera sensor ---
    # Spawn with a placeholder pose; re-aim using env_cfg.record_camera_{eye,target}
    # after the env's initial sim.reset.
    camera_cfg = CameraCfg(
        prim_path="/World/RecordCamera",
        update_period=0,
        height=360 if my_args.exploration_grid else 1080,
        width=640 if my_args.exploration_grid else 1920,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 10.0),  # placeholder
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="opengl",
        ),
    )
    cameras = []
    camera_indices = list(range(6)) if my_args.exploration_grid else [my_args.env_idx]
    if not my_args.no_video:
        import copy
        for index in camera_indices:
            cfg = copy.deepcopy(camera_cfg)
            cfg.prim_path = f"/World/RecordCamera_{index}"
            cameras.append(Camera(cfg=cfg))
    camera = cameras[0] if cameras else None
    env.sim.reset()

    # Aim the camera using the env cfg's recording pose (env-local frame).
    env_origin = env.scene.env_origins[my_args.env_idx]
    camera_eye = getattr(env_cfg, "record_camera_eye", env_cfg.viewer.eye)
    camera_target = getattr(env_cfg, "record_camera_target", env_cfg.viewer.lookat)
    eye = env_origin + torch.tensor(camera_eye, device=env.device)
    target = env_origin + torch.tensor(camera_target, device=env.device)
    if camera is not None:
        for sensor, index in zip(cameras, camera_indices):
            origin = env.scene.env_origins[index]
            sensor.set_world_poses_from_view(
                (origin + torch.tensor(camera_eye, device=env.device)).unsqueeze(0),
                (origin + torch.tensor(camera_target, device=env.device)).unsqueeze(0),
            )

    # Step the sim once to flush the pose change through PhysX/Hydra so
    # camera.data.pos_w updates.
    env.sim.step()
    if camera is not None:
        for sensor in cameras:
            sensor.update(0.0)

    print(f"[diag] env_origin = {env_origin.cpu().tolist()}")
    print(f"[diag] camera eye desired = {eye.cpu().tolist()}")
    print(f"[diag] camera target = {target.cpu().tolist()}")
    if camera is not None:
        print(f"[diag] camera pos_w actual = {camera.data.pos_w[0].cpu().tolist()}")
        print(f"[diag] camera quat_w actual = {camera.data.quat_w_world[0].cpu().tolist()}")

    # --- Agent cfg + rl_games wrap ---
    agent_cfg = saved_cfg["agent"] if saved_cfg else load_cfg_from_registry(my_args.task, my_args.agent)
    if my_args.seed is not None:
        agent_cfg["params"]["seed"] = my_args.seed
    import math

    clip_obs = float(agent_cfg["params"]["env"].get("clip_observations", math.inf))
    clip_actions = float(agent_cfg["params"]["env"].get("clip_actions", math.inf))
    wrapped = register_rlgames_env(
        env,
        rl_device=my_args.rl_device,
        clip_obs=clip_obs,
        clip_actions=clip_actions,
    )

    agent_cfg["params"]["config"]["device"] = my_args.rl_device
    agent_cfg["params"]["config"]["device_name"] = my_args.rl_device
    agent_cfg["params"]["config"]["num_actors"] = my_args.num_envs

    # --- Player ---
    from rl_games.torch_runner import Runner

    runner = Runner()
    runner.load(agent_cfg)
    runner.reset()
    player = runner.create_player()
    player.restore(my_args.checkpoint)
    if my_args.zero_exploration and player.intr_reward_coef_embd is not None:
        player.intr_reward_coef_embd[:] = player.intr_reward_coef_embd[-1:].clone()
    # `has_batch_dimension` is only set inside `player.run()`, which we bypass.
    # Our obs are always batched (num_envs, obs_dim), so set it explicitly.
    player.has_batch_dimension = True
    # Allocate LSTM hidden state (no-op for non-RNN models). Without this,
    # `player.get_action` for an LSTM checkpoint crashes inside
    # `network_builder.py` with `len(states)` on None.
    player.reset()

    exploration_labels = None
    exploration_stats = None
    if my_args.exploration_grid:
        if not player.expl_type.startswith("mixed_expl"):
            raise ValueError("exploration_grid requires a mixed-exploration SAPG checkpoint")
        scale = agent_cfg["params"]["config"]["expl_reward_coef_scale"]
        exploration_labels = [
            {"env_id": i, "condition_id": float(player.intr_reward_coef_embd[i, 0]),
             "entropy_coefficient": float(c)}
            for i, c in enumerate(torch.linspace(0.5, 0.0, 6) * scale)
        ]
        exploration_stats = torch.zeros((6, 3), device=player.device)

        def record_distribution(module, inputs, output):
            exploration_stats[:, 0].add_(output["sigmas"].detach().mean(dim=-1))
            exploration_stats[:, 1].add_((output["mus"].detach().abs() > 1).float().mean(dim=-1))
            exploration_stats[:, 2].add_((output["actions"].detach().abs() > 1).float().mean(dim=-1))

        player.model.register_forward_hook(record_distribution)
        print(f"[exploration] {exploration_labels}", flush=True)

    # --- Rollout + capture ---
    obs = player.env_reset(wrapped)
    # Use the policy dt (sim_dt × decimation), not the raw physics dt:
    # the rollout loop advances one policy step (= `decimation` physics steps)
    # per `env.step()`. Capturing every Nth physics step would oversample.
    physics_dt = env.sim.get_physics_dt()
    decimation = int(getattr(env.cfg, "decimation", 1) or 1)
    policy_dt = physics_dt * decimation
    capture_every = max(1, round((1.0 / my_args.video_fps) / policy_dt))

    frames = []
    returns = torch.zeros(my_args.num_envs)
    lengths = torch.zeros(my_args.num_envs, dtype=torch.long)
    episodes = []
    print(f"[play_video] Rolling out {my_args.steps} steps on {my_args.num_envs} envs...", flush=True)
    for step_i in range(my_args.steps):
        action = player.get_action(obs, is_deterministic=my_args.deterministic)
        obs, rew, dones, infos = player.env_step(wrapped, action)
        returns += rew.reshape(-1).cpu()
        lengths += 1
        done_ids = dones.reshape(-1).nonzero(as_tuple=False).flatten()
        for env_id in done_ids.tolist():
            episode = {"env_id": env_id, "return": returns[env_id].item(), "length": lengths[env_id].item()}
            if hasattr(env, "_prev_episode_successes"):
                episode["successes"] = float(env._prev_episode_successes[env_id].item())
            elif "successes" in infos:
                episode["successes"] = float(infos["successes"][env_id].item())
            episodes.append(episode)
        returns[done_ids] = 0
        lengths[done_ids] = 0
        if player.is_rnn:
            for state in player.states:
                state[:, done_ids.to(state.device), :] = 0

        if camera is not None and step_i % capture_every == 0:
            for sensor in cameras:
                sensor.update(capture_every * policy_dt)
            rgb = camera.data.output["rgb"]
            if rgb is not None and rgb.shape[0] > 0:
                frame = rgb[0].cpu().numpy()[:, :, :3]
                if my_args.exploration_grid:
                    import numpy as np
                    from PIL import Image, ImageDraw
                    tiles = []
                    for sensor, label in zip(cameras, exploration_labels):
                        tile = Image.new("RGB", (640, 396), (22, 25, 32))
                        tile.paste(Image.fromarray(sensor.data.output["rgb"][0].cpu().numpy()[:, :, :3]), (0, 36))
                        ImageDraw.Draw(tile).text(
                            (12, 10), f"Group {label['env_id']} | entropy coef {label['entropy_coefficient']:.4g} | sampled actions",
                            fill="white",
                        )
                        tiles.append(np.asarray(tile))
                    frame = np.concatenate([np.concatenate(tiles[:3], axis=1), np.concatenate(tiles[3:], axis=1)], axis=0)
                if not frames:
                    print(
                        f"[diag] first frame: shape={frame.shape} dtype={frame.dtype}"
                        f" min={frame.min()} max={frame.max()} mean={frame.mean():.2f}"
                    )
                    import imageio as _io

                    _debug_png = VIDEO_DIR / "first_frame.png"
                    _debug_png.parent.mkdir(parents=True, exist_ok=True)
                    _io.imwrite(str(_debug_png), frame)
                    print(f"[diag] wrote first-frame png to {_debug_png}")
                frames.append(frame)

    if exploration_stats is not None:
        for label, values in zip(exploration_labels, (exploration_stats / my_args.steps).cpu().tolist()):
            label.update(dict(zip(("mean_action_std", "mean_outside_bounds_fraction", "sample_clipped_fraction"), values)))
        print(f"[exploration] Rollout statistics: {exploration_labels}", flush=True)

    # --- Save ---
    import imageio

    # Strip any "Isaac-" prefix and version suffix for the default filename.
    slug = my_args.task.lower().replace("isaac-", "").rsplit("-v", 1)[0]
    out_path = Path(my_args.out) if my_args.out else VIDEO_DIR / f"{slug}_rollout.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if camera is not None:
        imageio.mimwrite(str(out_path), frames, fps=my_args.video_fps)
        print(f"[play_video] Wrote {len(frames)} frames to {out_path}")
    if my_args.metrics_out:
        metrics_path = Path(my_args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = {
            "checkpoint": str(Path(my_args.checkpoint).resolve()),
            "config": my_args.config,
            "seed": my_args.seed,
            "steps": my_args.steps,
            "num_envs": my_args.num_envs,
            "deterministic": my_args.deterministic,
            "success_tolerance": env.cfg.termination.eval_success_tolerance,
            "zero_exploration": my_args.zero_exploration,
            "exploration_groups": exploration_labels,
            "safe_tianji_start": my_args.safe_tianji_start,
            "episodes": episodes,
            "unfinished_returns": returns.tolist(),
            "unfinished_lengths": lengths.tolist(),
        }
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"[play_video] Wrote metrics for {len(episodes)} episodes to {metrics_path}")

    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
