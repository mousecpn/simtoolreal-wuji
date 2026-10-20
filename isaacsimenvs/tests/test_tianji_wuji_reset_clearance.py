"""Record repeated, stationary no-DR resets to detect spawn contact impulses.

Run on an RTX GPU with --headless --enable_cameras. This holds the reset joint
pose, so any tool launch is a reset/physics effect, not exploration.
"""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--legacy_pose", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app = AppLauncher(args).app
    import torch
    import numpy as np
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from omegaconf import OmegaConf
    from isaaclab.envs.utils.spaces import replace_env_cfg_spaces_with_strings, replace_strings_with_env_cfg_spaces
    from isaaclab.utils import replace_strings_with_slices
    from isaaclab.sensors import Camera, CameraCfg
    import isaaclab.sim as sim_utils
    from isaacsimenvs.tasks.tianji_wuji.config import TianjiWujiEnvCfg
    from isaacsimenvs.tasks.tianji_wuji.assets import DEFAULT_ARM_POSE, RAISED_ARM_POSE
    from isaacsimenvs.tasks.tianji_wuji.env import TianjiWujiEnv

    saved = replace_strings_with_slices(OmegaConf.to_container(OmegaConf.load(args.config), resolve=True))
    cfg = replace_env_cfg_spaces_with_strings(TianjiWujiEnvCfg())
    cfg.from_dict(saved['env'])
    cfg = replace_strings_with_env_cfg_spaces(cfg)
    cfg.robot.arm_default_joint_pos = DEFAULT_ARM_POSE if args.legacy_pose else RAISED_ARM_POSE
    cfg.scene.num_envs = 24
    cfg.assets.num_assets_per_type = 2
    cfg.seed = 42
    env = TianjiWujiEnv(cfg)
    camera = Camera(CameraCfg(
        prim_path='/World/ResetCheckCamera', width=960, height=540,
        data_types=['rgb'], update_period=0,
        spawn=sim_utils.PinholeCameraCfg(focal_length=24, horizontal_aperture=20.955, clipping_range=(.1, 100)),
    ))
    env.sim.reset()
    origin = env.scene.env_origins[5]
    camera.set_world_poses_from_view(
        (origin + torch.tensor((.7, -1.4, 1.3), device=env.device)).unsqueeze(0),
        (origin + torch.tensor((0., .1, .75), device=env.device)).unsqueeze(0),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    frames = []
    trials = []
    for trial in range(5):
        env.reset()
        # Arm action zero keeps the previous target. Hand zero would close to
        # joint midpoints, so invert the absolute mapping to HOLD reset angles.
        action_lab = torch.zeros_like(env._cur_targets)
        hand = env._hand_joint_ids
        action_lab[:, hand] = 2 * (env._prev_targets[:, hand] - env._hand_lower) / (env._hand_upper - env._hand_lower) - 1
        actions = action_lab[:, env._perm_lab_to_canon]
        peak_up = torch.zeros(24, device=env.device)
        peak_speed = torch.zeros_like(peak_up)
        peak_rise = torch.zeros_like(peak_up)
        initial_z = env.object.data.root_pos_w[:, 2].clone()
        initial_tip_z = env.robot.data.body_pos_w[:, env._fingertip_body_ids, 2] - env.scene.env_origins[:, 2:3]
        for step in range(60):
            env.step(actions)
            velocity = env.object.data.root_lin_vel_w
            peak_up = torch.maximum(peak_up, velocity[:, 2])
            peak_speed = torch.maximum(peak_speed, velocity.norm(dim=-1))
            peak_rise = torch.maximum(peak_rise, env.object.data.root_pos_w[:, 2] - initial_z)
            if step % 2 == 0:
                camera.update(env.step_dt * 2)
                frame = Image.fromarray(camera.data.output['rgb'][0].cpu().numpy()[:, :, :3])
                ImageDraw.Draw(frame).text((15, 15), f"{'OLD' if args.legacy_pose else 'RAISED'} reset | holding joint pose | trial {trial + 1}", fill='black')
                frames.append(np.asarray(frame))
        trials.append({
            'trial': trial, 'min_initial_fingertip_z': initial_tip_z.min().item(),
            'peak_upward_speed_per_env': peak_up.tolist(),
            'peak_speed_per_env': peak_speed.tolist(),
            'peak_rise_per_env': peak_rise.tolist(),
            'envs_rising_over_10cm': int((peak_rise > .1).sum()),
        })
    metrics = {'legacy_pose': args.legacy_pose, 'arm_pose': list(cfg.robot.arm_default_joint_pos),
               'num_envs': 24, 'trials': trials}
    (args.out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    imageio.mimwrite(args.out / 'reset.mp4', frames, fps=30)
    imageio.imwrite(args.out / 'first-frame.png', frames[0])
    if not args.legacy_pose:
        assert max(max(t['peak_rise_per_env']) for t in trials) < .05, 'Unexpected reset launch'
        assert max(max(t['peak_upward_speed_per_env']) for t in trials) < .5, 'Unexpected reset impulse'
    print(json.dumps(metrics), flush=True)
    print('RESET_CLEARANCE_CHECK_COMPLETE', flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
