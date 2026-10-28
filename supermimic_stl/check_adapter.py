"""Real-PhysX integration check for goal switching and pre-reset capture.

Creates only its own short-lived environment. This deliberately places the goal
on the object and primes the dwell counter; it is an interface test, not policy
performance. Run through an Isaac-enabled Python from repository root.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import tempfile


def main():
    from isaaclab.app import AppLauncher
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--robot',choices=['kuka-sharpa','tianji-wuji'],default='kuka-sharpa')
    AppLauncher.add_app_launcher_args(parser)
    args=parser.parse_args()
    args.out=args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'adapter_check.json').exists():
        parser.error('Use a fresh output directory')
    temp=args.out/'runtime_tmp'
    temp.mkdir(exist_ok=True)
    os.environ['TMPDIR']=str(temp)
    tempfile.tempdir=None
    args.headless=True
    app=AppLauncher(args).app
    try:
        import torch
        from .adapter import configure,snapshot
        args.seed=42
        args.num_envs=2
        args.assets_per_type=1
        args.tool_types=['hammer']
        args.success_tolerance=.075
        args.max_goals=50
        args.episode_steps=16
        args.horizon=4
        args.window=4
        args.hidden=64
        args.policy_hidden=32
        args.lr=2e-4
        args.reward_scale=.01
        args.action_noise=.1
        args.sensor_noise=.01
        env,schema,context,cfg,metadata=configure(args)
        try:
            env.reset(seed=args.seed)
            # Test a genuine goal-hit branch without learning a policy first.
            pose=env.object.data.root_state_w[:,:7].clone()
            env.goal_viz.write_root_pose_to_sim(pose)
            env._near_goal_steps.fill_(env.cfg.termination.success_steps-1)
            _,reward,term,trunc,_=env.step(torch.zeros(2,env.cfg.action_space,device=args.device))
            assert env.mbrl_success.all(), 'Primed object-aligned goal did not switch'
            assert (env.mbrl_successes==1).all(), 'Goal counters not captured'
            assert not (term|trunc).any(), 'Goal hit incorrectly became an episode terminal'
            goal_slice=slice(*schema['slices']['goal'])
            new_goal=env.mbrl_terminal_state[:,goal_slice]
            assert (new_goal[:,:3]-pose[:,:3]+env.scene.env_origins).abs().max()>1e-5
            # Force a pure time limit on the next step, independently of goal hit.
            env._near_goal_steps.zero_()
            env.episode_length_buf.fill_(args.episode_steps-1)
            _,reward,term,trunc,_=env.step(torch.zeros(2,env.cfg.action_space,device=args.device))
            assert trunc.all() and not term.any(), 'Expected time-limit-only reset'
            timer=slice(*schema['slices']['steps'])
            assert (env.mbrl_terminal_state[:,timer]==args.episode_steps).all()
            assert (snapshot(env)[:,timer]==0).all()
            report=dict(robot=args.robot,goal_switch_verified=True,multi_goal_continuation_verified=True,
                        pre_reset_terminal_capture_verified=True,timeout_reset_verified=True,
                        physics='Isaac Lab / PhysX',synthetic_interface_test_not_policy_success=True)
            (args.out/'adapter_check.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(report),flush=True)
        finally:
            env.close()
    finally:
        app.close()


if __name__=='__main__':
    main()
