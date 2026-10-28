"""Common deterministic PhysX evaluation for the fixed-object comparison."""
import json
from pathlib import Path


def main():
    from .train import parser, evaluate
    from isaaclab.app import AppLauncher
    p=parser()
    p.set_defaults(num_envs=12,eval_episodes=120,eval_max_steps=18000,
                   benchmark_single_object=True,sensor_noise=0.)
    p.add_argument('--algorithm',choices=['supermimic','sapg'],required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--sapg-config',type=Path)
    AppLauncher.add_app_launcher_args(p)
    args=p.parse_args()
    args.headless=True
    args.out.mkdir(parents=True,exist_ok=True)
    import os,tempfile
    runtime_tmp=args.out.resolve()/'runtime_tmp'; runtime_tmp.mkdir(exist_ok=True)
    os.environ['TMPDIR']=str(runtime_tmp); tempfile.tempdir=None
    app=AppLauncher(args).app
    try:
        import torch
        import numpy as np
        import random
        from .adapter import configure,snapshot
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        env,schema,context,cfg,metadata=configure(args)
        if args.algorithm=='supermimic':
            from .policy import load_actor
            actor=load_actor(args.checkpoint,args.device)
            if actor.metadata['asset_fingerprints'] != metadata['asset_fingerprints']:
                raise ValueError('Evaluation asset differs from trained actor')
            class Policy:
                def act(self,state): return actor(state[:,:schema['obs_dim']])
        else:
            from omegaconf import OmegaConf
            from rl_games.torch_runner import Runner
            from isaacsimenvs.utils.rlgames_utils import register_rlgames_env
            saved=OmegaConf.to_container(OmegaConf.load(args.sapg_config),resolve=True)
            agent=saved['agent']
            agent['params']['config'].update(num_actors=args.num_envs,device=args.device,device_name=args.device)
            wrapped=register_rlgames_env(env,rl_device=args.device,
                clip_obs=agent['params']['env'].get('clip_observations',float('inf')),
                clip_actions=agent['params']['env'].get('clip_actions',float('inf')))
            runner=Runner(); runner.load(agent); runner.reset()
            player=runner.create_player(); player.restore(str(args.checkpoint))
            player.has_batch_dimension=True; player.reset()
            if player.intr_reward_coef_embd is not None:
                player.intr_reward_coef_embd[:]=player.intr_reward_coef_embd[-1:].clone()
            class Policy:
                def act(self,state):
                    obs=state[:,:schema['obs_dim']]
                    if player.intr_reward_coef_embd is not None:
                        obs=torch.cat([obs,player.intr_reward_coef_embd],dim=-1)
                    return player.get_action(obs,is_deterministic=True)
                def reset_done(self,done):
                    if player.is_rnn:
                        for state in player.states: state[:,done,:]=0.
        env.reset(seed=args.seed)
        report=evaluate(env,Policy(),args,snapshot)
        report.update(algorithm=args.algorithm,checkpoint=str(args.checkpoint),seed=args.seed,
                      asset_fingerprints=metadata['asset_fingerprints'],success_tolerance=args.success_tolerance)
        path=args.out/'evaluation.json'; path.write_text(json.dumps(report,indent=2))
        print(json.dumps(report),flush=True)
        if report['status']!='complete': raise RuntimeError('Evaluation episode quota not met')
        env.close()
    finally: app.close()

if __name__=='__main__': main()
