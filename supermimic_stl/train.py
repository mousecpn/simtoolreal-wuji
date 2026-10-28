"""Opt-in Isaac Lab training/evaluation entry point. Run from repository root."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot',choices=['kuka-sharpa','tianji-wuji'],default='kuka-sharpa')
    p.add_argument('--mode',choices=['train','eval','validate'],default='train')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    p.add_argument('--num-envs',type=int,default=12288)
    p.add_argument('--iterations',type=int,default=10000)
    p.add_argument('--collect-steps',type=int,default=16)
    p.add_argument('--warmup-steps',type=int,default=128)
    p.add_argument('--replay-steps',type=int,default=1024)
    p.add_argument('--world-updates',type=int,default=10)
    p.add_argument('--policy-updates',type=int,default=1)
    p.add_argument('--model-pretrain',type=int,default=100)
    p.add_argument('--model-batch',type=int,default=128)
    p.add_argument('--policy-batch',type=int,default=12288)
    p.add_argument('--imagination-microbatch',type=int,default=512)
    p.add_argument('--replay-device',default='cpu')
    p.add_argument('--checkpoint-replay',action='store_true')
    p.add_argument('--benchmark-single-object',action='store_true')
    p.add_argument('--horizon',type=int,default=32)
    p.add_argument('--window',type=int,default=8)
    p.add_argument('--hidden',type=int,default=512)
    p.add_argument('--policy-arch',choices=['legacy','sapg'],default='sapg')
    p.add_argument('--rnn-burnin',type=int,default=16)
    p.add_argument('--policy-hidden',type=int,default=256)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--reward-scale',type=float,default=.01)
    p.add_argument('--action-noise',type=float,default=.1)
    p.add_argument('--sensor-noise',type=float,default=.01,
                   help='IID noise on normalized policy observations, added identically in simulation and imagination')
    p.add_argument('--save-every',type=int,default=100)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--assets-per-type',type=int,default=1)
    p.add_argument('--tool-types',nargs='+',default=['hammer','screwdriver','marker','spatula','eraser','brush'])
    p.add_argument('--success-tolerance',type=float,default=.075)
    p.add_argument('--max-goals',type=int,default=50)
    p.add_argument('--episode-steps',type=int,default=600)
    p.add_argument('--eval-episodes',type=int,default=32)
    p.add_argument('--eval-max-steps',type=int,default=3000)
    return p


def main():
    from isaaclab.app import AppLauncher
    p = parser()
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    for key in ('num_envs','iterations','collect_steps','replay_steps','world_updates','policy_updates',
                'model_batch','policy_batch','imagination_microbatch','horizon','window','hidden','policy_hidden','save_every',
                'assets_per_type','episode_steps','eval_episodes','eval_max_steps'):
        if getattr(args,key)<1:
            p.error(f'{key} must be positive')
    if not args.resume and (args.warmup_steps<args.window or args.replay_steps<max(args.warmup_steps,args.window)):
        p.error('Require replay-steps >= warmup-steps >= window')
    if args.model_pretrain<0 or args.reward_scale<=0 or args.lr<=0 or args.success_tolerance<=0:
        p.error('Invalid pretraining, learning rate, reward scale or tolerance')
    if args.policy_arch=='sapg' and (args.rnn_burnin<1 or args.warmup_steps<args.rnn_burnin+1 or args.replay_steps%16):
        p.error('SAPG architecture needs positive burnin, warmup >= burnin+1, replay capacity divisible by 16')
    if args.mode!='train' and args.resume is None:
        p.error('eval/validate requires --resume')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=True)
    if any(args.out.iterdir()) and (args.out/'metadata.json').exists():
        p.error('Use a fresh output directory; existing run outputs are never overwritten')
    # Remote /tmp is quota-limited. Keep this run's generated assets on disk.
    runtime_tmp = args.out/'runtime_tmp'
    runtime_tmp.mkdir(exist_ok=True)
    os.environ['TMPDIR'] = str(runtime_tmp)
    import tempfile
    tempfile.tempdir = None
    args.headless = True
    app = AppLauncher(args).app
    try:
        run(args)
    except BaseException:
        import traceback,sys
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
    finally:
        app.close()


def run(args):
    import numpy as np
    import torch
    from .adapter import configure,snapshot
    from .core import Learner,Replay

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    loaded = None
    if args.resume:
        loaded = torch.load(args.resume,map_location='cpu',weights_only=False)
        if loaded.get('format') not in ('supermimic-stl-v1','supermimic-stl-v2'):
            raise ValueError('Expected a new SuperMimic-STL checkpoint, not PPO/SAPG weights')
        # Reconstruct the environment and architecture rather than silently
        # accepting evaluation defaults that differ from training.
        saved = loaded['args']
        args.policy_arch=saved.get('policy_arch','legacy')
        for key in ('robot','seed','assets_per_type','tool_types','success_tolerance','max_goals',
                    'episode_steps','hidden','policy_hidden','horizon','window','lr','reward_scale',
                    'action_noise','sensor_noise','benchmark_single_object','imagination_microbatch','rnn_burnin'):
            setattr(args,key,saved.get(key,getattr(args,key)))
        if args.mode=='train' and args.num_envs!=saved['num_envs']:
            raise ValueError('Resume requires the original num-envs; evaluation may use fewer')
        if args.mode=='train' and args.replay_steps!=saved['replay_steps']:
            raise ValueError('Resume requires the original replay-steps')
    env,schema,context,cfg,metadata = configure(args)
    try:
        obs,_ = env.reset(seed=args.seed)
        state = snapshot(env)
        if state.shape!=(args.num_envs,schema['state_dim']):
            raise AssertionError('State schema mismatch')
        learner = Learner(schema,context.shape[-1],env.cfg.action_space,cfg).to(args.device)
        replay = Replay(args.replay_steps,args.num_envs,state.shape[-1],env.cfg.action_space,
                        context.shape[-1],args.replay_device,sample_device=args.device,
                        rnn_size=cfg.native_spec['actor']['rnn']['units'] if learner.native else 0)
        if learner.native and args.mode=='train' and args.num_envs%6:
            raise ValueError('Native SAPG exploration architecture requires six equal environment groups')
        conditions=(torch.linspace(50.,0.,6,device=args.device)[torch.arange(args.num_envs,device=args.device)*6//args.num_envs,None]
                    if learner.native else None)
        started = time.monotonic()
        sim_steps = 0
        start_iteration = 0
        if loaded:
            if loaded['metadata']!=metadata:
                raise ValueError('Environment asset fingerprints/schema/config changed; refuse incompatible checkpoint')
            learner.load_state_dict(loaded['learner']['model'])
            if args.mode=='train':
                for k,opt in learner.optimizers.items():
                    opt.load_state_dict(loaded['learner']['optimizers'][k])
                if loaded.get('replay') is not None:
                    replay.load_state_dict(loaded['replay'])
                # Simulator episodes restart on resume. Mark a boundary so no
                # world-model window joins old episodes to fresh resets.
                if replay.total:
                    replay.data['truncated'][(replay.total-1)%replay.capacity].fill_(1)
                sim_steps = loaded['sim_steps']
                start_iteration = loaded['iteration']
                random.setstate(loaded['rng']['python'])
                np.random.set_state(loaded['rng']['numpy'])
                torch.set_rng_state(loaded['rng']['torch'])
                torch.cuda.set_rng_state_all(loaded['rng']['cuda'])
        source_dir = Path(__file__).parent
        metadata_out = dict(metadata,network_parameters={name:sum(p.numel() for p in getattr(learner,name).parameters()) for name in ('actor','critic','world')},source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                                   for p in source_dir.glob('*.py')})
        (args.out/'metadata.json').write_text(json.dumps(metadata_out,indent=2))
        (args.out/'args.json').write_text(json.dumps(vars(args),default=str,indent=2))
        last_metrics = {}
        def collect(steps,random_actions=False):
            nonlocal state,sim_steps
            totals = dict(real_reward=0.,goal_hits=0,completed_episodes=0,terminated=0,timeouts=0,pre_reset_timeouts_verified=0)
            with torch.no_grad():
                for _ in range(steps):
                    learner.update_input_stats(state)
                    hidden=learner.rollout_hidden(state)
                    policy_action=learner.act(state,noisy=True,condition=conditions) if learner.native else None
                    if random_actions:
                        action = torch.rand(args.num_envs,env.cfg.action_space,device=args.device)*2-1
                    elif learner.native:
                        action=policy_action
                    else:
                        action = learner.act(state,noisy=True)
                        action = (action+torch.randn_like(action)*cfg.action_noise).clamp(-1,1)
                    _,reward,term,trunc,_ = env.step(action)
                    extra=dict(condition=conditions) if learner.native else {}
                    replay.add(state=state,action=action,context=context,next_state=env.mbrl_terminal_state,
                               reward=reward,terminated=term,truncated=trunc,switch=env.mbrl_success,
                               rnn_state=hidden,**extra)
                    learner.reset_done(term|trunc)
                    state = snapshot(env)
                    if trunc.any():
                        timer = slice(*schema['slices']['steps'])
                        if not (env.mbrl_terminal_state[trunc,timer] >= args.episode_steps).all():
                            raise AssertionError('Timeout target was captured after reset')
                        if not (state[trunc,timer] == 0).all():
                            raise AssertionError('Expected post-step timeout environments to reset')
                        totals['pre_reset_timeouts_verified'] += int(trunc.sum())
                    sim_steps += args.num_envs
                    totals['real_reward'] += float(reward.mean())/steps
                    totals['goal_hits'] += int(env.mbrl_success.sum())
                    totals['completed_episodes'] += int((term|trunc).sum())
                    totals['terminated'] += int(term.sum())
                    totals['timeouts'] += int(trunc.sum())
            return totals

        def save(iteration):
            payload = dict(format='supermimic-stl-v2' if learner.native else 'supermimic-stl-v1',iteration=iteration,sim_steps=sim_steps,
                           args=vars(args),metadata=metadata,learner=learner.checkpoint(),replay=replay.state_dict() if args.checkpoint_replay else None,
                           rng=dict(python=random.getstate(),numpy=np.random.get_state(),
                                    torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
            tmp = args.out/'latest.tmp'
            torch.save(payload,tmp)
            tmp.replace(args.out/'latest.pt')
            torch.save(dict(format='supermimic-stl-actor-v2' if learner.native else 'supermimic-stl-actor-v1',actor=learner.actor.state_dict(),
                            mean=learner.mean[:schema['obs_dim']].cpu(),
                            scale=learner.scale[:schema['obs_dim']].cpu(),
                            metadata=metadata),args.out/'actor.pt')
            if iteration % args.save_every == 0:
                import shutil
                shutil.copyfile(args.out/'actor.pt',args.out/f'actor-{iteration:06d}.pt')

        if args.mode=='eval':
            report = evaluate(env,learner,args,snapshot)
            (args.out/'evaluation.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(report),flush=True)
            return
        if args.mode=='validate':
            collect(max(args.window*2,args.collect_steps))
            with torch.no_grad():
                report = validate_world(learner,replay,args.model_batch,args.window)
            report['unseen_simulator_transitions'] = True
            report['optimizer_updates'] = 0
            (args.out/'world_validation.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(report),flush=True)
            return
        if replay.total == 0:
            collect(args.warmup_steps,random_actions=True)
        if not loaded:
            samples = replay.sample(min(8192,args.warmup_steps*args.num_envs))['state'][0]
            learner.calibrate(samples)
            for i in range(args.model_pretrain):
                last_metrics = learner.fit_world(replay.sample(args.model_batch,args.window))
                if (i+1)%20==0:
                    print(json.dumps(dict(phase='model_pretrain',update=i+1,**last_metrics)),flush=True)
        history = args.out/'metrics.jsonl'
        for iteration in range(start_iteration+1,start_iteration+args.iterations+1):
            cosine = .03+.97*.5*(1+math.cos(math.pi*min(iteration/250000,1.)))
            for name,optimizer in learner.optimizers.items():
                base_lr=cfg.native_spec['learning_rate'] if learner.native and name!='world' else cfg.lr
                for group in optimizer.param_groups:
                    group['lr'] = base_lr*cosine
            real = collect(args.collect_steps)
            for _ in range(args.world_updates):
                last_metrics = learner.fit_world(replay.sample(args.model_batch,args.window))
            for _ in range(args.policy_updates):
                policy_metrics = learner.improve(replay.sample_policy(args.policy_batch,cfg.rnn_burnin))
            metrics = dict(peak_torch_gpu_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,iteration=iteration,sim_steps=sim_steps,elapsed_seconds=time.monotonic()-started,
                           **real,**last_metrics,**policy_metrics)
            if not all(np.isfinite(v) for v in metrics.values()):
                raise FloatingPointError('Nonfinite training metrics')
            with history.open('a') as f:
                f.write(json.dumps(metrics)+'\n')
            print(json.dumps(metrics),flush=True)
            if iteration%args.save_every==0:
                save(iteration)
        save(start_iteration+args.iterations)
        (args.out/'result.json').write_text(json.dumps(dict(status='completed',**metrics),indent=2))
    finally:
        env.close()


def validate_world(learner,replay,batch_size,window):
    import torch
    from torch.nn import functional as F
    batch = replay.sample(batch_size,window)
    z = learner.normalize(batch['state'][0])
    alive = torch.ones(batch_size,device=z.device)
    rows = []
    for t in range(window):
        z,r,term,hit = learner.step_model(z,batch['action'][t],batch['context'][t],
                                        batch['switch'][t,:,0],batch['next_state'][t,:,learner.sl('goal')])
        target = learner.normalize(batch['next_state'][t])
        denom = alive.sum().clamp_min(1)
        rows.append(dict(horizon=t+1,valid_windows=int(alive.sum()),
                         normalized_state_mse=float(((z-target).square().mean(-1)*alive).sum()/denom),
                         scaled_reward_mse=float(((r-batch['reward'][t,:,0]*learner.cfg.reward_scale).square()*alive).sum()/denom),
                         termination_bce=float((F.binary_cross_entropy_with_logits(term,batch['terminated'][t,:,0],reduction='none')*alive).sum()/denom),
                         goal_switch_bce=float((F.binary_cross_entropy_with_logits(hit,batch['switch'][t,:,0],reduction='none')*alive).sum()/denom)))
        alive *= (1-batch['terminated'][t,:,0])*(1-batch['truncated'][t,:,0])
    return dict(rollouts=rows,goal_switch_conditioning='logged exogenous goal proposals; switch prediction BCE reported separately')


def evaluate(env,learner,args,snapshot):
    import torch
    # Per-env quotas avoid overcounting fast resetting environments.
    quota = torch.full((args.num_envs,),args.eval_episodes//args.num_envs,device=args.device,dtype=torch.long)
    quota[:args.eval_episodes%args.num_envs] += 1
    counts = torch.zeros_like(quota)
    returns = torch.zeros(args.num_envs,device=args.device)
    lengths = torch.zeros_like(counts)
    episodes = []
    with torch.no_grad():
        for step in range(args.eval_max_steps):
            state = snapshot(env)
            _,r,term,trunc,_ = env.step(learner.act(state))
            returns += r
            lengths += 1
            for i in ((term|trunc)&(counts<quota)).nonzero().flatten().tolist():
                episodes.append(dict(env=i,return_value=float(returns[i]),steps=int(lengths[i]),
                                     goals=int(env.mbrl_successes[i]),timeout=bool(trunc[i]),
                                     all_goals=bool(env.mbrl_successes[i]>=args.max_goals)))
                counts[i] += 1
            done = term|trunc
            if hasattr(learner,'reset_done'):
                learner.reset_done(done)
            returns[done] = 0
            lengths[done] = 0
            if bool((counts>=quota).all()):
                break
    n = len(episodes)
    return dict(status='complete' if n==args.eval_episodes else 'incomplete_step_cap',
                requested_episodes=args.eval_episodes,completed_episodes=n,
                simulator_steps=step+1,episodes=episodes,
                mean_return=sum(e['return_value'] for e in episodes)/n if n else None,
                mean_goals=sum(e['goals'] for e in episodes)/n if n else None,
                all_goals_success_rate=sum(e['all_goals'] for e in episodes)/n if n else None,
                physics='Isaac Lab / PhysX; deterministic actor, clean/no-delay v1')


if __name__=='__main__':
    main()
