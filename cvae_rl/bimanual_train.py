"""Train a frozen single-arm CVAE as a two-arm route-F latent skill policy.

The two CVAE decoder copies, encoder, LSTM and observation normalization remain
frozen.  The only actor parameters optimized here are the centralized latent
coordinator and its exploration standard deviations.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/home/pinhao/simtoolreal'))
    p.add_argument('--repo', type=Path, default=Path('/home/pinhao/sapg'))
    p.add_argument('--student', type=Path,
                   default=Path('outputs/cvae_distillation_20261029/train/best.pt'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--num-envs', type=int, default=128)
    p.add_argument('--horizon', type=int, default=32)
    p.add_argument('--iterations', type=int, default=10000)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--seed', type=int, default=20261030)
    p.add_argument('--latent-std', type=float, default=.1)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--base-layout', choices=('opposed', 'table_facing', 'side_by_side', 'long_edge'),
                   default='long_edge')
    return p


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_state(policy):
    """Hashes all modules that route F promises not to alter."""
    return {
        'cvae': {k: v.detach().cpu().clone() for k, v in policy.model.state_dict().items()},
        'decoders': [{k: v.detach().cpu().clone() for k, v in d.state_dict().items()}
                     for d in policy.decoders],
    }


def frozen_unchanged(policy, before):
    import torch
    return (all(torch.equal(v.cpu(), before['cvae'][k]) for k, v in policy.model.state_dict().items())
            and all(all(torch.equal(v.cpu(), before['decoders'][i][k]) for k, v in d.state_dict().items())
                    for i, d in enumerate(policy.decoders)))


def main(a):
    import numpy as np
    import torch
    from scripts.cvae_policy import CVAEPolicy
    from kuka_sharpa_anypose.env import KukaSharpaAnyPoseCfg, KukaSharpaAnyPoseEnv
    from cvae_rl.bimanual import local_observations
    from cvae_rl.core import Config, History, Learner, gae

    if a.num_envs % 4:
        raise ValueError('--num-envs must be divisible by four exploration groups')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    a.root = a.root.resolve()
    a.repo = a.repo.resolve()
    if not a.student.is_absolute():
        a.student = a.root / a.student
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    cfg = KukaSharpaAnyPoseCfg()
    cfg.scene.num_envs = a.num_envs
    cfg.seed = a.seed
    cfg.log_dir = str(a.out)
    cfg.base_layout = a.base_layout
    env = KukaSharpaAnyPoseEnv(cfg)
    device = env.device
    saved = torch.load(a.student, map_location='cpu', weights_only=True)
    pretrained = CVAEPolicy(**saved['model_config'])
    pretrained.load_state_dict(saved['model'])
    lc = Config(route='F', arms=2, global_dim=290, groups=4, hidden=a.hidden,
                latent_std=a.latent_std, action_std=.01, lr=a.lr, decoder_lr=0.,
                adapt_latent=True, adapt_action=False)
    learner = Learner(pretrained, lc, device)
    learner.policy.train()
    learner.policy.model.eval()
    learner.policy.decoders.eval()
    before = frozen_state(learner.policy)
    if any(p.requires_grad for p in learner.policy.model.parameters()):
        raise RuntimeError('Route F must freeze the pretrained CVAE')
    if any(p.requires_grad for p in learner.policy.decoders.parameters()):
        raise RuntimeError('Route F must freeze both decoders')
    source_hashes = {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name('bimanual.py'),
                                                 Path(__file__).with_name('core.py'))}
    manifest = dict(route='F', base_layout=a.base_layout, seed=a.seed, args=vars(a),
                    learner=asdict(lc), student_checkpoint=str(a.student),
                    student_sha256=sha256(a.student), student_epoch=saved.get('epoch'),
                    source_sha256=source_hashes,
                    frozen='CVAE encoder/LSTM/norm/decoder and both arm decoder copies')
    (a.out/'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    obs, _ = env.reset()
    history = History(local_observations(env))
    steps = 0
    start = time.monotonic()
    coordinator_before = {k: v.detach().cpu().clone() for k, v in learner.policy.coordinator.state_dict().items()}
    critic_before = {k: v.detach().cpu().clone() for k, v in learner.critic.state_dict().items()}

    def checkpoint(name, metrics):
        unchanged = frozen_unchanged(learner.policy, before)
        if not unchanged:
            raise RuntimeError('Frozen CVAE or decoder weights changed')
        coordinator_changed = any(not torch.equal(v.cpu(), coordinator_before[k])
                                  for k, v in learner.policy.coordinator.state_dict().items())
        critic_changed = any(not torch.equal(v.cpu(), critic_before[k])
                             for k, v in learner.critic.state_dict().items())
        learner.save(a.out/name, steps=steps, student_checkpoint=str(a.student),
                     student_sha256=manifest['student_sha256'], base_layout=a.base_layout,
                     frozen_unchanged=unchanged, coordinator_changed=coordinator_changed,
                     critic_changed=critic_changed, last_metrics=metrics)

    for iteration in range(1, a.iterations + 1):
        rows = []
        reward_sums = {}
        ended = []
        for _ in range(a.horizon):
            groups = torch.arange(a.num_envs, device=device) * 4 // a.num_envs
            with torch.no_grad():
                features = learner.policy.encode(history.data)
                sample = learner.policy.sample(obs['critic'], features, groups)
                value = learner.value(obs['critic'], groups)
            row = {k: v.clone() for k, v in sample.items() if k != 'action'}
            row.update(features=features.clone(), global_=obs['critic'].clone(), groups=groups,
                       value=value.clone(), policy_version=torch.full_like(groups, learner.updates))
            next_obs, reward, term, trunc, info = env.step(sample['action'].reshape(a.num_envs, 58))
            # DirectRLEnv has already reset slots when it returns.  For a terminal or
            # timeout state, use zero bootstrap instead of the next episode's state.
            with torch.no_grad():
                next_value = learner.value(next_obs['critic'], groups)
                next_value[term | trunc] = 0.
            row.update(reward=reward.clone(), term=(term | trunc).clone(),
                       trunc=torch.zeros_like(trunc), next_value=next_value.clone())
            rows.append(row)
            metrics = info.get('log', {})
            for name, value_ in metrics.items():
                if hasattr(value_, 'mean'):
                    reward_sums[name] = reward_sums.get(name, 0.) + float(value_.mean())
            done = term | trunc
            for i in torch.where(done)[0].tolist():
                ended.append(dict(success=bool(env.success[i]), handover=bool(env.tracker.transferred[i])))
            history.push(local_observations(env), done)
            obs = next_obs
        batch = {k: torch.stack([r[k] for r in rows]) for k in rows[0]}
        adv, returns = gae(batch['reward'], batch['value'], batch['next_value'], batch['term'], batch['trunc'],
                           lc.gamma, lc.gae_lambda)
        batch['adv'] = adv
        batch['returns'] = returns
        batch['global'] = batch.pop('global_')
        batch = {k: v.flatten(0, 1) for k, v in batch.items()}
        metrics = learner.update(batch)
        steps += a.horizon * a.num_envs
        metrics.update(iteration=iteration, steps=steps, seconds=time.monotonic()-start,
                       reward=float(batch['reward'].mean()), ended_episodes=len(ended),
                       ended_successes=sum(x['success'] for x in ended),
                       ended_handovers=sum(x['handover'] for x in ended),
                       **{f'env/{k}': v/a.horizon for k, v in reward_sums.items()})
        with (a.out/'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(metrics) + '\n')
        print('TRAIN', json.dumps(metrics), flush=True)
        if iteration % a.save_every == 0 or iteration == a.iterations:
            checkpoint('last.pt', metrics)
    final = dict(status='complete', route='F', iterations=a.iterations, steps=steps,
                 frozen_unchanged=frozen_unchanged(learner.policy, before), updates=learner.updates)
    (a.out/'complete.json').write_text(json.dumps(final, indent=2))
    env.close()


if __name__ == '__main__':
    from isaaclab.app import AppLauncher
    p = parser()
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    app = AppLauncher(args).app
    try:
        main(args)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        app.close()
