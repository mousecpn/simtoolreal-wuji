"""Fresh-rollout score-function SAPG for CVAE skills (not PPO/rl_games SAPG).

The encoder and observation statistics stay frozen in this first experiment.
Each arm gets an independent decoder copy; no mirror/parameter sharing is assumed.
"""
import copy
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence


@dataclass
class Config:
    route: str = 'F'
    arms: int = 1
    global_dim: int = 167
    groups: int = 4
    hidden: int = 256
    lr: float = 3e-4
    decoder_lr: float = 3e-5
    gamma: float = .99
    gae_lambda: float = .95
    latent_std: float = .3
    action_std: float = .03
    adapt_latent: bool = True
    adapt_action: bool = False
    std_rate: float = .005
    latent_bounds: tuple = (.03, 1.)
    action_bounds: tuple = (.005, .3)
    max_grad_norm: float = 1.
    update_chunk: int = 128
    target_kl: float = .05
    adapt_learning_rate: bool = True
    algorithm: str = 'fresh_score_function_v1'


def mlp(n, out, hidden):
    return nn.Sequential(nn.Linear(n, hidden), nn.ELU(),
                         nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, out))


class History:
    """Raw sliding windows; reset repeats first observation for each done slot."""
    def __init__(self, first, frames=16):
        self.frames = frames
        self.data = first.unsqueeze(-2).expand(*first.shape[:-1], frames, first.shape[-1]).clone()

    def push(self, obs, done):
        self.data = self.data.roll(-1, dims=-2)
        self.data[..., -1, :] = obs
        self.data[done] = obs[done].unsqueeze(-2).expand_as(self.data[done])
        return self.data


def gae(reward, value, next_value, terminated, truncated, gamma=.99, lam=.95):
    result = torch.zeros_like(reward)
    carry = torch.zeros_like(reward[0])
    for t in reversed(range(len(reward))):
        delta = reward[t] + gamma * (~terminated[t]) * next_value[t] - value[t]
        carry = delta + gamma * lam * (~(terminated[t] | truncated[t])) * carry
        result[t] = carry
    return result, result + value


class SkillPolicy(nn.Module):
    def __init__(self, pretrained, cfg):
        super().__init__()
        if cfg.route not in ('F', 'D', 'J', 'S') or cfg.arms not in (1, 2):
            raise ValueError('Unsupported route or arm count')
        self.cfg = cfg
        self.model = copy.deepcopy(pretrained).requires_grad_(False).eval()
        self.feature_dim = pretrained.config['hidden']
        self.latent_dim = pretrained.config['latent']
        self.action_dim = pretrained.config['action_dim']
        if cfg.route == 'S':
            # Same fixed-window architecture, random actor, frozen norm statistics.
            for module in (self.model.lstm, self.model.norm, self.model.decoder):
                for child in module.modules():
                    if hasattr(child, 'reset_parameters'):
                        child.reset_parameters()
            self.model.lstm.requires_grad_(True)
            self.model.norm.requires_grad_(True)
            self.model.lstm.train()
        self.decoders = nn.ModuleList([copy.deepcopy(self.model.decoder) for _ in range(cfg.arms)])
        self.branches = nn.ModuleList([nn.Linear(cfg.global_dim + cfg.groups, self.action_dim)
                                      for _ in range(cfg.arms)])
        for branch in self.branches:
            nn.init.zeros_(branch.weight); nn.init.zeros_(branch.bias)
        for decoder in self.decoders:
            decoder.requires_grad_(False)
            if cfg.route in ('D', 'J'):
                decoder[6].requires_grad_(True); decoder[8].requires_grad_(True)
            elif cfg.route == 'S':
                decoder.requires_grad_(True)
        self.branches.requires_grad_(cfg.route != 'F')
        self.coordinator = mlp(cfg.global_dim + cfg.groups + cfg.arms * self.feature_dim,
                               cfg.arms * self.latent_dim, cfg.hidden)
        nn.init.zeros_(self.coordinator[-1].weight); nn.init.zeros_(self.coordinator[-1].bias)
        self.coordinator.requires_grad_(cfg.route in ('F', 'J'))
        self.log_std_z = nn.Parameter(torch.full((cfg.groups, cfg.arms, self.latent_dim),
            math.log(cfg.latent_std)), requires_grad=cfg.route in ('F', 'J'))
        self.log_std_u = nn.Parameter(torch.full((cfg.groups, cfg.arms, self.action_dim),
            math.log(cfg.action_std)), requires_grad=cfg.route in ('D', 'J', 'S'))
        # Explicit per-group, per-layer differential entropy targets (can be negative).
        factors = torch.linspace(1., 2., cfg.groups)
        self.register_buffer('target_entropy_z',
            cfg.arms * self.latent_dim * (.5 * math.log(2 * math.pi * math.e)
                                         + (cfg.latent_std * factors).log()))
        self.register_buffer('target_entropy_u',
            cfg.arms * self.action_dim * (.5 * math.log(2 * math.pi * math.e)
                                         + (cfg.action_std * factors).log()))
        self.register_buffer('entropy_coefficients', torch.linspace(.001, 0., cfg.groups))

    def condition(self, g, groups):
        return torch.cat((g, nn.functional.one_hot(groups, self.cfg.groups).to(g.dtype)), -1)

    def encode(self, history):
        shape = history.shape
        return self.model.features(history.reshape(-1, *shape[-2:])).reshape(
            *shape[:-2], self.feature_dim)

    def distributions(self, g, features, groups, skill_z=None):
        conditioned = self.condition(g, groups)
        z_mean = self.coordinator(torch.cat((conditioned, features.flatten(1)), -1)).reshape(
            -1, self.cfg.arms, self.latent_dim)
        zd = Normal(z_mean, self.log_std_z[groups].exp())
        if skill_z is None:
            return zd
        means = []
        for arm, decoder in enumerate(self.decoders):
            z = skill_z[:, arm] if self.cfg.route in ('F', 'J') else torch.zeros_like(skill_z[:, arm])
            means.append(decoder(torch.cat((features[:, arm], z), -1))
                         + (self.branches[arm](conditioned) if self.cfg.route != 'F' else 0))
        return zd, Normal(torch.stack(means, 1), self.log_std_u[groups].exp())

    @torch.no_grad()
    def sample(self, g, features, groups, deterministic=False, intervention=None):
        zd = self.distributions(g, features, groups)
        skill_z = zd.mean if deterministic else zd.sample()
        if self.cfg.route in ('D', 'S') or intervention == 'zero':
            skill_z = torch.zeros_like(skill_z)
        elif intervention == 'random':
            skill_z = torch.randn_like(skill_z) * self.cfg.latent_std
        _, ud = self.distributions(g, features, groups, skill_z)
        raw_action_u = ud.mean if deterministic or self.cfg.route == 'F' else ud.sample()
        logp, _ = self.score(g, features, groups, skill_z, raw_action_u)
        return dict(skill_z=skill_z, raw_action_u=raw_action_u,
                    action=raw_action_u.clamp(-1, 1), old_logp=logp,
                    old_z_mean=zd.mean, old_z_std=zd.stddev,
                    old_u_mean=ud.mean, old_u_std=ud.stddev)

    def score(self, g, features, groups, skill_z, raw_action_u):
        # Augmented actions are constants in the score-function objective.
        zd, ud = self.distributions(g, features, groups, skill_z.detach())
        lz = zd.log_prob(skill_z.detach()).sum((-1, -2))
        lu = ud.log_prob(raw_action_u.detach()).sum((-1, -2))
        logp = (lz if self.cfg.route in ('F', 'J') else 0) + (lu if self.cfg.route != 'F' else 0)
        return logp, (zd.entropy().sum((-1, -2)), ud.entropy().sum((-1, -2)))

    @torch.no_grad()
    def adapt(self):
        for name, enabled, dims, bounds in (
            ('z', self.cfg.adapt_latent and self.cfg.route in ('F', 'J'),
             self.cfg.arms * self.latent_dim, self.cfg.latent_bounds),
            ('u', self.cfg.adapt_action and self.cfg.route != 'F',
             self.cfg.arms * self.action_dim, self.cfg.action_bounds)):
            param = getattr(self, f'log_std_{name}')
            if enabled:
                entropy = (param + .5 * math.log(2 * math.pi * math.e)).sum((-1, -2))
                target = getattr(self, f'target_entropy_{name}')
                param.add_(self.cfg.std_rate * ((target - entropy) / dims).clamp(-1, 1)[:, None, None])
            param.clamp_(math.log(bounds[0]), math.log(bounds[1]))


class Learner:
    def __init__(self, model, cfg, device='cpu'):
        self.cfg = cfg
        self.policy = SkillPolicy(model, cfg).to(device)
        self.critic = mlp(cfg.global_dim + cfg.groups, 1, cfg.hidden).to(device)
        lower = [p for module in (self.policy.decoders, self.policy.branches) for p in module.parameters()
                 if p.requires_grad]
        lower_ids = {id(p) for p in lower}
        upper = [p for p in self.policy.parameters() if p.requires_grad and id(p) not in lower_ids]
        self.actor_optimizer = torch.optim.Adam([
            {'params': upper, 'lr': cfg.lr, 'initial_lr': cfg.lr},
            {'params': lower, 'lr': cfg.decoder_lr, 'initial_lr': cfg.decoder_lr}])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.updates = 0

    def value(self, g, groups):
        return self.critic(self.policy.condition(g, groups)).squeeze(-1)

    def update(self, batch):
        cfg = self.cfg
        if 'policy_version' not in batch or not bool((batch['policy_version'] == self.updates).all()):
            raise RuntimeError('Only one update per fresh rollout: stale/missing policy version')
        count = len(batch['adv'])
        advantage = batch['adv'].detach()
        advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        metrics = dict(policy_loss=0., value_loss=0., recompute_logp_max_error=0.)
        for start in range(0, count, cfg.update_chunk):
            sl = slice(start, start + cfg.update_chunk)
            b = {k: v[sl].detach() for k, v in batch.items()}
            # S's random encoder is trainable; pretrained F/D/J cache frozen features.
            features = self.policy.encode(b['history']) if cfg.route == 'S' else b['features']
            logp, entropy = self.policy.score(b['global'], features, b['groups'],
                                              b['skill_z'], b['raw_action_u'])
            error = float((logp.detach() - b['old_logp']).abs().max())
            metrics['recompute_logp_max_error'] = max(metrics['recompute_logp_max_error'], error)
            if error > .005:
                raise RuntimeError(f'Executed actions/log probabilities disagree: {error}')
            active_entropy = (entropy[0] if cfg.route in ('F', 'J') else 0) + (
                entropy[1] if cfg.route != 'F' else 0)
            loss = -(logp * advantage[sl] + self.policy.entropy_coefficients[b['groups']]
                     * active_entropy).sum() / count
            loss.backward()
            value = self.value(b['global'], b['groups'])
            vloss = .5 * (value - b['returns']).square().sum() / count
            vloss.backward()
            metrics['policy_loss'] += loss.item(); metrics['value_loss'] += vloss.item()
        # Both actor levels are stepped together, once per fresh batch.
        params = [p for p in self.policy.parameters() if p.requires_grad]
        metrics['actor_grad_norm'] = float(nn.utils.clip_grad_norm_(params, cfg.max_grad_norm))
        metrics['decoder_grad_norm'] = math.sqrt(sum(float(p.grad.square().sum())
            for p in self.policy.decoders.parameters() if p.grad is not None))
        metrics['critic_grad_norm'] = float(nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm))
        if not all(math.isfinite(v) for v in metrics.values()):
            raise FloatingPointError(metrics)
        self.actor_optimizer.step(); self.critic_optimizer.step(); self.policy.adapt()
        self.updates += 1
        with torch.no_grad():
            klz = torch.zeros(count, device=advantage.device); klu = klz.clone()
            for start in range(0, count, cfg.update_chunk):
                sl = slice(start, start + cfg.update_chunk)
                b = {k: v[sl] for k, v in batch.items()}
                features = self.policy.encode(b['history']) if cfg.route == 'S' else b['features']
                zd, ud = self.policy.distributions(b['global'], features, b['groups'], b['skill_z'])
                klz[sl] = kl_divergence(Normal(b['old_z_mean'], b['old_z_std']), zd).sum((-1, -2))
                klu[sl] = kl_divergence(Normal(b['old_u_mean'], b['old_u_std']), ud).sum((-1, -2))
            metrics.update(latent_kl=float(klz.mean()), action_kl=float(klu.mean()),
                joint_kl=float(((klz if cfg.route in ('F', 'J') else 0)
                               + (klu if cfg.route != 'F' else 0)).mean()),
                action_saturation=float((batch['raw_action_u'].abs() >= 1).float().mean()))
            for name in ('z', 'u'):
                logs = getattr(self.policy, f'log_std_{name}')
                ent = (logs + .5 * math.log(2 * math.pi * math.e)).sum((-1, -2))
                for group in range(cfg.groups):
                    metrics[f'{name}/group{group}/std'] = float(logs[group].exp().mean())
                    metrics[f'{name}/group{group}/entropy'] = float(ent[group])
                    metrics[f'{name}/group{group}/target_entropy'] = float(
                        getattr(self.policy, f'target_entropy_{name}')[group])
        # Only the NEXT fresh rollout's learning rate changes. No ratio clipping,
        # repeated gradient updates, resampling, or rollback on this batch.
        if cfg.adapt_learning_rate:
            scale = .5 if metrics['joint_kl'] > 2*cfg.target_kl else (
                1.05 if metrics['joint_kl'] < .5*cfg.target_kl else 1.)
            for group in self.actor_optimizer.param_groups:
                group['lr'] = max(group['initial_lr']*.01,
                                  min(group['initial_lr'], group['lr']*scale))
        for i, group in enumerate(self.actor_optimizer.param_groups):
            metrics[f'optimizer/group{i}/next_lr'] = group['lr']
        return metrics

    def save(self, path, **metadata):
        import random
        import numpy as np
        payload = dict(config=asdict(self.cfg), policy=self.policy.state_dict(),
            critic=self.critic.state_dict(), actor_optimizer=self.actor_optimizer.state_dict(),
            critic_optimizer=self.critic_optimizer.state_dict(), updates=self.updates,
            torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
            python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else [], **metadata)
        tmp = path.with_suffix('.tmp'); torch.save(payload, tmp); tmp.replace(path)
