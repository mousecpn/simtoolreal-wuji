"""Simulator-independent differentiable Dyna / short-horizon actor-critic.

SuperMimic (arXiv:2610.28480), equations 1--3. Learned reward and termination
heads preserve the simulator's task reward labels; goal resampling is exogenous.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    imagination_microbatch: int = 512
    horizon: int = 32
    window: int = 8
    hidden: int = 512
    policy_hidden: int = 256
    gamma: float = .99
    lam: float = .95
    lr: float = 2e-4
    reward_scale: float = .01
    action_noise: float = .1
    sensor_noise: float = .01
    critic_epochs: int = 5
    critic_minibatches: int = 4
    episode_steps: int = 600
    goal_mode: str = 'delta'
    goal_distance: float = .1
    goal_degrees: float = 90.
    goal_mins: tuple = (-.35, -.2, .6)
    goal_maxs: tuple = (.35, .2, .95)
    goal_region_scale: float = 1.


def mlp(inp, out, width, depth=3):
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(inp, width), nn.ELU()]
        inp = width
    layers.append(nn.Linear(inp, out))
    return nn.Sequential(*layers)


@contextmanager
def frozen(*modules):
    params = [p for m in modules for p in m.parameters()]
    flags = [p.requires_grad for p in params]
    try:
        for p in params:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(params, flags):
            p.requires_grad_(flag)


def quat_normalize(q):
    return F.normalize(q, dim=-1, eps=1e-6)


def quat_mul(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat([aw*bw-(av*bv).sum(-1, keepdim=True),
                      aw*bv+bw*av+torch.cross(av, bv, dim=-1)], -1)


def quat_apply(q, v):
    q = quat_normalize(q)
    uv = 2*torch.cross(q[..., 1:], v, dim=-1)
    return v+q[..., :1]*uv+torch.cross(q[..., 1:], uv, dim=-1)


def sample_goal(goal, cfg):
    lo = goal.new_tensor(cfg.goal_mins)
    hi = goal.new_tensor(cfg.goal_maxs)
    center = (lo+hi)/2
    lo, hi = center+(lo-center)*cfg.goal_region_scale, center+(hi-center)*cfg.goal_region_scale
    if cfg.goal_mode == 'absolute':
        pos = lo+(hi-lo)*torch.rand_like(goal[:, :3])
        quat = quat_normalize(torch.randn_like(goal[:, 3:]))
    else:
        pos = (goal[:, :3]+(2*torch.rand_like(goal[:, :3])-1)*cfg.goal_distance).clamp(lo, hi)
        axis = F.normalize(torch.randn_like(pos), dim=-1)
        angle = (2*torch.rand_like(goal[:, :1])-1)*cfg.goal_degrees*math.pi/180
        dq = torch.cat([(angle/2).cos(), axis*(angle/2).sin()], -1)
        quat = quat_mul(dq, quat_normalize(goal[:, 3:]))
    # Align double cover before interpolation.
    quat = torch.where((quat*goal[:, 3:]).sum(-1, keepdim=True)<0, -quat, quat)
    return torch.cat([pos, quat], -1)


class Replay:
    """Time-major ring, exact pre-reset next states, masked reset boundaries."""
    def __init__(self, capacity, num_envs, state_dim, action_dim, context_dim, device, sample_device=None):
        self.capacity, self.num_envs, self.device = capacity, num_envs, device
        self.sample_device = sample_device or device
        self.total = 0
        dims = dict(state=state_dim, action=action_dim, context=context_dim,
                    next_state=state_dim, reward=1, terminated=1, truncated=1, switch=1)
        self.data = {k: torch.empty(capacity, num_envs, d, device=device) for k,d in dims.items()}

    def add(self, **transition):
        i = self.total % self.capacity
        for key, buf in self.data.items():
            x = transition[key].detach().reshape(self.num_envs, -1)
            if not torch.isfinite(x).all():
                raise FloatingPointError(f'Nonfinite simulator {key}')
            buf[i].copy_(x)
        self.total += 1

    def sample(self, batch, window=1):
        oldest = max(0, self.total-self.capacity)
        if self.total-oldest < window:
            raise ValueError('Not enough contiguous replay steps')
        starts = torch.randint(oldest, self.total-window+1, (batch,), device=self.device)
        envs = torch.randint(self.num_envs, (batch,), device=self.device)
        times = (starts[None, :]+torch.arange(window, device=self.device)[:, None]) % self.capacity
        return {k:v[times, envs[None, :]].to(self.sample_device).clone() for k,v in self.data.items()}

    def state_dict(self):
        # Preserve absolute ring indexing, including partially filled buffers.
        count = min(self.total, self.capacity)
        return dict(total=self.total, data={k:v[:count].cpu() for k,v in self.data.items()})

    def load_state_dict(self, state):
        self.total = state['total']
        for k,v in state['data'].items():
            self.data[k][:len(v)].copy_(v)


def lambda_returns(rewards, discounts, next_values, time_limits, lam):
    """Fixed lambda targets: physical terminals have d=0; time limits bootstrap."""
    ret = next_values[-1]
    result = []
    for t in reversed(range(len(rewards))):
        time_end = time_limits[t]
        ret = rewards[t]+discounts[t]*(time_end*next_values[t]+(1-time_end)*(
            (1-lam)*next_values[t]+lam*ret))
        result.append(ret)
    return torch.stack(list(reversed(result))).detach()


class World(nn.Module):
    def __init__(self, state_dim, action_dim, context_dim, cfg):
        super().__init__()
        self.state_dim = state_dim
        self.net = mlp(state_dim+action_dim+context_dim, state_dim+3, cfg.hidden)
        # Start near identity dynamics and low termination/goal-hit probability.
        nn.init.normal_(self.net[-1].weight, std=.001)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[-2:] = -5.

    def forward(self, z, a, context):
        y = self.net(torch.cat([z, a, context], -1))
        return z+y[:, :self.state_dim], y[:, -3], y[:, -2], y[:, -1]


class Learner(nn.Module):
    def __init__(self, schema, context_dim, action_dim, cfg):
        super().__init__()
        self.schema, self.cfg = schema, cfg
        self.obs_dim, self.state_dim = schema['obs_dim'], schema['state_dim']
        self.world = World(self.state_dim, action_dim, context_dim, cfg)
        self.actor = mlp(self.obs_dim, action_dim, cfg.policy_hidden, 2)
        self.critic = mlp(self.state_dim+context_dim, 1, cfg.policy_hidden, 2)
        nn.init.normal_(self.actor[-1].weight, std=.001)
        nn.init.zeros_(self.actor[-1].bias)
        self.register_buffer('mean', torch.zeros(self.state_dim))
        self.register_buffer('scale', torch.ones(self.state_dim))
        self.optimizers = {
            'world': torch.optim.Adam(self.world.parameters(), lr=cfg.lr),
            'actor': torch.optim.Adam(self.actor.parameters(), lr=cfg.lr, betas=(.9,.95)),
            'critic': torch.optim.Adam(self.critic.parameters(), lr=cfg.lr),
        }

    def sl(self, name):
        return slice(*self.schema['slices'][name])

    def normalize(self, s):
        return (s-self.mean)/self.scale

    def raw(self, z):
        return z*self.scale+self.mean

    @torch.no_grad()
    def calibrate(self, states):
        self.mean.copy_(states.mean(0))
        self.scale.copy_(states.std(0, unbiased=False).clamp_min(.05))
        # Avoid huge normalized reward-counter outliers at first successes.
        for name in ('near_goal', 'successes', 'steps'):
            self.scale[self.sl(name)].clamp_(min=1.)

    def act(self, state, noisy=False):
        obs = self.normalize(state)[:, :self.obs_dim]
        if noisy and self.cfg.sensor_noise:
            obs = obs+torch.randn_like(obs)*self.cfg.sensor_noise
        return self.actor(obs).tanh()

    def value(self, z, context):
        return self.critic(torch.cat([z, context], -1)).squeeze(-1)

    def project(self, predicted, old, switch, next_goal=None):
        """Goal random walk is sampled externally, not regressed to its mean.

        Simulator fitting uses logged hard goal switches/proposals. Imagination
        uses learned soft goal-hit probabilities and fresh random proposals.
        """
        goal = old[:, self.sl('goal')]
        if next_goal is None:
            next_goal = sample_goal(goal, self.cfg)
        gate = switch.reshape(-1, 1)
        new_goal = goal+gate*(next_goal-goal)
        new_goal = torch.cat([new_goal[:, :3], quat_normalize(new_goal[:, 3:])], -1)
        obj = predicted[:, self.sl('object')]
        obj = torch.cat([obj[:, :3], quat_normalize(obj[:, 3:7]), obj[:, 7:]], -1)
        updates = {'goal':new_goal, 'object':obj}
        offsets = old[:, self.sl('kp_offsets')].reshape(-1, 4, 3)
        obj_kp = obj[:, None, :3]+quat_apply(obj[:, None, 3:7].expand(-1,4,-1), offsets)
        goal_kp = new_goal[:, None, :3]+quat_apply(new_goal[:, None, 3:].expand(-1,4,-1), offsets)
        # Rebuild exact goal-dependent observation fields in both obs groups.
        for prefix in ('actor.', 'critic.'):
            updates[prefix+'keypoints_rel_goal'] = (obj_kp-goal_kp).flatten(1)
            updates[prefix+'object_rot'] = obj[:, [4,5,6,3]]
            updates[prefix+'object_scales'] = old[:, self.sl(prefix+'object_scales')]
            palm = predicted[:, self.sl(prefix+'palm_pos')]
            updates[prefix+'keypoints_rel_palm'] = (obj_kp-palm[:,None,:]).flatten(1)
        # Immutable episode context is copied exactly.
        for name in ('kp_offsets', 'object_init_z', 'table_z'):
            updates[name] = old[:, self.sl(name)]
        # Concatenate instead of in-place writes: preserve BPTT autograd versions.
        parts = []
        for name in self.schema['order']:
            parts.append(updates.get(name, predicted[:, self.sl(name)]))
        return torch.cat(parts, -1)

    def step_model(self, z, action, context, switch=None, goal=None):
        raw_next, reward, term_logits, switch_logits = self.world(z, action, context)
        gate = switch_logits.sigmoid() if switch is None else switch
        state = self.project(self.raw(raw_next), self.raw(z), gate, goal)
        return self.normalize(state), reward, term_logits, switch_logits

    def fit_world(self, batch):
        z = self.normalize(batch['state'][0])
        alive = torch.ones_like(batch['reward'][0,:,0])
        losses = []
        errors = []
        for t in range(len(batch['action'])):
            next_state = batch['next_state'][t]
            z, reward, term, hit = self.step_model(
                z, batch['action'][t], batch['context'][t],
                batch['switch'][t,:,0], next_state[:,self.sl('goal')])
            target = self.normalize(next_state)
            state_loss = (z-target).square().mean(-1)
            reward_loss = (reward-batch['reward'][t,:,0]*self.cfg.reward_scale).square()
            term_loss = F.binary_cross_entropy_with_logits(term, batch['terminated'][t,:,0], reduction='none')
            hit_loss = F.binary_cross_entropy_with_logits(hit, batch['switch'][t,:,0], reduction='none')
            losses.append(((state_loss+reward_loss+term_loss+hit_loss)*alive).sum()/alive.sum().clamp_min(1))
            errors.append((state_loss.detach()*alive).sum()/alive.sum().clamp_min(1))
            alive = alive*(1-batch['terminated'][t,:,0])*(1-batch['truncated'][t,:,0])
        loss = torch.stack(losses).mean()
        self._update('world', loss, self.world.parameters())
        return dict(world_loss=float(loss.detach()), prediction_mse=float(torch.stack(errors).mean()))

    def _update(self, name, loss, params, element_clip=False):
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite {name} loss')
        params = list(params)
        opt = self.optimizers[name]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        if element_clip:
            for p in params:
                if p.grad is not None:
                    p.grad.clamp_(-.1,.1)
        opt.step()
        return float(norm)

    def improve(self, batch):
        # Accumulate all requested trajectories before one clipped actor step.
        total = batch['state'].shape[1]
        opt = self.optimizers['actor']
        opt.zero_grad(set_to_none=True)
        chunks, actor_loss = [], 0.
        with frozen(self.world, self.critic):
            for start in range(0, total, self.cfg.imagination_microbatch):
                stop = min(total, start+self.cfg.imagination_microbatch)
                part = {k:v[:,start:stop] for k,v in batch.items()}
                loss, data = self._imagine(part)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite actor loss')
                fraction = (stop-start)/total
                (loss*fraction).backward()
                actor_loss += float(loss.detach())*fraction
                chunks.append(data)
        grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(),1.,error_if_nonfinite=True)
        for p in self.actor.parameters():
            if p.grad is not None:
                p.grad.clamp_(-.1,.1)
        opt.step()
        zs, cs, targets, ws = [torch.cat([c[i] for c in chunks]) for i in range(4)]
        critic_losses = []
        # Bound critic activation memory too, preserving fixed lambda targets.
        size = min(self.cfg.imagination_microbatch*4,
                   max(1, math.ceil(len(zs)/self.cfg.critic_minibatches)))
        for _ in range(self.cfg.critic_epochs):
            for ids in torch.randperm(len(zs),device=zs.device).split(size):
                loss = ((self.value(zs[ids],cs[ids])-targets[ids]).square()*ws[ids]).sum()/ws[ids].sum().clamp_min(1)
                self._update('critic',loss,self.critic.parameters())
                critic_losses.append(float(loss.detach()))
        return dict(actor_loss=actor_loss,critic_loss=sum(critic_losses)/len(critic_losses),
                    actor_grad_norm=float(grad_norm),imagined_rollouts=total)

    def _imagine(self, batch):
        cfg = self.cfg
        context = batch['context'][0]
        z = self.normalize(batch['state'][0]).detach()
        # Time limits stop imagination and bootstrap the pre-reset state. They
        # are NOT physical terminals and never suppress its terminal value.
        remaining = (cfg.episode_steps-self.raw(z)[:,self.sl('steps')].squeeze(-1)).round().clamp_min(1)
        states, rewards, discounts, next_values, time_limits, weights = [], [], [], [], [], []
        survival = torch.ones(len(z), device=z.device)
        objective = torch.zeros_like(survival)
        with frozen(self.world, self.critic):
            for t in range(cfg.horizon):
                active = (remaining > t).to(z.dtype)
                weight = survival*active
                states.append(z.detach())
                weights.append(weight.detach())
                obs = z[:, :self.obs_dim]
                obs = obs+torch.randn_like(obs)*cfg.sensor_noise
                action = self.actor(obs).tanh()
                next_z, r, term, _ = self.step_model(z, action, context)
                d = cfg.gamma*(1-term.sigmoid())
                objective = objective+(cfg.gamma**t)*weight*r
                time_end = (remaining == t+1).to(z.dtype)
                horizon_end = (t == cfg.horizon-1)
                bootstrap = torch.ones_like(time_end) if horizon_end else time_end
                objective = objective+(cfg.gamma**t)*weight*d*bootstrap*self.value(next_z, context)
                rewards.append(r.detach())
                discounts.append(d.detach())
                next_values.append(self.value(next_z, context).detach())
                time_limits.append(time_end)
                survival = survival*(1-term.sigmoid())
                z = next_z
            actor_loss = -objective.mean()

        # GAE / lambda-return targets fixed once for all critic epochs (Eq. 3).
        targets = lambda_returns(rewards, discounts, next_values, time_limits, cfg.lam)
        zs = torch.stack(states).flatten(0,1)
        cs = context.unsqueeze(0).expand(cfg.horizon,-1,-1).flatten(0,1)
        targets = targets.flatten()
        ws = torch.stack(weights).flatten()
        return actor_loss, (zs.detach(),cs.detach(),targets,ws)

    def checkpoint(self):
        return dict(model=self.state_dict(),optimizers={k:v.state_dict() for k,v in self.optimizers.items()},
                    schema=self.schema,cfg=asdict(self.cfg))
