"""Standalone deterministic actor loader; does not import Isaac or register tasks."""
from __future__ import annotations
import torch
from torch import nn
from .core import mlp


class Policy(nn.Module):
    def __init__(self,payload):
        super().__init__()
        if payload.get('format') not in ('supermimic-stl-actor-v1','supermimic-stl-actor-v2'):
            raise ValueError('Expected actor.pt exported by this trainer')
        self.metadata=payload['metadata']
        cfg=self.metadata['core_cfg']
        self.native=cfg.get('policy_arch','legacy')=='sapg'
        self.hidden=None
        if self.native:
            from .aligned import NativeNetwork
            self.actor=NativeNetwork(cfg['native_spec'],len(payload['mean']),self.metadata['action_dim'])
        else:
            self.actor=mlp(len(payload['mean']),self.metadata['action_dim'],cfg['policy_hidden'],2)
        self.actor.load_state_dict(payload['actor'])
        self.register_buffer('mean',payload['mean'])
        self.register_buffer('scale',payload['scale'])

    def forward(self,clean_policy_observation):
        if clean_policy_observation.shape[-1]!=len(self.mean):
            raise ValueError('Wrong policy observation dimension')
        if self.native:
            action,self.hidden,_=self.actor.step(clean_policy_observation,self.hidden)
            self.hidden=tuple(x.detach() for x in self.hidden)
            return action
        return self.actor((clean_policy_observation-self.mean)/self.scale).tanh()

    def reset_done(self,done):
        if self.native and self.hidden is not None:
            from .aligned import reset_hidden
            self.hidden=reset_hidden(self.hidden,done)

    def reset(self): self.hidden=None


def load_actor(path,device='cpu'):
    return Policy(torch.load(path,map_location=device,weights_only=False)).to(device).eval()
