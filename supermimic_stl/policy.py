"""Standalone deterministic actor loader; does not import Isaac or register tasks."""
from __future__ import annotations
import torch
from torch import nn
from .core import mlp


class Policy(nn.Module):
    def __init__(self,payload):
        super().__init__()
        if payload.get('format')!='supermimic-stl-actor-v1':
            raise ValueError('Expected actor.pt exported by this trainer')
        self.metadata=payload['metadata']
        cfg=self.metadata['core_cfg']
        self.actor=mlp(len(payload['mean']),self.metadata['action_dim'],cfg['policy_hidden'],2)
        self.actor.load_state_dict(payload['actor'])
        self.register_buffer('mean',payload['mean'])
        self.register_buffer('scale',payload['scale'])

    def forward(self,clean_policy_observation):
        if clean_policy_observation.shape[-1]!=len(self.mean):
            raise ValueError('Wrong policy observation dimension')
        return self.actor((clean_policy_observation-self.mean)/self.scale).tanh()


def load_actor(path,device='cpu'):
    return Policy(torch.load(path,map_location=device,weights_only=False)).to(device).eval()
