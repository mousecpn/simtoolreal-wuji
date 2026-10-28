"""Native SAPG network topology with differentiable, frozen-stat normalization.

rl_games Model.norm_obs uses no_grad, which would sever model-based policy
BPTT. We reuse its network builders/parameters and RunningMeanStd buffers,
but evaluate normalization algebra with gradients and detached statistics.
"""
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn


def native_spec():
    import yaml
    path=Path(__file__).resolve().parents[1]/'isaacsimenvs/cfg/train/SimToolRealSAPG.yaml'
    p=yaml.safe_load(path.read_text())['params']
    return dict(actor=deepcopy(p['network']),critic=deepcopy(p['config']['central_value_config']['network']),
                model=deepcopy(p['model']),clip_observations=float(p['env']['clip_observations']),learning_rate=float(p['config']['learning_rate']),
                critic_epochs=p['config']['central_value_config']['mini_epochs'],
                critic_minibatch=p['config']['central_value_config']['minibatch_size'],
                bounds_loss_coef=p['config']['bounds_loss_coef'],seq_length=p['config']['seq_length'])


def norm(x,stats):
    return ((x-stats.running_mean.detach().float())/
            (stats.running_var.detach().float()+stats.epsilon).sqrt()).clamp(-5.,5.)


def denorm(x,stats):
    return x.clamp(-5.,5.)*(stats.running_var.detach().float()+stats.epsilon).sqrt()+stats.running_mean.detach().float()


@torch.no_grad()
def update_stats(stats,x):
    if len(x)<2: return
    stats.train()
    stats(x.detach())
    stats.eval()


def make_model(spec,obs_dim,action_dim,critic=False):
    from rl_games.algos_torch.model_builder import ModelBuilder
    params=dict(model={'name':'central_value'} if critic else spec['model'],
                network=deepcopy(spec['critic' if critic else 'actor']))
    return ModelBuilder().load(params).build(dict(actions_num=action_dim,input_shape=(obs_dim+1,),
        num_seqs=1,value_size=1,normalize_input=True,normalize_value=True,type='extra_param',
        coef_ids=torch.linspace(50.,0.,6),coef_id_idx=obs_dim))


class NativeNetwork(nn.Module):
    def __init__(self,spec,obs_dim,action_dim,critic=False):
        super().__init__()
        self.obs_dim,self.critic=obs_dim,critic
        self.clip_obs=spec.get("clip_observations",10.)
        self.model=make_model(spec,obs_dim,action_dim,critic)
        self.model.running_mean_std.eval()
        self.model.value_mean_std.eval()

    def _apply(self,fn):
        result=super()._apply(fn)
        # Native builder's condition IDs are plain tensors, not buffers.
        net=self.model.a2c_network
        for name in ('param_ids','sigma_ids'):
            if hasattr(net,name): setattr(net,name,fn(getattr(net,name)))
        return result

    def initial_state(self,batch):
        net=self.model.a2c_network
        template=next(net.parameters())
        return tuple(s.to(template).expand(-1,batch,-1).contiguous()
                     for s in net.get_default_rnn_state())

    def inputs(self,obs,condition=None,noise=0.):
        x=norm(obs.clamp(-self.clip_obs,self.clip_obs),self.model.running_mean_std)
        if noise: x=x+torch.randn_like(x)*noise
        if condition is None: condition=x.new_zeros(len(x),1)
        return torch.cat([x,condition.reshape(-1,1)],-1)

    def step(self,obs,hidden=None,condition=None,stochastic=False,noise=0.):
        if hidden is None: hidden=self.initial_state(len(obs))
        hidden=tuple(x.contiguous() for x in hidden)
        mu,logstd,_,hidden=self.model.a2c_network(dict(obs=self.inputs(obs,condition,noise),
                                                     rnn_states=hidden,seq_length=1))
        action=mu+logstd.exp()*torch.randn_like(mu) if stochastic else mu
        return action.clamp(-1.,1.),hidden,mu

    def forward(self,obs,condition=None,normalized=False):
        value,_=self.model.a2c_network(dict(obs=self.inputs(obs,condition)))
        return value.squeeze(-1) if normalized else denorm(value,self.model.value_mean_std).squeeze(-1)

    def normalize_targets(self,target):
        return norm(target[:,None],self.model.value_mean_std).squeeze(-1)

    def update_inputs(self,obs): update_stats(self.model.running_mean_std,obs.clamp(-self.clip_obs,self.clip_obs))
    def update_values(self,value): update_stats(self.model.value_mean_std,value[:,None])


def reset_hidden(hidden,done):
    return tuple(s*(~done.bool()).to(s.dtype)[None,:,None] for s in hidden)
