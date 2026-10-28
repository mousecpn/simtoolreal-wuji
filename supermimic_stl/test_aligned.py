"""Recurrent boundaries, differentiable normalization, and native equivalence."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import torch
from .core import Config,Learner,Replay
from .aligned import NativeNetwork,make_model,reset_hidden
from .test_core import fixture


def small_spec():
    actor=dict(name='actor_critic',separate=False,
        space={'continuous':dict(mu_activation='None',sigma_activation='None',mu_init={'name':'default'},
              sigma_init={'name':'const_initializer','val':0},fixed_sigma='coef_cond')},
        mlp=dict(units=[16,16,8,8],activation='elu',d2rl=False,initializer={'name':'default'},regularizer={'name':'None'}),
        rnn=dict(name='lstm',units=8,layers=1,before_mlp=True,layer_norm=True))
    critic=dict(name='actor_critic',central_value=True,mlp=deepcopy(actor['mlp']))
    return dict(actor=actor,critic=critic,model={'name':'continuous_a2c_logstd'},learning_rate=1e-4,
                critic_epochs=2,critic_minibatch=16,bounds_loss_coef=.0001,seq_length=16)


def aligned_fixture():
    old,state=fixture()
    cfg=replace(old.cfg,policy_arch='sapg',native_spec=small_spec(),rnn_burnin=16,imagination_microbatch=2)
    return Learner(old.schema,3,2,cfg),state


class AlignedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(17)

    def test_native_actor_outputs_and_parameter_shapes_match(self):
        spec=small_spec(); actor=NativeNetwork(spec,5,2)
        native=make_model(spec,5,2).eval()
        native.load_state_dict(actor.model.state_dict())
        self.assertEqual({k:tuple(v.shape) for k,v in native.state_dict().items()},
                         {k:tuple(v.shape) for k,v in actor.model.state_dict().items()})
        x=torch.randn(6,5); ids=torch.linspace(50.,0.,6)[:,None]
        hidden=actor.initial_state(6)
        with torch.no_grad():
            action,new,mu=actor.step(x,hidden,ids)
            result=native(dict(is_train=False,prev_actions=None,obs=torch.cat([x,ids],-1),rnn_states=hidden))
        self.assertTrue(torch.allclose(mu,result['mus'],atol=1e-6))
        self.assertTrue(torch.allclose(action,result['mus'].clamp(-1,1),atol=1e-6))
        self.assertTrue(all(torch.allclose(a,b) for a,b in zip(new,result['rnn_states'])))

    def test_differentiable_normalization_preserves_input_gradient(self):
        actor=NativeNetwork(small_spec(),5,2)
        x=torch.randn(6,5,requires_grad=True)
        count=actor.model.running_mean_std.count.clone()
        actor.step(x)[2].sum().backward()
        self.assertGreater(float(x.grad.abs().sum()),0)
        self.assertTrue(torch.equal(count,actor.model.running_mean_std.count))

    def test_recurrent_export_carries_memory_and_resets_per_episode(self):
        from .policy import load_actor
        learner,state=aligned_fixture()
        obs=state[:,:learner.obs_dim]
        metadata=dict(action_dim=2,core_cfg=learner.cfg.__dict__)
        payload=dict(format='supermimic-stl-actor-v2',actor=learner.actor.state_dict(),
            mean=learner.mean[:learner.obs_dim],scale=learner.scale[:learner.obs_dim],metadata=metadata)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'actor.pt';torch.save(payload,path);actor=load_actor(path)
            first=actor(obs); second=actor(obs)
            self.assertFalse(torch.allclose(first,second))
            actor.reset_done(torch.tensor([True,False,True,False,True,False]))
            third=actor(obs)
            self.assertTrue(torch.allclose(first[::2],third[::2],atol=1e-6))
            self.assertFalse(torch.allclose(first[1::2],third[1::2]))

    def test_recurrent_replay_restores_collected_hidden_and_contiguous_prefix(self):
        replay=Replay(64,6,1,1,1,'cpu',rnn_size=8)
        for t in range(96):
            transition={k:torch.full((6,1),float(t)) for k in replay.data}
            hidden=tuple(torch.full((1,6,8),float(t)) for _ in range(2))
            replay.add(rnn_state=hidden,**transition)
        batch=replay.sample_policy(100,16)
        self.assertTrue(torch.equal(batch['rnn_state'][0,:,0],batch['state'][0,:,0]))
        self.assertTrue(torch.equal(batch['state'][1:]-batch['state'][:-1],torch.ones_like(batch['state'][1:])))
        self.assertGreaterEqual(float(batch['state'].min()),32.)
        restored=Replay(64,6,1,1,1,'cpu',rnn_size=8)
        restored.load_state_dict(replay.state_dict())
        self.assertTrue(torch.equal(restored.rnn_data,replay.rnn_data))

    def test_native_critic_uses_only_original_privileged_observations(self):
        learner,state=aligned_fixture();state.requires_grad_()
        value=learner.value(learner.normalize(state),torch.randn(6,3))
        value.sum().backward()
        self.assertEqual(float(state.grad[:,:learner.obs_dim].abs().sum()),0.)
        self.assertEqual(float(state.grad[:,learner.critic_end:].abs().sum()),0.)
        self.assertGreater(float(state.grad[:,learner.obs_dim:learner.critic_end].abs().sum()),0.)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_imagination_accepts_replayed_noncontiguous_hidden(self):
        learner,state=aligned_fixture();learner=learner.cuda()
        replay=Replay(64,6,state.shape[-1],2,3,'cpu',sample_device='cuda:0',rnn_size=8)
        for t in range(40):
            replay.add(state=state,action=torch.zeros(6,2),context=torch.zeros(6,3),next_state=state,
                reward=torch.zeros(6),terminated=torch.zeros(6),truncated=torch.zeros(6),switch=torch.zeros(6),
                condition=torch.linspace(50.,0.,6)[:,None],rnn_state=tuple(torch.zeros(1,6,8) for _ in range(2)))
        result=learner.improve(replay.sample_policy(6,16))
        self.assertEqual(result['imagined_rollouts'],6)
        self.assertGreater(result['actor_grad_norm'],0.)

    def test_burnin_resets_hidden_at_physical_episode_boundary(self):
        learner,state=aligned_fixture()
        batch=dict(state=state[None].repeat(17,1,1),context=torch.zeros(17,6,3),
            condition=torch.zeros(17,6,1),terminated=torch.zeros(17,6,1),
            truncated=torch.zeros(17,6,1),rnn_state=torch.randn(2,6,8))
        batch['truncated'][15]=1.
        other={k:v.clone() for k,v in batch.items()};other['rnn_state']+=100.
        torch.manual_seed(99); a,data_a=learner._imagine(batch)
        torch.manual_seed(99); b,data_b=learner._imagine(other)
        self.assertTrue(torch.allclose(a,b,atol=1e-7))
        self.assertTrue(all(torch.allclose(x,y,atol=1e-7) for x,y in zip(data_a,data_b)))

    def test_recurrent_bptt_updates_actor_sigma_embedding_but_freezes_world(self):
        learner,state=aligned_fixture()
        replay=Replay(64,6,state.shape[-1],2,3,'cpu',rnn_size=8)
        conditions=torch.linspace(50.,0.,6)[:,None]
        for t in range(40):
            replay.add(state=state,action=torch.zeros(6,2),context=torch.zeros(6,3),next_state=state,
                reward=torch.zeros(6),terminated=torch.zeros(6),truncated=torch.zeros(6),switch=torch.zeros(6),
                condition=conditions,rnn_state=tuple(torch.zeros(1,6,8) for _ in range(2)))
        world=[p.clone() for p in learner.world.parameters()]
        sigma=learner.actor.model.a2c_network.sigma.clone()
        embedding=learner.actor.model.a2c_network.extra_params.clone()
        result=learner.improve(replay.sample_policy(6,16))
        self.assertEqual(result['imagined_rollouts'],6)
        self.assertEqual(result['critic_optimizer_steps'],4)
        self.assertTrue(all(int(v['step'])==4 for v in learner.optimizers['critic'].state.values()))
        self.assertGreater(result['actor_grad_norm'],0.)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(world,learner.world.parameters())))
        self.assertFalse(torch.equal(sigma,learner.actor.model.a2c_network.sigma))
        self.assertFalse(torch.equal(embedding,learner.actor.model.a2c_network.extra_params))
        self.assertTrue(all(int(v['step'])==1 for v in learner.optimizers['actor'].state.values()))

if __name__=='__main__': unittest.main()
