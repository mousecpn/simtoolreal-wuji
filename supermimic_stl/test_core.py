"""Meaningful numerical contracts, runnable without starting Isaac Sim."""
import copy
import unittest
import tempfile
from pathlib import Path
import torch
from .core import Config,Learner,Replay,frozen,sample_goal,lambda_returns


def fixture():
    sizes = [('actor.palm_pos',3),('actor.object_rot',4),('actor.object_scales',3),
             ('actor.keypoints_rel_palm',12),('actor.keypoints_rel_goal',12),
             ('critic.palm_pos',3),('critic.object_rot',4),('critic.object_scales',3),
             ('critic.keypoints_rel_palm',12),('critic.keypoints_rel_goal',12),
             ('object',13),('goal',7),('near_goal',1),('successes',1),('steps',1),
             ('kp_offsets',12),('object_init_z',1),('table_z',1)]
    schema = dict(order=[],slices={},obs_dim=34)
    cursor=0
    for key,dim in sizes:
        schema['order'].append(key)
        schema['slices'][key]=[cursor,cursor+dim]
        cursor+=dim
    schema['state_dim']=cursor
    cfg=Config(hidden=32,policy_hidden=16,horizon=4,window=4,critic_epochs=1,critic_minibatches=2)
    learner=Learner(schema,3,2,cfg)
    state=torch.randn(6,cursor)*.1
    state[:,learner.sl('object')][:,3]=1.
    state[:,learner.sl('goal')][:,3]=1.
    state[:,learner.sl('steps')]=0
    return learner,state


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(1)

    def test_ring_wrap_windows_never_join_nonadjacent_times(self):
        replay=Replay(5,2,1,1,1,'cpu')
        for t in range(13):
            replay.add(**{k:torch.full((2,1),float(t)) for k in replay.data})
        sample=replay.sample(200,4)['state'][:,:,0]
        self.assertTrue(torch.equal(sample[1:]-sample[:-1],torch.ones_like(sample[1:])))
        self.assertGreaterEqual(float(sample.min()),8)
        restored=Replay(5,2,1,1,1,'cpu')
        restored.load_state_dict(replay.state_dict())
        for k in replay.data:
            self.assertTrue(torch.equal(replay.data[k],restored.data[k]))

    def test_goal_projection_preserves_goal_without_success_and_is_differentiable(self):
        learner,s=fixture()
        predicted=s.clone().requires_grad_()
        proposal=s[:,learner.sl('goal')].clone()
        proposal[:,:3]+=1
        projected=learner.project(predicted,s,torch.zeros(len(s)),proposal)
        self.assertTrue(torch.allclose(projected[:,learner.sl('goal')][:,:3],s[:,learner.sl('goal')][:,:3]))
        switched=learner.project(predicted,s,torch.ones(len(s)),proposal)
        self.assertTrue(torch.allclose(switched[:,learner.sl('goal')][:,:3],proposal[:,:3]))
        switched[:,learner.sl('actor.keypoints_rel_goal')].sum().backward()
        self.assertGreater(float(predicted.grad[:,learner.sl('object')].abs().sum()),0)
        self.assertTrue(torch.isfinite(predicted.grad).all())

    def test_actor_bptt_changes_actor_but_not_world_weights(self):
        learner,s=fixture()
        batch=dict(state=s[None],context=torch.randn(1,len(s),3))
        before_world=[p.clone() for p in learner.world.parameters()]
        before_actor=[p.clone() for p in learner.actor.parameters()]
        result=learner.improve(batch)
        self.assertGreater(result['actor_grad_norm'],0.)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before_world,learner.world.parameters())))
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before_actor,learner.actor.parameters())))
        self.assertTrue(all(p.requires_grad for p in learner.world.parameters()))

    def test_world_autoregression_masks_labels_after_reset(self):
        learner,s=fixture()
        batch={
            'state':s[None].repeat(4,1,1),
            'next_state':(s+.01)[None].repeat(4,1,1),
            'action':torch.zeros(4,len(s),2),
            'context':torch.zeros(4,len(s),3),
            'reward':torch.zeros(4,len(s),1),
            'terminated':torch.ones(4,len(s),1),
            'truncated':torch.zeros(4,len(s),1),
            'switch':torch.zeros(4,len(s),1),
        }
        other=copy.deepcopy(learner)
        changed={k:v.clone() for k,v in batch.items()}
        changed['reward'][1:]=10000
        changed['next_state'][1:]+=100
        first=learner.fit_world(batch)
        second=other.fit_world(changed)
        self.assertAlmostEqual(first['world_loss'],second['world_loss'],places=5)
        for a,b in zip(learner.world.parameters(),other.world.parameters()):
            self.assertTrue(torch.allclose(a,b,atol=1e-7))

    def test_goal_random_walk_and_quaternion_norm(self):
        learner,s=fixture()
        goal=s[:,learner.sl('goal')].clone()
        goal[:,:3]=torch.tensor([0.,0.,.8])
        sampled=sample_goal(goal,learner.cfg)
        self.assertTrue(torch.all((sampled[:,:3]-goal[:,:3]).abs()<=.100001))
        self.assertTrue(torch.allclose(sampled[:,3:].norm(dim=-1),torch.ones(len(s)),atol=1e-6))

    def test_timeouts_bootstrap_but_physical_terminals_do_not(self):
        rewards=torch.tensor([[1.,1.,1.],[100.,100.,100.]])
        # First sample times out, second physically terminates, third continues.
        discounts=torch.tensor([[.9,0.,.9],[.9,.9,.9]])
        values=torch.tensor([[10.,10.,10.],[20.,20.,20.]])
        time_limits=torch.tensor([[1.,0.,0.],[0.,0.,0.]])
        result=lambda_returns(rewards,discounts,values,time_limits,1.)
        self.assertTrue(torch.allclose(result[0],torch.tensor([10.,1.,107.2])))

    def test_exported_actor_matches_training_actor(self):
        from .policy import load_actor
        learner,state=fixture()
        metadata=dict(action_dim=2,core_cfg=learner.cfg.__dict__)
        payload=dict(format='supermimic-stl-actor-v1',actor=learner.actor.state_dict(),
                     mean=learner.mean[:learner.obs_dim],scale=learner.scale[:learner.obs_dim],metadata=metadata)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'actor.pt'
            torch.save(payload,path)
            actor=load_actor(path)
            self.assertTrue(torch.allclose(actor(state[:,:learner.obs_dim]),learner.act(state)))

    def test_freezing_preserves_input_gradients(self):
        layer=torch.nn.Linear(3,1)
        x=torch.ones(2,3,requires_grad=True)
        with frozen(layer):
            layer(x).sum().backward()
        self.assertGreater(float(x.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in layer.parameters()))
        self.assertTrue(all(p.requires_grad for p in layer.parameters()))


if __name__=='__main__':
    unittest.main()

class ScalingTests(unittest.TestCase):
    def test_microbatch_accumulates_all_rollouts_with_one_actor_step(self):
        torch.manual_seed(11)
        torch.set_num_threads(1)
        learner,s=fixture()
        learner.cfg.imagination_microbatch=2
        result=learner.improve(dict(state=s[None],context=torch.zeros(1,len(s),3)))
        self.assertEqual(result['imagined_rollouts'],6)
        steps=[int(v['step']) for v in learner.optimizers['actor'].state.values()]
        self.assertTrue(steps and all(x==1 for x in steps))
        self.assertTrue(all(torch.isfinite(p).all() for p in learner.parameters()))
