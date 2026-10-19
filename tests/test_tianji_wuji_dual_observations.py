"""CPU checks for both-hand observation geometry, independent of Isaac Sim."""
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
import unittest
from unittest.mock import patch


def load_observations():
    # Unit-test identity-pose assembly without starting an Omniverse app.
    math = ModuleType("isaaclab.utils.math")
    math.convert_quat = lambda q, to: torch.roll(q, -1, dims=-1)
    def apply(q, v):
        assert torch.allclose(q[..., 0], torch.ones_like(q[..., 0]))
        assert torch.allclose(q[..., 1:], torch.zeros_like(q[..., 1:]))
        return v
    math.quat_apply = apply
    math.quat_from_angle_axis = lambda *a: None
    math.quat_mul = lambda *a: None
    path = Path(__file__).parents[1] / 'isaacsimenvs/tasks/tianji_wuji_dual/observations.py'
    spec = importlib.util.spec_from_file_location('_dual_obs_test', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(__import__('sys').modules, {'isaaclab.utils.math': math}):
        spec.loader.exec_module(module)
    return module


def mock_env(module, delay=False):
    n = 2
    state = torch.zeros(n, 12, 13)
    state[..., 3] = 1.
    origins = torch.tensor([[0., 0., 0.], [2., 0., 0.]])
    state[..., :3] += origins.unsqueeze(1)
    state[:, 0, 1] = .3
    state[:, 1, 1] = -.3
    state[:, 2:7, 1] = .35
    state[:, 7:12, 1] = -.35
    palm_fields = ('palm_pos', 'palm_rot', 'fingertip_pos_rel_palm', 'keypoints_rel_palm')
    obs_cfg = SimpleNamespace(obs_list=palm_fields, state_list=tuple(module.OBS_FIELD_SIZES), clamp_abs_observations=100.)
    dr = SimpleNamespace(use_object_state_delay_noise=False, joint_velocity_obs_noise_std=0., use_obs_delay=delay)
    object_data = SimpleNamespace(root_pos_w=origins.clone(), root_quat_w=torch.tensor([[1.,0.,0.,0.]]).repeat(n,1), root_lin_vel_w=torch.zeros(n,3), root_ang_vel_w=torch.zeros(n,3))
    object2_data = SimpleNamespace(root_pos_w=origins + torch.tensor([0., -.5, 0.]),
        root_quat_w=object_data.root_quat_w.clone(), root_lin_vel_w=torch.zeros(n,3), root_ang_vel_w=torch.zeros(n,3))
    return SimpleNamespace(
        objects=(SimpleNamespace(data=object_data), SimpleNamespace(data=object2_data)),
        goals=(SimpleNamespace(data=object_data), SimpleNamespace(data=object2_data)),
        num_envs=n, device='cpu', scene=SimpleNamespace(env_origins=origins),
        cfg=SimpleNamespace(obs=obs_cfg, domain_randomization=dr),
        robot=SimpleNamespace(data=SimpleNamespace(body_state_w=state, joint_pos=torch.zeros(n,54), joint_vel=torch.zeros(n,54))),
        object=SimpleNamespace(data=object_data), goal_viz=SimpleNamespace(data=object_data),
        _palm_body_ids=[0,1], _fingertip_body_ids=list(range(2,12)),
        _perm_lab_to_canon=torch.arange(54), _joint_lower_canon=-torch.ones(54), _joint_upper_canon=torch.ones(54),
        _prev_targets=torch.zeros(n,54), _keypoint_offsets=torch.zeros(n,2,4,3),
        _object_scale_multiplier=torch.ones(n,2,3), _tool_scales=torch.ones(n,2,3),
        _closest_keypoint_max_dist=torch.zeros(n,2), _closest_fingertip_dist=torch.zeros(n,10),
        _lifted_object=torch.zeros(n,2,dtype=torch.bool), episode_length_buf=torch.zeros(n), _successes=torch.zeros(n), reward_buf=torch.zeros(n),
        _obs_queue=torch.zeros(n,3,module.compute_obs_dim(palm_fields)),
    )


class DualObservationTests(unittest.TestCase):
    def test_each_finger_and_keypoint_uses_its_own_palm(self):
        module = load_observations()
        env = mock_env(module)
        obs = module.build_observations(env)
        assert obs['critic'].shape == (2, module.compute_obs_dim(env.cfg.obs.state_list))
        assert obs['policy'].shape == (2, 68)
        # Packed fields: two palm positions, two rotations, ten relative tips,
        # followed by four object keypoints relative to each palm.
        tips = obs['policy'][:, 14:44].reshape(2,10,3)
        assert torch.allclose(tips[:, :5, 1], torch.full((2,5), .05))
        assert torch.allclose(tips[:, 5:, 1], torch.full((2,5), -.05))
        keypoints = obs['policy'][:,44:].reshape(2,2,4,3)
        assert torch.allclose(keypoints[:,0,:,1], torch.full((2,4), -.3))
        assert torch.allclose(keypoints[:,1,:,1], torch.full((2,4), -.2))
        # Environment-origin subtraction must make the two clones identical.
        assert torch.allclose(obs['policy'][0], obs['policy'][1])


    def test_second_tool_changes_only_its_assigned_geometry(self):
        module = load_observations()
        env = mock_env(module)
        before = module.build_observations(env)['policy'][:,44:].reshape(2,2,4,3)
        env.objects[1].data.root_pos_w[:,0] += .12
        after = module.build_observations(env)['policy'][:,44:].reshape(2,2,4,3)
        assert torch.allclose(after[:,0], before[:,0])
        assert torch.allclose(after[:,1,:,0] - before[:,1,:,0], torch.full((2,4), .12))


    def test_success_requires_both_tools_and_reach_uses_assigned_hand(self):
        module = load_observations()
        env = mock_env(module)
        env.cfg.reward = SimpleNamespace(fixed_size_keypoint_reward=False, keypoint_scale=1.)
        env.cfg.termination = SimpleNamespace(force_consecutive_near_goal_steps=True, success_steps=2)
        env._current_success_tolerance = .01
        env._near_goal_steps = torch.zeros(2,2,dtype=torch.long)
        env._closest_keypoint_max_dist.fill_(-1.)
        # Use separate goal states; move just the second one outside tolerance.
        goal = SimpleNamespace(root_pos_w=env.objects[1].data.root_pos_w.clone() + torch.tensor([.2,0.,0.]),
            root_quat_w=env.objects[1].data.root_quat_w.clone())
        env.goals = (env.goals[0], SimpleNamespace(data=goal))
        reward_math = ModuleType('isaacsimenvs.tasks.simtoolreal.utils.reward_utils')
        reward_math.update_near_goal_steps = lambda near_goal, near_goal_steps, force_consecutive: (near_goal_steps + near_goal.long()) * near_goal.long()
        module.__package__ = 'isaacsimenvs.tasks.tianji_wuji_dual'
        with patch.dict(__import__('sys').modules, {reward_math.__name__: reward_math}):
            module.compute_intermediate_values(env)
            module.compute_intermediate_values(env)
            assert env._tool_success[:,0].all()
            assert not env._tool_success[:,1].any()
            assert not env._is_success.any()
            # Fingertips are .35 m from tool0, .15 m from tool1.
            assert torch.allclose(env._curr_fingertip_distances[:,:5], torch.full((2,5), .35))
            assert torch.allclose(env._curr_fingertip_distances[:,5:], torch.full((2,5), .15))
            goal.root_pos_w.copy_(env.objects[1].data.root_pos_w)
            module.compute_intermediate_values(env)
            module.compute_intermediate_values(env)
            assert env._is_success.all()


    def test_only_the_lifted_tool_receives_its_one_shot_bonus(self):
        root = Path(__file__).parents[1]
        helper_name = 'isaacsimenvs.tasks.simtoolreal.utils.reward_utils'
        spec = importlib.util.spec_from_file_location(helper_name, root / 'isaacsimenvs/tasks/simtoolreal/utils/reward_utils.py')
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        name = 'isaacsimenvs.tasks.tianji_wuji_dual.rewards'
        spec = importlib.util.spec_from_file_location(name, root / 'isaacsimenvs/tasks/tianji_wuji_dual/rewards.py')
        rewards = importlib.util.module_from_spec(spec)
        with patch.dict(__import__('sys').modules, {helper_name: helper}):
            spec.loader.exec_module(rewards)
        env = mock_env(load_observations())
        env.cfg.reward = SimpleNamespace(lifting_bonus_threshold=.1, lifting_bonus=7., lifting_rew_scale=0.,
            distance_delta_rew_scale=0., keypoint_rew_scale=0., kuka_actions_penalty_scale=0., hand_actions_penalty_scale=0., reach_goal_bonus=0.)
        env.cfg.termination = SimpleNamespace(success_steps=2, force_consecutive_near_goal_steps=True)
        env._object_init_z = torch.zeros(2,2)
        env.objects[1].data.root_pos_w[:,2] = .2
        env._curr_fingertip_distances = torch.zeros(2,10)
        env._keypoints_max_dist = torch.zeros(2,2)
        env._near_goal = torch.zeros(2,2,dtype=torch.bool)
        env._is_success = torch.zeros(2,dtype=torch.bool)
        env._arm_joint_ids = list(range(14))
        env._hand_joint_ids = list(range(14,54))
        reward = rewards.compute_rewards(env)
        assert reward.shape == (2,)
        assert torch.allclose(reward, torch.full((2,),7.))
        assert not env._lifted_object[:,0].any()
        assert env._lifted_object[:,1].all()
        assert torch.allclose(rewards.compute_rewards(env), torch.zeros(2))


    def test_observation_delay_flush_retains_both_hands(self):
        module = load_observations()
        env = mock_env(module, delay=True)
        obs = module.build_observations(env)
        assert torch.isfinite(obs['policy']).all()
        assert torch.allclose(env._obs_queue[:,0], obs['policy'])
        assert torch.allclose(env._obs_queue[:,1], obs['policy'])
        assert torch.allclose(env._obs_queue[:,2], obs['policy'])


if __name__ == "__main__":
    unittest.main()
