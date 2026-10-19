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
    return SimpleNamespace(
        num_envs=n, device='cpu', scene=SimpleNamespace(env_origins=origins),
        cfg=SimpleNamespace(obs=obs_cfg, domain_randomization=dr),
        robot=SimpleNamespace(data=SimpleNamespace(body_state_w=state, joint_pos=torch.zeros(n,54), joint_vel=torch.zeros(n,54))),
        object=SimpleNamespace(data=object_data), goal_viz=SimpleNamespace(data=object_data),
        _palm_body_ids=[0,1], _fingertip_body_ids=list(range(2,12)),
        _perm_lab_to_canon=torch.arange(54), _joint_lower_canon=-torch.ones(54), _joint_upper_canon=torch.ones(54),
        _prev_targets=torch.zeros(n,54), _keypoint_offsets=torch.zeros(n,4,3),
        _object_scale_multiplier=torch.ones(n,3), _object_scale_per_env=torch.ones(n,3),
        _closest_keypoint_max_dist=torch.zeros(n), _closest_fingertip_dist=torch.zeros(n,10),
        _lifted_object=torch.zeros(n,dtype=torch.bool), episode_length_buf=torch.zeros(n), _successes=torch.zeros(n), reward_buf=torch.zeros(n),
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
        assert torch.allclose(keypoints[:,1,:,1], torch.full((2,4), .3))
        # Environment-origin subtraction must make the two clones identical.
        assert torch.allclose(obs['policy'][0], obs['policy'][1])


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
