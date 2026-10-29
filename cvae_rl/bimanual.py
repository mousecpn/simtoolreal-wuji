"""Frozen route-F skill interface for the actual dual KUKA/Sharpa task."""
from types import SimpleNamespace
import torch
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul
from isaacsimenvs.tasks.simtoolreal.utils.obs_utils import build_observations, KEYPOINT_CORNERS
from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg


def local_observations(env):
    """Reuse original builder; express each arm in the teacher's environment frame.

    Both physical assets are the same left hand. Rotate arm B, never mirror it.
    Teacher base is (0,.8,0), NOT the observation origin. Restore that offset
    after inverse physical-base transform. Cube scales describe actual 5cm size.
    """
    cfg = SimToolRealEnvCfg()
    cfg.domain_randomization.use_object_state_delay_noise = False
    cfg.domain_randomization.use_obs_delay = False
    cfg.domain_randomization.joint_velocity_obs_noise_std = 0.
    n = env.num_envs
    results = []
    for side, robot in enumerate(env.robots):
        base = robot.data.root_pos_w
        inv = quat_conjugate(robot.data.root_quat_w)
        teacher_base = torch.tensor([0., .8, 0.],device=env.device)
        def pos(x):
            if x.ndim == 3:
                return quat_apply(inv[:,None].expand(-1,x.shape[1],-1).reshape(-1,4),
                                  (x-base[:,None]).reshape(-1,3)).reshape_as(x)+teacher_base
            return quat_apply(inv,x-base)+teacher_base
        def rot(x):
            if x.ndim == 3:
                return quat_mul(inv[:,None].expand(-1,x.shape[1],-1).reshape(-1,4),
                                x.reshape(-1,4)).reshape_as(x)
            return quat_mul(inv,x)
        body = robot.data.body_state_w.clone()
        body[:,:,:3] = pos(body[:,:,:3]); body[:,:,3:7] = rot(body[:,:,3:7])
        for sl in (slice(7,10),slice(10,13)):
            body[:,:,sl] = quat_apply(inv[:,None].expand(-1,body.shape[1],-1).reshape(-1,4),
                                      body[:,:,sl].reshape(-1,3)).reshape(n,-1,3)
        rd = SimpleNamespace(joint_pos=robot.data.joint_pos,joint_vel=robot.data.joint_vel,body_state_w=body)
        def rigid(asset):
            d=asset.data
            return SimpleNamespace(data=SimpleNamespace(root_pos_w=pos(d.root_pos_w),
                root_quat_w=rot(d.root_quat_w),root_lin_vel_w=quat_apply(inv,d.root_lin_vel_w),
                root_ang_vel_w=quat_apply(inv,d.root_ang_vel_w)))
        scales = torch.full((n,3),1.25,device=env.device)
        proxy=SimpleNamespace(cfg=cfg,num_envs=n,device=env.device,
            robot=SimpleNamespace(data=rd), object=rigid(env.object),goal_viz=rigid(env.goal_viz),
            scene=SimpleNamespace(env_origins=torch.zeros_like(base)),
            _perm_lab_to_canon=env.ids[side], _joint_lower_canon=env.lo[side*29:(side+1)*29],
            _joint_upper_canon=env.hi[side*29:(side+1)*29],
            _prev_targets=torch.zeros_like(robot.data.joint_pos),
            _palm_body_id=env.palms[side],_fingertip_body_ids=env.tips[side],
            _object_scale_per_env=scales,_object_scale_multiplier=torch.ones_like(scales),
            _keypoint_offsets=(torch.tensor(KEYPOINT_CORNERS,device=env.device)[None]*.0375).expand(n,-1,-1),
            _closest_keypoint_max_dist=torch.zeros(n,device=env.device),
            _closest_fingertip_dist=torch.zeros(n,5,device=env.device),_lifted_object=env.lifted,
            episode_length_buf=env.episode_length_buf,_successes=env.success.long(),reward_buf=env.last_reward)
        proxy._prev_targets[:,env.ids[side]]=env.prev_targets[:,side*29:(side+1)*29]
        observation=build_observations(proxy)['policy']
        # Semantic checks against actual scene geometry (not only tensor sizes).
        expected_palm=quat_apply(inv,env.palm[:,side,:3]+env.scene.env_origins-base)+teacher_base
        assert torch.allclose(observation[:,87:90],expected_palm.clamp(-10,10),atol=2e-6)
        # Joint representation is unaffected by base pose, including rotated arm B.
        expected_q=2*(robot.data.joint_pos[:,env.ids[side]]-proxy._joint_lower_canon)/(proxy._joint_upper_canon-proxy._joint_lower_canon)-1
        assert torch.allclose(observation[:,:29],expected_q.clamp(-10,10),atol=1e-6)
        results.append(observation)
    result=torch.stack(results,1)
    assert result.shape==(n,2,140) and torch.isfinite(result).all()
    return result
