"""Read-only adapter around existing Isaac Lab tasks; imported after AppLauncher.

Only these new subclasses capture pre-reset states. No monkey-patching, original
module changes or Gym registration. Clean/no-delay dynamics are explicit v1 scope.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import torch
from .core import Config


def configure(args):
    if args.robot == 'tianji-wuji':
        from isaacsimenvs.tasks.tianji_wuji.config import TianjiWujiEnvCfg
        from isaacsimenvs.tasks.tianji_wuji.env import TianjiWujiEnv
        cfg, base = TianjiWujiEnvCfg(), TianjiWujiEnv
        obs_module = importlib.import_module('isaacsimenvs.tasks.tianji_wuji.observations')
    else:
        from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
        from isaacsimenvs.tasks.simtoolreal.simtoolreal_env import SimToolRealEnv
        cfg, base = SimToolRealEnvCfg(), SimToolRealEnv
        obs_module = importlib.import_module('isaacsimenvs.tasks.simtoolreal.utils.obs_utils')
    if getattr(args,'benchmark_single_object',False) and args.robot == 'kuka-sharpa':
        import yaml
        overlay = Path(__file__).resolve().parents[1]/'isaacsimenvs/cfg/task/SimToolReal.yaml'
        cfg.from_dict(yaml.safe_load(overlay.read_text()))
    cfg.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.assets.num_assets_per_type = args.assets_per_type
    cfg.assets.handle_head_types = tuple(args.tool_types)
    cfg.student_obs.enabled = False
    # Explicitly exclude unknown per-step random impulses/latencies; not a
    # silent reduction of the original PPO/SAPG configuration.
    dr = cfg.domain_randomization
    dr.use_obs_delay = dr.use_action_delay = dr.use_object_state_delay_noise = False
    dr.force_scale = dr.torque_scale = dr.joint_velocity_obs_noise_std = 0.
    dr.object_scale_noise_multiplier_range = (1.,1.)
    dr.object_friction_scale_range = dr.fingertip_friction_scale_range = (1.,1.)
    # Hold tolerance fixed within a run so replay reward labels are stationary.
    cfg.termination.eval_success_tolerance = args.success_tolerance
    cfg.termination.max_consecutive_successes = args.max_goals
    cfg.termination.episode_length = args.episode_steps
    cfg.episode_length_s = args.episode_steps*cfg.decimation*cfg.sim.dt
    if getattr(args,'benchmark_single_object',False):
        from .benchmark import apply_benchmark
        apply_benchmark(cfg)
        args.sensor_noise = 0.
    # 3060-safe contact buffer capacities; do not alter global/task defaults.
    for key,value in dict(gpu_max_rigid_contact_count=262144,gpu_max_rigid_patch_count=65536,
                          gpu_found_lost_pairs_capacity=262144,
                          gpu_found_lost_aggregate_pairs_capacity=262144,
                          gpu_total_aggregate_pairs_capacity=262144).items():
        if args.num_envs <= 256:
            setattr(cfg.sim.physx,key,value)

    class CapturedEnv(base):
        def _get_rewards(self):
            reward = super()._get_rewards()
            self.reward_buf = reward
            # Called after _get_dones and before DirectRLEnv automatic reset.
            self.mbrl_terminal_state = snapshot(self,obs_module)
            self.mbrl_success = self._is_success.clone()
            self.mbrl_successes = self._successes.clone()
            return reward

    env = CapturedEnv(cfg)
    env.mbrl_obs_module = obs_module
    sizes = obs_module.OBS_FIELD_SIZES
    order, slices = [], {}
    cursor = 0
    def add(name, dim):
        nonlocal cursor
        order.append(name)
        slices[name] = [cursor,cursor+dim]
        cursor += dim
    for group,fields in [('actor',cfg.obs.obs_list),('critic',cfg.obs.state_list)]:
        for field in fields:
            add(f'{group}.{field}',sizes[field])
        if group=='actor':
            obs_dim = cursor
    for name,dim in [('object',13),('goal',7),('joint_pos',env.robot.num_joints),
                     ('joint_vel',env.robot.num_joints),('targets',env.robot.num_joints),
                     ('object_init_z',1),('table_z',1),('near_goal',1),('successes',1),
                     ('steps',1),('kp_offsets',12)]:
        add(name,dim)
    schema = dict(order=order,slices=slices,obs_dim=obs_dim,state_dim=cursor)
    # Shape identity conditions the model, not the actor. The one-hot precisely
    # identifies each generated asset; bbox alone cannot distinguish geometry.
    paths = env._object_urdf_paths
    fingerprints = [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths]
    index = env._object_asset_index_per_env
    if index.min()<0 or index.max()>=len(paths):
        raise AssertionError('Invalid spawned asset index')
    identity = torch.nn.functional.one_hot(index,len(paths)).float()
    mass = env._object_mass.reshape(env.num_envs,-1)
    context = torch.cat([identity,mass,env._object_scale_per_env],-1)
    core_cfg = Config(imagination_microbatch=getattr(args,"imagination_microbatch",512),horizon=args.horizon,window=args.window,hidden=args.hidden,
                      policy_hidden=args.policy_hidden,lr=args.lr,reward_scale=args.reward_scale,
                      action_noise=args.action_noise,sensor_noise=args.sensor_noise,
                      episode_steps=args.episode_steps,goal_mode=cfg.reset.goal_sampling_type,
                      goal_distance=cfg.reset.delta_goal_distance,goal_degrees=cfg.reset.delta_rotation_degrees,
                      goal_mins=tuple(cfg.reset.target_volume_mins),goal_maxs=tuple(cfg.reset.target_volume_maxs),
                      goal_region_scale=cfg.reset.target_volume_region_scale)
    metadata = dict(robot=args.robot,asset_fingerprints=fingerprints,obs_fields=list(cfg.obs.obs_list),
                    action_dim=cfg.action_space,context_dim=context.shape[-1],schema=schema,
                    domain_randomization=('single fixed hammer; no DR; fixed reset' if getattr(args,'benchmark_single_object',False)
                                          else 'clean/no-delay v1; physical reset randomization retained'),
                    success_tolerance=args.success_tolerance,max_goals=args.max_goals,
                    assets_per_type=args.assets_per_type,tool_types=args.tool_types,
                    seed=args.seed,core_cfg=core_cfg.__dict__)
    return env,schema,context,core_cfg,metadata


def snapshot(env, obs_module=None):
    module = obs_module or env.mbrl_obs_module
    observations = module.build_observations(env)
    obj = env.object.data.root_state_w.clone()
    obj[:,:3] -= env.scene.env_origins
    goal = torch.cat([env.goal_viz.data.root_pos_w-env.scene.env_origins,
                      env.goal_viz.data.root_quat_w],-1)
    n = env.num_envs
    fields = [observations['policy'],observations['critic'],obj,goal,
              env.robot.data.joint_pos,env.robot.data.joint_vel,env._prev_targets,
              env._object_init_z.reshape(n,1),env._table_z_per_env.reshape(n,1),
              env._near_goal_steps.float().reshape(n,1),env._successes.float().reshape(n,1),
              env.episode_length_buf.float().reshape(n,1),env._keypoint_offsets.reshape(n,12)]
    return torch.cat(fields,-1).clone()
