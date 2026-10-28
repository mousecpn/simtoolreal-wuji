"""Single, immutable hammer benchmark settings shared by both algorithms."""
from pathlib import Path
import argparse
import json
import os
import sys

ASSET = Path(__file__).resolve().parent/'assets'/'fixed_hammer.urdf'
SETTINGS = {
    'assets.object_urdf': str(ASSET),
    'assets.object_scale': [3.5,.625,.625],
    'assets.num_assets_per_type': 1,
    'assets.handle_head_types': ['hammer'],
    'student_obs.enabled': False,
    'student_obs.use_camera_pose_rand': False,
    'student_obs.use_depth_aug': False,
    'student_obs.use_camera_delay': False,
    'student_obs.use_student_obs_delay': False,
    'action.arm_control_mode': 'joint',
    'domain_randomization.use_obs_delay': False,
    'domain_randomization.obs_delay_max': 0,
    'domain_randomization.use_action_delay': False,
    'domain_randomization.action_delay_max': 0,
    'domain_randomization.use_object_state_delay_noise': False,
    'domain_randomization.object_state_delay_max': 0,
    'domain_randomization.object_state_xyz_noise_std': 0.,
    'domain_randomization.object_state_rotation_noise_degrees': 0.,
    'domain_randomization.object_scale_noise_multiplier_range': [1.,1.],
    'domain_randomization.joint_velocity_obs_noise_std': 0.,
    'domain_randomization.force_scale': 0.,
    'domain_randomization.torque_scale': 0.,
    'domain_randomization.object_friction_scale_range': [1.,1.],
    'domain_randomization.fingertip_friction_scale_range': [1.,1.],
    'reset.reset_position_noise_x': 0.,
    'reset.reset_position_noise_y': 0.,
    'reset.reset_position_noise_z': 0.,
    'reset.fixed_start_pose': [0.,0.,.63,1.,0.,0.,0.],
    'reset.reset_dof_pos_random_interval_arm': 0.,
    'reset.reset_dof_pos_random_interval_fingers': 0.,
    'reset.reset_dof_vel_random_interval': 0.,
    'reset.table_reset_z_range': 0.,
    'reset.table_reset_xy_range_m': [0.,0.],
    'reset.table_reset_yaw_range_deg': 0.,
    'termination.eval_success_tolerance': .075,
    'termination.max_consecutive_successes': 50,
    'termination.episode_length': 600,
    'episode_length_s': 10.,
}


def apply_benchmark(cfg):
    for path,value in SETTINGS.items():
        obj = cfg
        parts = path.split('.')
        for part in parts[:-1]:
            obj = getattr(obj,part)
        if not hasattr(obj,parts[-1]):
            raise AttributeError(path)
        setattr(obj,parts[-1],tuple(value) if isinstance(value,list) else value)


def make_env_cfg():
    # Set nullable fields before Hydra configclass serialization; Isaac Lab's
    # from_dict otherwise rejects None -> float/list overrides at runtime.
    from isaacsimenvs.tasks.simtoolreal.simtoolreal_env_cfg import SimToolRealEnvCfg
    cfg=SimToolRealEnvCfg()
    import yaml
    cfg.from_dict(yaml.safe_load((Path(__file__).resolve().parents[1]/'isaacsimenvs/cfg/task/SimToolReal.yaml').read_text()))
    apply_benchmark(cfg)
    return cfg


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--num-envs',type=int,default=24576)
    p.add_argument('--epochs',type=int,default=40000)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    a.out=a.out.resolve()
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'.hydra').exists(): p.error('Use a fresh output directory')
    runtime_tmp=a.out/'runtime_tmp'
    runtime_tmp.mkdir(exist_ok=True)
    os.environ['TMPDIR']=str(runtime_tmp)
    overrides={f'env.{k}':v for k,v in SETTINGS.items()}
    overrides.update({
        'env.scene.num_envs':a.num_envs,
        'agent.params.seed':42,
        'agent.params.config.name':'0_single_hammer_sapg_nodr',
        'agent.params.config.max_epochs':a.epochs,
        'agent.params.config.save_frequency':2000 if not a.smoke else 1,
        'agent.params.config.expl_coef_block_size':4096 if not a.smoke else a.num_envs//6,
        'hydra.run.dir':str(a.out),
    })
    if a.smoke:
        if a.num_envs%6: p.error('SAPG smoke requires six equal exploration blocks')
        overrides['agent.params.config.minibatch_size']=a.num_envs*16
        overrides['agent.params.config.central_value_config.minibatch_size']=a.num_envs*16
    cmd=[sys.executable,'-u','-m','supermimic_stl.sapg_entry','--task','Isaacsimenvs-SingleHammerBenchmark-Direct-v0',
         '--agent','rl_games_sapg_cfg_entry_point','--headless']
    cmd += [f'{k}={json.dumps(v,separators=(",",":"))}' for k,v in overrides.items()]
    print(json.dumps(dict(command=cmd,benchmark=SETTINGS)),flush=True)
    os.execv(sys.executable,cmd)

if __name__=='__main__': main()
