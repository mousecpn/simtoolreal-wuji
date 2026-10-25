#!/bin/bash
#SBATCH --job-name=kuka-sharpa-c4-se3-sapg-nodr
#SBATCH --partition=fang
#SBATCH --account=fang
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --chdir=/home/ps2338/simtoolreal
#SBATCH --output=/home/ps2338/simtoolreal/logs/kuka-sharpa-c4-se3-sapg-nodr-%j.log
#SBATCH --requeue

set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMNI_KIT_CACHE_PATH="/tmp/${USER}/kuka-sharpa-c4-se3-${SLURM_JOB_ID}/ov_cache"
mkdir -p "$OMNI_KIT_CACHE_PATH"
hostname
nvidia-smi

# C4 equivariant SAPG: six SE3 arm actions and 22 hand joint targets.
# Keep FP32 and disable input normalization as required by the C4 policy.
# Job-local overrides: the task's normal DR defaults are unchanged.
# Keep the tool asset pool and goal sampling as task diversity. Fix robot,
# table and object starts; disable all perturbations and sensor augmentation.
exec .venv_isaacsim/bin/python -u isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-C4-SE3-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point --headless \
  env.scene.num_envs=24576 \
  agent.params.config.minibatch_size=98304 \
  agent.params.config.central_value_config.minibatch_size=98304 \
  agent.params.config.expl_coef_block_size=4096 \
  agent.params.seed=42 \
  agent.params.config.name=0_kuka_sharpa_c4_se3_sapg_nodr \
  agent.params.config.max_epochs=100000 \
  agent.params.config.save_frequency=10000 \
  env.domain_randomization.use_obs_delay=false \
  env.domain_randomization.obs_delay_max=0 \
  env.domain_randomization.use_action_delay=false \
  env.domain_randomization.action_delay_max=0 \
  env.domain_randomization.use_object_state_delay_noise=false \
  env.domain_randomization.object_state_delay_max=0 \
  env.domain_randomization.object_state_xyz_noise_std=0.0 \
  env.domain_randomization.object_state_rotation_noise_degrees=0.0 \
  env.domain_randomization.object_scale_noise_multiplier_range='[1.0,1.0]' \
  env.domain_randomization.joint_velocity_obs_noise_std=0.0 \
  env.domain_randomization.force_scale=0.0 \
  env.domain_randomization.torque_scale=0.0 \
  env.domain_randomization.object_friction_scale_range='[1.0,1.0]' \
  env.domain_randomization.fingertip_friction_scale_range='[1.0,1.0]' \
  env.reset.reset_position_noise_x=0.0 \
  env.reset.reset_position_noise_y=0.0 \
  env.reset.reset_position_noise_z=0.0 \
  env.reset.fixed_start_pose='[0.0,0.0,0.63,1.0,0.0,0.0,0.0]' \
  env.reset.reset_dof_pos_random_interval_arm=0.0 \
  env.reset.reset_dof_pos_random_interval_fingers=0.0 \
  env.reset.reset_dof_vel_random_interval=0.0 \
  env.reset.table_reset_z_range=0.0 \
  env.reset.table_reset_xy_range_m='[0.0,0.0]' \
  env.reset.table_reset_yaw_range_deg=0.0 \
  env.student_obs.use_camera_pose_rand=false \
  env.student_obs.use_depth_aug=false \
  env.student_obs.use_camera_delay=false \
  env.student_obs.use_student_obs_delay=false \
  hydra.run.dir="outputs/slurm-${SLURM_JOB_ID}-kuka-sharpa-c4-se3-sapg-nodr"
