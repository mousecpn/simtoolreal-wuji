#!/bin/bash
#SBATCH --job-name=c4-amp-smoke
#SBATCH --partition=gpu
#SBATCH --account=fang
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/ps2338/simtoolreal
#SBATCH --output=/home/ps2338/simtoolreal/logs/c4-amp-smoke-%j.log
#SBATCH --requeue

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
hostname
nvidia-smi

exec .venv_isaacsim/bin/python -u scripts/verify_c4_amp.py \
  --device cuda:0 \
  --batch 32 \
  --steps 3 \
  --amp-dtype float16
