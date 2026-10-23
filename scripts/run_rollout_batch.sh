#!/usr/bin/env bash
set -euo pipefail

out="$1"
checkpoint="$2"
config="$3"
tolerance="$4"
shift 4

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OMNI_KIT_CACHE_PATH="/tmp/ps2338/rollout-${SLURM_JOB_ID}/ov_cache"
mkdir -p "$OMNI_KIT_CACHE_PATH"

for seed in "$@"; do
  echo "Recording seed ${seed}"
  timeout --kill-after=30s 15m .venv_isaacsim/bin/python -u isaacsimenvs/play_video.py \
    --task Isaacsimenvs-SimToolReal-SE3-Direct-v0 \
    --checkpoint "$checkpoint" \
    --config "$config" \
    --num_envs 6 --env_idx 5 --num_assets_per_type 2 \
    --seed "$seed" --steps 600 --video_fps 30 \
    --deterministic --zero_exploration --success_tolerance "$tolerance" \
    --out "$out/seed-${seed}.mp4" \
    --metrics_out "$out/seed-${seed}.json" \
    > "$out/seed-${seed}.log" 2>&1
  echo "Finished seed ${seed}"
done
