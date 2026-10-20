#!/usr/bin/env bash
set -euo pipefail
cd /home/ps2338/simtoolreal

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export OMNI_KIT_CACHE_PATH="/tmp/ps2338/video-${SLURM_JOB_ID}/ov_cache"
mkdir -p "$OMNI_KIT_CACHE_PATH"

# 修改这里，选择 checkpoint 和对应训练配置
checkpoint="outputs/slurm-378081/0_simtoolreal_sapg/last/model.pth"
config="outputs/slurm-378081/.hydra/config.yaml"

out="outputs/videos-378081-${SLURM_JOB_ID}"
mkdir -p "$out"

# 固定一份权重，确保三个视频使用同一个 checkpoint
cp "$checkpoint" "$out/model.pth"
cp "$config" "$out/config.yaml"

for seed in 11 22 33; do
  .venv_isaacsim/bin/python -u isaacsimenvs/play_video.py \
    --task Isaacsimenvs-SimToolReal-Direct-v0 \
    --checkpoint "$out/model.pth" \
    --config "$out/config.yaml" \
    --num_envs 6 --env_idx 5 \
    --num_assets_per_type 2 \
    --seed "$seed" --steps 600 --video_fps 30 \
    --deterministic --zero_exploration \
    --out "$out/seed-${seed}.mp4" \
    --metrics_out "$out/seed-${seed}.json" \
    > "$out/seed-${seed}.log" 2>&1
done

echo "视频已保存到：$out"
