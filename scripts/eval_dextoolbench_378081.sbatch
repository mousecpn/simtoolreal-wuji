#!/bin/bash
#SBATCH --job-name=dextool-378081
#SBATCH --partition=fang
#SBATCH --account=fang
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --chdir=/home/ps2338/simtoolreal
#SBATCH --output=/home/ps2338/simtoolreal/logs/dextoolbench-378081-%j.log
#SBATCH --requeue

set -uo pipefail

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OMNI_KIT_CACHE_PATH="/tmp/${USER}/dextoolbench-378081-${SLURM_JOB_ID}/ov_cache"
mkdir -p "$OMNI_KIT_CACHE_PATH" "logs"

CHECKPOINT="outputs/slurm-378081/0_simtoolreal_sapg/last/model.pth"
# RlPlayer expects the rl_games agent YAML (not the full Hydra env config).
CONFIG="isaacsimenvs/cfg/train/SimToolRealSAPG.yaml"
OUT="outputs/dextoolbench-378081-${SLURM_JOB_ID}"
mkdir -p "$OUT"

hostname
nvidia-smi
printf 'checkpoint=%s\nconfig=%s\noutput=%s\n' "$CHECKPOINT" "$CONFIG" "$OUT"

declare -a CASES=(
  "hammer claw_hammer swing_down"
  "hammer claw_hammer swing_side"
  "hammer mallet_hammer swing_down"
  "hammer mallet_hammer swing_side"
  "spatula flat_spatula serve_plate"
  "spatula flat_spatula flip_over"
  "spatula spoon_spatula serve_plate"
  "spatula spoon_spatula flip_over"
  "eraser flat_eraser wipe_smile"
  "eraser flat_eraser wipe_c"
  "eraser handle_eraser wipe_smile"
  "eraser handle_eraser wipe_c"
  "screwdriver long_screwdriver spin_vertical"
  "screwdriver long_screwdriver spin_horizontal"
  "screwdriver short_screwdriver spin_vertical"
  "screwdriver short_screwdriver spin_horizontal"
  "marker sharpie_marker draw_smile"
  "marker sharpie_marker write_c"
  "marker staples_marker draw_smile"
  "marker staples_marker write_c"
  "brush blue_brush sweep_forward"
  "brush blue_brush sweep_right"
  "brush red_brush sweep_forward"
  "brush red_brush sweep_right"
)

total=${#CASES[@]}
for ((i = 0; i < total; i++)); do
  read -r category object task <<< "${CASES[$i]}"
  case_dir="$OUT/$category/$object/$task"
  mkdir -p "$case_dir"
  echo "[$((i + 1))/$total] $category/$object/$task"
  .venv_isaacsim/bin/python -u dextoolbench/eval_isaacsim.py \
    --object_category "$category" \
    --object_name "$object" \
    --task_name "$task" \
    --config_path "$CONFIG" \
    --checkpoint_path "$CHECKPOINT" \
    --output_dir "$case_dir" \
    --num_episodes 10 \
    > "$case_dir/run.log" 2>&1
  status=$?
  if [[ ! -f "$case_dir/eval.json" ]]; then
    echo "FAILED: $category/$object/$task (exit $status; no eval.json)" | tee -a "$OUT/failures.log"
  else
    echo "DONE: $category/$object/$task (exit $status)"
  fi
done

.venv_isaacsim/bin/python - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("*/*/*/eval.json")):
    data = json.loads(path.read_text())
    rel = path.relative_to(root).parts
    rows.append({
        "category": rel[0],
        "object": rel[1],
        "task": rel[2],
        **data,
    })

summary = {
    "checkpoint": "outputs/slurm-378081/0_simtoolreal_sapg/last/model.pth",
    "num_combinations": len(rows),
    "num_episodes": sum(len(row["episode_goal_pcts"]) for row in rows),
    "avg_goal_pct": sum(row["avg_goal_pct"] for row in rows) / len(rows) if rows else None,
    "avg_time_sec": sum(row["avg_time_sec"] for row in rows) / len(rows) if rows else None,
    "results": rows,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: summary[k] for k in ("num_combinations", "num_episodes", "avg_goal_pct", "avg_time_sec")}, indent=2))
PY
