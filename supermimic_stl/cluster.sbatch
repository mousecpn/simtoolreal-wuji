#!/usr/bin/env bash
#SBATCH --partition=fang
#SBATCH --account=fang
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/ps2338/simtoolreal/logs/single-hammer-aligned-%j.log
set -euo pipefail
algorithm="${1:?supermimic or sapg}"
repo="${BENCHMARK_REPO:-/home/ps2338/simtoolreal-sm-aligned}"
cd "$repo"
export PYTHONPATH="$repo:$repo/rl_games${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=YES WANDB_MODE=disabled PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PXR_WORK_THREAD_LIMIT=16
python_bin=/home/ps2338/simtoolreal/.venv_isaacsim/bin/python
out="/home/ps2338/simtoolreal/outputs/single_hammer_aligned_20261028/${algorithm}-${SLURM_JOB_ID}"
mkdir -p "$out"
export TMPDIR="$out/tmp"
mkdir -p "$TMPDIR"
git rev-parse HEAD > "$out/git_commit.txt"
nvidia-smi > "$out/gpu.txt"
"$python_bin" -m unittest supermimic_stl.test_core supermimic_stl.test_aligned -v > "$out/unit-tests.log" 2>&1
if [[ "$algorithm" == supermimic ]]; then
  "$python_bin" -m supermimic_stl.train --headless --benchmark-single-object \
    --num-envs 12 --policy-batch 12288 --imagination-microbatch 512 \
    --replay-device cpu --replay-steps 64 --warmup-steps 32 --model-batch 16 \
    --model-pretrain 2 --world-updates 1 --iterations 2 --sensor-noise 0 \
    --out "$out/smoke"
  test -f "$out/smoke/result.json"
else
  "$python_bin" -m supermimic_stl.benchmark --num-envs 12 --epochs 2 --smoke --out "$out/smoke"
  "$python_bin" - "$out/smoke" <<'PY2'
import sys
from pathlib import Path
assert list(Path(sys.argv[1]).rglob('nn/last*ep_2*.pth')), 'SAPG preflight did not finish'
PY2
fi
train_started=$(date +%s)
if [[ "$algorithm" == supermimic ]]; then
  "$python_bin" -m supermimic_stl.train --headless --benchmark-single-object \
    --num-envs 12288 --policy-batch 12288 --imagination-microbatch 512 \
    --replay-device cpu --replay-steps 1024 --model-batch 1024 \
    --iterations 40000 --sensor-noise 0 --save-every 2000 --out "$out/train"
  test -f "$out/train/result.json"
  checkpoint="$out/train/actor.pt"
  extra=()
elif [[ "$algorithm" == sapg ]]; then
  "$python_bin" -m supermimic_stl.benchmark --num-envs 24576 --epochs 40000 --out "$out/train"
  checkpoint=$("$python_bin" - "$out/train" <<'PY'
import sys
from pathlib import Path
paths=list(Path(sys.argv[1]).rglob('nn/last*ep_40000*.pth'))
if not paths: raise SystemExit('No final SAPG epoch-40000 checkpoint found')
print(max(paths,key=lambda p:p.stat().st_mtime))
PY
)
  extra=(--sapg-config "$out/train/.hydra/config.yaml")
else
  exit 2
fi
train_finished=$(date +%s)
echo "$((train_finished-train_started))" > "$out/training_elapsed_seconds.txt"
for seed in 101 202 303; do
  "$python_bin" -m supermimic_stl.evaluate_benchmark --algorithm "$algorithm" \
    --checkpoint "$checkpoint" "${extra[@]}" --seed "$seed" --num-envs 12 \
    --eval-episodes 120 --eval-max-steps 18000 --out "$out/eval-$seed" --headless
done
