#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f .envrc ]]; then source .envrc; fi
export PYTHONPATH="$PWD:$PWD/rl_games${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
python_bin="${SUPER_MIMIC_PYTHON:-$PWD/.venv_isaacsim/bin/python}"
exec "$python_bin" -m supermimic_stl.train "$@"
