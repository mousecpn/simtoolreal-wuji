#!/bin/bash
#SBATCH --job-name=isaacsim-setup
#SBATCH --partition=default_partition
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ps2338/simtoolreal/isaacsim-install-%j.log

set -euo pipefail
cd /home/ps2338/simtoolreal
hostname
nvidia-smi
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="/scratch/$USER/uv-cache"
export OMNI_KIT_CACHE_PATH="/scratch/$USER/ov_cache"
mkdir -p "$UV_CACHE_DIR" "$OMNI_KIT_CACHE_PATH"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version
if [[ ! -x .venv_isaacsim/bin/python ]]; then
    uv venv .venv_isaacsim --python 3.11
fi
uv pip install --python .venv_isaacsim/bin/python "torch==2.7.1" --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv_isaacsim/bin/python -e ./rl_games/
uv pip install --python .venv_isaacsim/bin/python omegaconf hydra-core "gym==0.23.1" scipy numpy yourdfpy requests tqdm tyro "imageio[ffmpeg]" wandb termcolor
uv pip install --python .venv_isaacsim/bin/python "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com
uv pip install --python .venv_isaacsim/bin/python coacd "typing_extensions>=4.13"
uv pip install --python .venv_isaacsim/bin/python -e . --no-deps
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
.venv_isaacsim/bin/python -c 'import torch, isaaclab, isaacsimenvs; print("torch:", torch.__version__, "cuda:", torch.cuda.is_available()); assert torch.cuda.is_available(); print("isaaclab:", isaaclab.__file__)'
timeout 15m .venv_isaacsim/bin/python isaacsimenvs/tests/test_load_isaacsim.py
timeout 15m .venv_isaacsim/bin/python isaacsimenvs/tests/test_gym_register.py
timeout 20m .venv_isaacsim/bin/python isaacsimenvs/tests/test_simtoolreal_env_smoke.py --num_envs 8 --num_assets_per_type 2 --steps 10
printf '%s\n' 'ISAACSIM_INSTALL_AND_SMOKE_TESTS_PASSED'