# IsaacSim (Isaac Lab) installation

The `isaacsimenvs/` package runs the SimToolReal task in Isaac Sim via Isaac Lab. It requires **Python 3.11** (Isaac Sim 5.x / Isaac Lab 2.3.x requirement) and lives in a **second venv** at `.venv_isaacsim/`, separate from the Python 3.8 Isaac Gym venv that [isaacgym_installation.md](isaacgym_installation.md) sets up. The two environments are independent — never install Isaac Gym packages into `.venv_isaacsim` or vice versa.

## Prerequisites

- Python 3.11
- NVIDIA GPU with driver >= 525.60
- CUDA 12+
- Ubuntu 22.04 or 24.04 is recommended for the supported Isaac Sim 5.1 RTX
  configuration.
- `uv` for package management ([install instructions](https://docs.astral.sh/uv/getting-started/installation/))

## Install

```bash
uv venv .venv_isaacsim --python 3.11

# PyTorch for CUDA 12.6 (Isaac Sim 5.x is built against this)
uv pip install --python .venv_isaacsim/bin/python torch --index-url https://download.pytorch.org/whl/cu126

# Vendored rl_games + inference deps
uv pip install --python .venv_isaacsim/bin/python -e ./rl_games/
uv pip install --python .venv_isaacsim/bin/python \
  omegaconf hydra-core "gym==0.23.1" scipy numpy yourdfpy requests tqdm tyro "imageio[ffmpeg]" wandb termcolor

# Isaac Lab + Isaac Sim (~15 GB download; first launch builds RTX shaders, takes ~2-5 min)
uv pip install --python .venv_isaacsim/bin/python \
  "isaaclab[isaacsim,all]==2.3.2.post1" --extra-index-url https://pypi.nvidia.com

# Offline collision decomposition (CoACD) + tyro CLI fix. Install AFTER isaaclab so it
# wins the resolution: tyro.cli (used by dextoolbench/generate_collision_meshes.py) needs
# NoExtraItems from typing_extensions>=4.13, but the isaaclab install pulls in 4.12.2.
# typing_extensions 4.15 is verified compatible with isaaclab 2.3.2.post1.
uv pip install --python .venv_isaacsim/bin/python coacd "typing_extensions>=4.13"

# Register repo-local packages (isaacsimenvs, isaacgymenvs, deployment, ...)
uv pip install --python .venv_isaacsim/bin/python -e . --no-deps
```

`--no-deps` on the last line is required: the root `pyproject.toml` pins `numpy==1.23.0`, `warp-lang==0.10.1`, and `isaacgym-stubs`, which all conflict with Python 3.11 / Isaac Sim. The repo packages themselves install cleanly.

Keep the version pins exact (`isaaclab==2.3.2.post1`, torch 2.7.x+cu126) — newer Isaac Lab releases change `DirectRLEnv` / `UrdfConverter` APIs that `isaacsimenvs` depends on.

Verify:

```bash
.venv_isaacsim/bin/python -c "
import torch, isaaclab, isaacsimenvs
print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
print('isaaclab:', isaaclab.__file__)
"
```

## Running

Non-interactive launches need the EULA env var:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

Optional speedup — point the Omniverse shader cache at a local SSD instead of NFS:

```bash
export OMNI_KIT_CACHE_PATH=/scratch/$USER/ov_cache
mkdir -p "$OMNI_KIT_CACHE_PATH"
```

Smoke tests (one Kit boot per process, ~1-2 min each; run individually, not via pytest):

```bash
.venv_isaacsim/bin/python isaacsimenvs/tests/test_load_isaacsim.py
.venv_isaacsim/bin/python isaacsimenvs/tests/test_gym_register.py
.venv_isaacsim/bin/python isaacsimenvs/tests/test_simtoolreal_env_smoke.py \
  --num_envs 8 --num_assets_per_type 2 --steps 10
```

To save a headless smoke-test video for offline inspection:

```bash
.venv_isaacsim/bin/python isaacsimenvs/tests/test_simtoolreal_env_smoke.py \
  --num_envs 8 --num_assets_per_type 2 --steps 600 \
  --video outputs/smoke_test.mp4
```

Recording automatically enables camera rendering. The MP4 follows environment 0
with the configured viewer camera; use `--video_env 1` to inspect another
environment or `--video_width 1280 --video_height 720` for higher resolution.
The smoke test uses a single-GPU compatibility renderer by default: it disables
the Fabric Scene Delegate and multi-GPU RTX startup, neither of which this task
needs. Pass `--no-video_compat` to restore Isaac Lab's default rendering path.
It also removes the script's `--video <path>` option from Kit's raw command
line after parsing it. `video` is a Kit-reserved option; forwarding it would
create an unintended default RTX viewport before the offscreen render product.
At the default 60 simulation steps per second, 600 steps produce about 10 seconds.
Actions remain zero, as in the original smoke test; this is not a trained-policy
rollout. Video encoding requires `imageio[ffmpeg]` from the installation steps.
If using compatibility libraries, launch from the terminal where their
`LD_LIBRARY_PATH` is configured. Avoid running another Isaac Sim process on the
same GPU during recording.

Training:

```bash
.venv_isaacsim/bin/python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-Direct-v0 \
  --agent rl_games_cfg_entry_point \
  --headless
```

## Gotchas

- **AppLauncher before isaaclab imports.** Isaac Lab sub-namespaces (`isaaclab.sim`, `isaaclab.envs`, ...) only resolve after `AppLauncher(args)` runs. Any script importing `isaaclab.*` must instantiate `AppLauncher` first.
- **Kit shutdown can hang** after the work is done. Scripts flush stdout/stderr and call `os._exit(0)` rather than waiting for a clean Kit teardown.
- **First SimToolReal startup is slow**: object URDFs are procedurally generated and converted to USD per run (no cache); startup scales with `num_assets_per_type`.
- **One Isaac Sim instance per GPU.** Booting a second Kit process while another is running on the same GPU can crash the booting one mid-startup. Wait for training/eval/demo processes to finish before launching another.
