# Isaac Sim Smoke Test 与视频录制（本地工作站）

适用于带 NVIDIA RTX GPU 的 Linux 工作站，使用 Bash 直接运行，不需要集群或 Slurm。
所有命令都从本机仓库根目录执行。不要在同一张 GPU 上同时启动多个 Isaac Sim / Kit 进程，也不要与大规模训练争用显存。

## 1. 准备环境
- 在工作站上取得包含最新 [play_video.py](../isaacsimenvs/play_video.py) 的项目代码，并进入本机的 `simtoolreal` 仓库目录。
- 按 [安装指南](isaacsim_installation.md)在本机安装 `.venv_isaacsim`。不要直接复制另一台机器的虚拟环境，其中包含绝对路径和平台相关二进制文件。
- 使用 Python 3.11、Isaac Lab `2.3.2.post1`、PyTorch `2.7.x+cu126`，并检查系统、GPU 和驱动符合 Isaac Sim 5.1 的要求。不是所有支持 CUDA 的 GPU 都支持 RTX 渲染。
- 流程曾在 RTX A6000 48 GB 上成功写出视频，这不是最低硬件要求，也不保证其他工作站自动兼容。
- 安装需要下载约 15 GB 级别的依赖，还应给解压、缓存、模型和视频留出磁盘空间。600 步录制会先缓存约 1.9 GB 原始 RGB 帧，主内存还需容纳仿真和模型。

如果没有 uv，可先执行以下命令，再按安装指南安装依赖：
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

在仓库根目录检查环境并设置运行变量，每次开启新终端后重新执行：
```bash
test -f isaacsimenvs/play_video.py
test -x .venv_isaacsim/bin/python
nvidia-smi
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export OMNI_KIT_CACHE_PATH="$HOME/.cache/simtoolreal/ov_cache"
mkdir -p "$OMNI_KIT_CACHE_PATH" logs
.venv_isaacsim/bin/python -c 'import torch, isaaclab, coacd; print("torch:", torch.__version__); print("CUDA:", torch.cuda.is_available()); assert torch.cuda.is_available()'
```

请先阅读并接受 NVIDIA 的许可条款；`OMNI_KIT_ACCEPT_EULA=YES` 用于非交互启动。
脚本使用 headless 模式，不要求弹出 Isaac Sim 窗口，但仍需要 GPU 渲染驱动。
多 GPU 工作站默认使用 `cuda:0`，运行前确认该卡空闲。

## 2. 执行 Smoke Test
依次启动三个独立进程。下面使用子 shell，任一步失败就停止，但不会退出你的交互终端：

```bash
(
set -euo pipefail
timeout --kill-after=30s 15m .venv_isaacsim/bin/python \
  isaacsimenvs/tests/test_load_isaacsim.py 2>&1 | tee logs/smoke-load.log

timeout --kill-after=30s 15m .venv_isaacsim/bin/python \
  isaacsimenvs/tests/test_gym_register.py 2>&1 | tee logs/smoke-register.log

timeout --kill-after=30s 20m .venv_isaacsim/bin/python \
  isaacsimenvs/tests/test_simtoolreal_env_smoke.py \
  --num_envs 8 --num_assets_per_type 2 --steps 10 \
  2>&1 | tee logs/smoke-env.log
echo SMOKE_TESTS_PASSED
)
```

预期成功输出：

1. `Isaac Sim load OK`：Kit 和 GPU 基本启动正常。
2. `Phase 1 registration smoke test OK`：Gym 注册、配置加载和 YAML overlay 正常。
3. `8 Object + 8 GoalViz prims`、`10 steps, no NaN`：场景实例数量、reset 和短程仿真正常。
4. 最后出现 `SMOKE_TESTS_PASSED`：三个命令都成功退出。

这些测试不验证训练收敛或策略成功率。第一次运行可能下载扩展、编译 shader 和转换 URDF，耗时会比后续运行长。

## 3. 下载预训练模型

模型还不存在时执行（需要网络）：

```bash
.venv_isaacsim/bin/python download_pretrained_policy.py
```

得到 `pretrained_policy/config.yaml` 和 `pretrained_policy/model.pth`。此模型是 SAPG 模型，录制时必须显式选择 SAPG 配置，不能使用视频脚本默认的 PPO 配置。

## 4. 先录一段短片

```bash
(
set -euo pipefail
timeout --kill-after=30s 12m .venv_isaacsim/bin/python isaacsimenvs/play_video.py \
  --checkpoint pretrained_policy/model.pth \
  --agent rl_games_sapg_cfg_entry_point \
  --num_envs 6 --env_idx 5 \
  --num_assets_per_type 2 --seed 11 \
  --steps 60 --video_fps 30 --deterministic \
  --out isaacsimenvs/videos/manual/preview.mp4 \
  2>&1 | tee logs/video-preview.log
)
```

正常结束后得到约 1 秒的 `isaacsimenvs/videos/manual/preview.mp4`。
Linux 桌面可用播放器打开（无桌面的 SSH 会话可下载到本机播放）：

```bash
xdg-open isaacsimenvs/videos/manual/preview.mp4
```

请确认画面不是全黑、机器人和物体取景正常，再录制完整 demo。

## 5. 连续录制三个 Demo

以下命令在前台串行运行，每段 600 步，约 10 秒；每段单独保存日志。
需要保持终端打开，SSH 到工作站时可在 tmux 会话中执行。

```bash
(
set -euo pipefail
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export OMNI_KIT_CACHE_PATH="$HOME/.cache/simtoolreal/ov_cache"
run_id=$(date +%Y%m%d-%H%M%S)
output_dir="isaacsimenvs/videos/pretrained-${run_id}"
log_dir="logs/pretrained-${run_id}"
mkdir -p "$OMNI_KIT_CACHE_PATH" "$output_dir" "$log_dir"
printf 'Videos: %s\nLogs: %s\n' "$output_dir" "$log_dir"
for seed in 11 22 33; do
  timeout --kill-after=30s 12m .venv_isaacsim/bin/python isaacsimenvs/play_video.py \
    --checkpoint pretrained_policy/model.pth \
    --agent rl_games_sapg_cfg_entry_point \
    --num_envs 6 --env_idx 5 --num_assets_per_type 2 \
    --seed "$seed" --steps 600 --video_fps 30 --deterministic \
    --out "$output_dir/demo-seed-${seed}.mp4" \
    2>&1 | tee "$log_dir/demo-seed-${seed}.log"
done
echo ALL_PRETRAINED_DEMOS_RENDERED
)
```

输出目录使用时间戳，不覆盖旧 demo：

- 视频：`isaacsimenvs/videos/pretrained-日期时间/demo-seed-11.mp4`，以及 seed 22、33。
- 日志：`logs/pretrained-日期时间/demo-seed-11.log`，以及 seed 22、33。
- 单段成功标志：`Wrote ... frames to ...mp4`。
- 全部成功标志：`ALL_PRETRAINED_DEMOS_RENDERED`。

运行中可按 Ctrl+C 停止。不要使用会影响其他训练的全局 kill 命令。
三个 seed 表示不同随机初始化，不代表三个指定工具类别；成功写出 MP4 也不等于通过策略成功率评估。

## 录制原理和参数

[play_video.py](../isaacsimenvs/play_video.py) 以 headless rendering 模式启动 Kit，创建环境和 RGB 相机，加载 checkpoint，再逐步执行策略、采集 RGB，最后用 imageio/FFmpeg 写 MP4。

- `--num_envs 6 --env_idx 5`：现有 SAPG player 为六个环境分配六组条件 ID，相机跟随零探索系数组环境。
- `--deterministic`：使用策略均值作为动作，不代表场景随机性被完全消除。
- `--num_assets_per_type 2`：生成 24 个候选物体 URDF，而非默认 1200 个，缩短启动时间，但减少物体多样性。
- `--seed`：设置环境随机种子。
- `--steps 600 --video_fps 30`：当前环境为 60 Hz 控制，通常得到 300 帧、10 秒视频。
- `--out`：指定输出文件，父目录自动创建。

相机优先使用 `record_camera_eye/target`，不存在时回退到 `viewer.eye/lookat`。
脚本还会覆盖保存 `isaacsimenvs/videos/first_frame.png`，供首帧诊断。
帧先保存在内存，rollout 正常完成后才编码视频，因此中途终止不保证留下 MP4。
FFmpeg 可能把 1920 x 1080 自动调整到 1920 x 1088，这是编码对齐提示，不是失败。

## 常见问题

- `uv: command not found`：安装 uv 并将 `$HOME/.local/bin` 加入当前 PATH。
- `CUDA: False`：检查驱动、GPU 和 PyTorch CUDA 构建，不要使用 CPU-only PyTorch。
- `AttributeError: ... record_camera_eye`：工作站使用了旧录制脚本，请同步包含 viewer 回退修复的代码。
- `extra_params`、`sigma [6,29]`、LSTM 输入 `172/140` 不匹配：误用 PPO 加载 SAPG 模型，应指定 `--agent rl_games_sapg_cfg_entry_point`，不要删权重或给观测补零。
- GPU 显存不足：先关闭争用同一 GPU 的程序；smoke test 可降低环境数。SAPG 六环境用于保留六组条件，先验证短片，不要随意改权重。
- timeout 退出码 `124` 或强制结束后的 `137`：已超时，不表示成功。先检查 traceback；若只是首次编译慢，可适当增大 `12m/15m/20m`。
- Kit 关闭可能卡住，Python 异常会返回失败状态，外层 timeout 设置了 30 秒强制结束宽限时间。
- headless 下 GLFW/display warning 不一定致命，以退出码、traceback 和输出文件为准。
- 相机初始化姿态诊断曾显示全零；非黑帧不等于取景正确，仍需人工播放检查。

本指南命令沿用此前 RTX 工作节点上通过的 smoke test 和视频流程；工作站版尚未在你的另一台机器上实测。

## Tianji + Wuji 单手 / 双手随机视频

Tianji/Wuji 使用独立脚本，不使用上述 Kuka/Sharpa 预训练 checkpoint。`--arms single` 保留原单臂单手设置，`--arms dual` 选择双臂双手。以下命令记录真实物理 rollout，再在 CPU 上渲染原始 meshes：

```bash
source .envrc
.venv_isaacsim/bin/python scripts/capture_tianji_wuji_state.py \
  --arms dual --headless --steps 600 --trajectory --frame-stride 2 \
  --out outputs/tianji_wuji_dual_video/trajectory.json
.venv_isaacsim/bin/python scripts/render_tianji_wuji_video.py \
  --trajectory outputs/tianji_wuji_dual_video/trajectory.json \
  --out outputs/tianji_wuji_dual_video/random_rollout.mp4
```

视频及首、中、末帧预览写入该输出目录。它展示随机动作下的仿真状态，并非训练策略效果。训练时使用 `--task Isaacsimenvs-SimToolReal-TianjiWujiDual-Direct-v0` 选择双手，使用 `--task Isaacsimenvs-SimToolReal-TianjiWuji-Direct-v0` 选择单手。完整说明见 [Tianji/Wuji 训练指南](docs/tianji_wuji_training.md)。
