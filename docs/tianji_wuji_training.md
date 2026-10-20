# Tianji + Wuji 单手 / 双手训练

新环境使用 DexTele 的**右 Tianji 臂 + 右 Wuji 手**，固定躯干底座，共 7 个臂关节和 20 个手关节。任务仍是 SimToolReal 的工具抓取、抬升和目标位姿跟踪，沿用物体生成、奖励和 domain randomization；机器人资产、动作、观测、reset 和训练配置位于独立的 `isaacsimenvs/tasks/tianji_wuji/` 路径。

Gym ID 为 `Isaacsimenvs-SimToolReal-TianjiWuji-Direct-v0`。原有 `Isaacsimenvs-SimToolReal-Direct-v0` 继续使用 Kuka + Sharpa 配置。

双臂双手使用独立的 Gym ID `Isaacsimenvs-SimToolReal-TianjiWujiDual-Direct-v0` 和 `isaacsimenvs/tasks/tianji_wuji_dual/` 配置；通过训练参数 `--task` 选择。单手 ID、默认设置和动作顺序保持原样。双手版本固定同一躯干，保留 Tianji 左右两条运动链，各连接对应 Wuji 手，宽桌面上各放置一个独立工具。左手负责工具 1，右手负责工具 2；各自计算抓取、抬升和目标位姿误差，同时达到各自目标后才记为双手成功。

```bash
# 单臂单手（原设置）
.venv_isaacsim/bin/python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-TianjiWuji-Direct-v0 --headless

# 双臂双手
.venv_isaacsim/bin/python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-TianjiWujiDual-Direct-v0 --headless
```

双手策略为 54 维动作：先左臂 7 + 左手 20，再右臂 7 + 右手 20，各侧采用相同的臂增量和手关节位置控制语义。双手 actor/critic 观测分别为 268/309 维，包含双掌、十个指尖及两个工具的状态与目标。双手使用独立的 `TianjiWujiDualPPO.yaml` / `TianjiWujiDualSAPG.yaml` 及实验名称。双手策略需要独立训练，不能直接加载单手 checkpoint。当前双工具版本的观测维度也与此前单工具双手版本（249/282）不同，旧双手 checkpoint 需要重新训练。

双手桌面沿左右工作区方向（Y）加宽至原来的 3 倍，桌面为 0.475 × 1.2 m，桌高不变。双手工具默认分别在桌面 y = +0.28 m 和 -0.28 m 区域内采样，XY 噪声为 ±0.035 m；每个工具有独立目标，两个目标配对重采样。可通过 `env.tools.object_centers_xy` 调整两个工具的工作区域。单手的桌面、工具数量、初始化和训练设置保持原样。双手不接受仅描述单物体的 `fixed_start_pose`、`fixed_goal_pose` 或 `fixed_trajectory_file` 配置。

## 运行依赖与资产

使用仓库已有的 Isaac Sim / Isaac Lab Python 环境。在项目根目录启动；本机需先加载 `.envrc`，让 Isaac Sim 找到兼容的 `libxml2.so.2`：

```bash
source .envrc
```

机器人源资产已随 Git 收录在 `assets/robots/tianji_wuji/`，默认从该目录加载，**不需要同级 DexTele、sim_infra 或外部软链接**。包含 Tianji 双臂 meshes、原生 URDF、DexTele 的 Tianji/Wuji MJCF 场景，以及 Wuji 左右手的原生 MJCF、URDF 和 meshes；单手任务使用右侧，双手任务使用左右两侧。完整来源、范围和逐文件 SHA-256 见该目录的 `README.md` 与 `MANIFEST.json`。

环境启动时从仓库内的 canonical MJCF 按所选任务生成单侧或双侧 URDF，并复制 meshes 到本次运行的临时资产目录，再转换为 USD。生成缓存不纳入 Git；Isaac Sim / Isaac Lab 和 Python 依赖仍需单独安装。

如需开发时使用外部版本，可显式设置 `DEXTELE_ROOT=/absolute/path/to/DexTele`，或在训练命令中设置 `env.robot.dextele_root=/absolute/path/to/DexTele`。该外部 checkout 必须同时提供 MJCF、meshes 和对应侧原生 Wuji URDF（双手需要 `urdf/left.urdf` 与 `urdf/right.urdf`）；取消覆盖后恢复使用仓库内资产。

可先独立检查资产转换，无需启动 Isaac Sim：

```bash
.venv_isaacsim/bin/python isaacsimenvs/tasks/tianji_wuji/assets.py \
  --output-dir /tmp/tianji_wuji_asset_check
```

输出包括 URDF、meshes 和包含来源 hash、关节限位等信息的 `manifest.json`。

## 动作与观测

策略动作是裁剪到 `[-1, 1]` 的 27 维向量，固定顺序如下：

1. `Joint1_R` 到 `Joint7_R`：7 维臂动作，表示相对上一目标的关节位置增量，乘以 `env.action.dof_speed_scale * step_dt`。
2. `right_finger1_joint1` 到 `right_finger5_joint4`：按 finger 编号、再按 joint 编号排列的 20 维手动作，线性映射到各关节的位置限位。

臂和手分别使用 `arm_moving_average`、`hand_moving_average` 平滑，并裁剪到真实关节限位。策略顺序与 Isaac Lab 的内部关节顺序通过名称映射，不能直接假设 USD 导入顺序相同。默认 policy 观测为 134 维，非对称 critic 为 156 维；改变 `env.obs.obs_list` 或 `state_list` 时会重新推导空间大小。

需要从头训练新策略。原 Kuka + Sharpa checkpoint 的 29 维动作和机器人状态维度不同，不能直接用于该环境；PPO 与 SAPG 的探索条件结构也应使用各自对应的 checkpoint。

## 验证与启动

CPU 测试覆盖资产结构、独立正向运动学校验，以及乱序关节条件下的动作映射：

```bash
.venv_isaacsim/bin/python -m unittest discover -s tests -p 'test_tianji_wuji_*.py' -v
```

Isaac Sim smoke 测试创建两个 clone，检查关节和 body 身份、27 维动作、观测、有限数值和目标限位，并运行随机动作、局部 reset、超时自动 reset 和成功后目标重采样：

```bash
source .envrc
.venv_isaacsim/bin/python isaacsimenvs/tests/test_tianji_wuji_env_smoke.py \
  --headless --num_envs 2 --steps 40
```

先运行一个小规模 PPO 训练，检查完整 actor 和 critic 更新链路。16 个环境乘以 16 步 rollout 共 256 个样本，因此 actor、critic 的 minibatch 都设为 256：

```bash
source .envrc
.venv_isaacsim/bin/python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-TianjiWuji-Direct-v0 \
  --agent rl_games_cfg_entry_point --headless \
  env.scene.num_envs=16 env.assets.num_assets_per_type=1 \
  agent.params.config.max_epochs=2 \
  agent.params.config.minibatch_size=256 \
  agent.params.config.central_value_config.minibatch_size=256
```

正式 PPO 的独立配置是 `isaacsimenvs/cfg/train/TianjiWujiPPO.yaml`，默认使用 1024 个环境：

```bash
source .envrc
.venv_isaacsim/bin/python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-TianjiWuji-Direct-v0 \
  --agent rl_games_cfg_entry_point --headless
```

SAPG 使用 `TianjiWujiSAPG.yaml`。下面显式设为 1536 个环境、每块 256 个环境，共六个探索系数块：

```bash
source .envrc
.venv_isaacsim/bin/python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-TianjiWuji-Direct-v0 \
  --agent rl_games_sapg_cfg_entry_point --headless \
  env.scene.num_envs=1536 \
  agent.params.config.expl_coef_block_size=256
```

调整 SAPG 并行数时，保持环境数可被 `expl_coef_block_size` 整除，并同时检查 actor、critic minibatch 与 rollout 样本量的整除关系。训练输出写入 Hydra 本次运行目录，其中包含 rl_games checkpoint 和 summaries。`--checkpoint` 只应载入匹配的新机器人和算法配置的权重。

当前共享的 `--capture_viewer` HTML 导出器内置 Kuka + Sharpa 的 URDF 与关节名称，因此上述 Tianji/Wuji 命令不启用该选项。需要查看仿真窗口时，去掉 `--headless`。

## 已完成的验证

- CPU 测试 11 项通过，包括完整资产清单/hash、647 处包内引用、隔离目录转换与 8 组随机有效姿态的独立 FK 对照。
- 使用默认仓库内资产，独立 checkout 通过 2 环境、40 步随机动作、局部 reset、超时和目标重采样检查。
- RTX 3060 上，2 个并行环境通过 600 步随机动作、臂/手实际运动响应、局部 reset、超时自动 reset、成功后目标重采样检查。
- PPO（16 环境）和 SAPG（24 环境、每块 4 环境、六块）均完成 2 个 epoch，保存 checkpoint；权重和优化器状态张量均为有限数值。
- 原 Kuka + Sharpa 环境通过 2 环境、10 步 smoke 回归，仍使用 29 维动作与 140/162 维观测。其环境实现、训练入口、PPO/SAPG 配置均未修改。

这些验证证明训练链路可以运行；尚未进行长期训练或评估抓取成功率。机器人 PD 增益、初始姿态和任务分布可通过 `env.robot` 等配置继续调节。

## 随机 rollout 与视频

录制脚本通过 `--arms single|dual` 选择机器人，默认 `single`。以下先在 Isaac Sim 中运行双臂双手随机动作并记录实际物理状态，再用原始 URDF meshes 在 CPU 上渲染视频；无需训练好的策略，也不依赖 RTX 相机渲染。

```bash
source .envrc
.venv_isaacsim/bin/python scripts/capture_tianji_wuji_state.py \
  --arms dual --headless --steps 600 --trajectory --frame-stride 2 \
  --out outputs/tianji_wuji_dual_two_tools_video/trajectory.json
.venv_isaacsim/bin/python scripts/render_tianji_wuji_video.py \
  --trajectory outputs/tianji_wuji_dual_two_tools_video/trajectory.json \
  --out outputs/tianji_wuji_dual_two_tools_video/random_rollout.mp4
```

双手视频根据完整轨迹固定取景，分别展示整机和双手工作区域。双手录制使用 hammer / screwdriver 候选池，导出两个工具的独立实际轨迹、目标及本次仿真使用的加宽桌面 URDF，两个工具以橙色和青色区分。渲染器也兼容此前只有一个工具的轨迹。画面是实际仿真状态的 mesh 渲染，随机动作不代表策略已学会抓取。换成 `--arms single` 并使用独立输出目录，即可录制原单臂单手环境。轨迹、资产副本、MP4 和预览图保存在 `outputs/`，不纳入 Git。

双手物理验证命令：

```bash
source .envrc
.venv_isaacsim/bin/python -u isaacsimenvs/tests/test_tianji_wuji_dual_env_smoke.py \
  --headless --num_envs 2 --steps 120
```

当前 25 项 CPU 测试通过。双手已通过两侧臂/手实际运动、120 步随机动作、加宽桌面、两个工具独立位置、局部 reset、第二工具掉落重置、超时和配对目标成功检查；8 个环境的双工具 PPO 完成 2 个 epoch 并保存有限数值 checkpoint。这些运行验证不代表已学会双手抓取。

## Unicorn / fang：单臂 SAPG，关闭 domain randomization

从仓库根目录提交（1 张 RTX A6000、8 CPU、128 GB 内存，7 天上限）：

```bash
mkdir -p logs
sbatch scripts/train_tianji_wuji_nodr_fang.sbatch
```

脚本使用单右臂 + 右 Wuji 手任务，从头运行 SAPG，24,576 个并行环境（每块 4096，共六个探索系数块）、seed 42、100,000 epochs，每 10,000 epochs 保存 checkpoint。仅通过本次 job 的 Hydra 参数关闭观测/动作/物体状态延迟、物体位姿观测噪声、关节速度观测噪声、物体尺寸观测扰动、随机外力/力矩、物体/指尖摩擦随机化，以及相机位姿、深度增强和相机/student 延迟。机器人关节和桌面 reset 扰动归零，物体初始位姿固定为环境局部坐标 `(0, 0, 0.63, 1, 0, 0, 0)`（位置 + wxyz 四元数），避免默认 reset 仍随机采样物体朝向。工具资产池与目标采样保留为任务分布；正常任务默认配置不变。

日志为 `logs/tianji-wuji-sapg-nodr-<JOBID>.log`，Hydra 配置、checkpoint 与 TensorBoard summaries 位于 `outputs/slurm-<JOBID>-tianji-wuji-sapg-nodr/`。可用 `squeue -j <JOBID>` 查看调度状态，用 `tail -f logs/tianji-wuji-sapg-nodr-<JOBID>.log` 查看训练进度。集群使用方式见 [Unicorn 官方指南](https://it.coecis.cornell.edu/researchit/unicorn/)。

## 单手重置时的手部净空修复

单右臂任务默认使用 `RAISED_ARM_POSE`。在底座位置不变时，掌心相对 canonical
姿态抬高 0.20 m，XY 和朝向基本不变；按 URDF FK 计算，最低指尖从 z≈0.578 m
升到 z≈0.778 m。旧姿态的部分指尖低于约 0.60 m 的桌面，重置时可能与桌面、
工具发生初始穿插。该默认值只适用于单手任务；双手配置仍独立。

新训练（包括 `scripts/train_tianji_wuji_nodr_fang.sbatch`）自动使用新姿态。
旧 Hydra 配置会恢复旧 `env.robot.arm_default_joint_pos`；用旧 checkpoint 录制时，
给 `isaacsimenvs/play_video.py` 增加 `--safe_tianji_start`，即可在载入配置后覆盖为
抬高的姿态。此覆盖改变了 reset 分布，不能把旧策略在新姿态的效果当作重新训练的效果。
它用于清除初始手部穿插，不保证随机探索过程中不会再次把工具碰飞。

重复重置检查（需 GPU，使用无 DR 的已保存配置）：

```bash
.venv_isaacsim/bin/python isaacsimenvs/tests/test_tianji_wuji_reset_clearance.py \
  --headless --enable_cameras \
  --config outputs/exploration-comparison/tianji-wuji/config.yaml \
  --out outputs/wuji-reset-clearance/raised
# 加 --legacy_pose 对照旧姿态。
```

检查固定手部关节目标，在 24 个环境重复 reset 5 次，每次仿真 60 步，输出
物体最大上升高度/速度和 5 秒视频；不使用训练策略或随机动作。

本次无 DR 固定起始位姿测试：旧姿态在 120 个环境/重置观测窗口中有 10 个出现
物体上升超过 10 cm，最大上升约 1.03 m、最大向上速度约 5.04 m/s；抬高姿态
分别为 0 个、0 m、0.019 m/s。视频和原始指标保存在
`outputs/wuji-reset-clearance/`。这验证重置净空，不验证旧策略在新姿态下的成功率。
