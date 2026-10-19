# Tianji 单臂 + Wuji 单手训练

新环境使用 DexTele 的**右 Tianji 臂 + 右 Wuji 手**，固定躯干底座，共 7 个臂关节和 20 个手关节。任务仍是 SimToolReal 的工具抓取、抬升和目标位姿跟踪，沿用物体生成、奖励和 domain randomization；机器人资产、动作、观测、reset 和训练配置位于独立的 `isaacsimenvs/tasks/tianji_wuji/` 路径。

Gym ID 为 `Isaacsimenvs-SimToolReal-TianjiWuji-Direct-v0`。原有 `Isaacsimenvs-SimToolReal-Direct-v0` 继续使用 Kuka + Sharpa 配置。

## 运行依赖与资产

使用仓库已有的 Isaac Sim / Isaac Lab Python 环境。在项目根目录启动；本机需先加载 `.envrc`，让 Isaac Sim 找到兼容的 `libxml2.so.2`：

```bash
source .envrc
```

机器人源资产已随 Git 收录在 `assets/robots/tianji_wuji/`，默认从该目录加载，**不需要同级 DexTele、sim_infra 或外部软链接**。包含 Tianji 双臂 meshes、原生 URDF、DexTele 的 Tianji/Wuji MJCF 场景，以及 Wuji 左右手的原生 MJCF、URDF 和 meshes；训练仍只使用右臂和右手。完整来源、范围和逐文件 SHA-256 见该目录的 `README.md` 与 `MANIFEST.json`。

环境启动时从仓库内的 canonical MJCF 生成单右臂单右手 URDF，并复制 meshes 到本次运行的临时资产目录，再转换为 USD。生成缓存不纳入 Git；Isaac Sim / Isaac Lab 和 Python 依赖仍需单独安装。

如需开发时使用外部版本，可显式设置 `DEXTELE_ROOT=/absolute/path/to/DexTele`，或在训练命令中设置 `env.robot.dextele_root=/absolute/path/to/DexTele`。该外部 checkout 必须同时提供 MJCF、meshes 和原生 Wuji `urdf/right.urdf`；取消覆盖后恢复使用仓库内资产。

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
