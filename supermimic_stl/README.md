# SuperMimic-style training for SimToolReal (opt-in v1)

论文：**SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model**, arXiv:2610.28480v1，§3 / Eq.1–3。

这是面向工具抓取、抬升和连续目标位姿任务的新训练器，**不是论文作者发布的代码，也不是已经证明优于 PPO/SAPG 的结果**。默认使用本仓库原生 KUKA/Sharpa 的 Isaac Lab / PhysX 环境；也支持现有单臂 Tianji/Wuji 环境。保留原任务的奖励、动作映射、目标随机游走、成功计数和掉落终止。

## 隔离约定

- 所有新增源文件只在 `supermimic_stl/`，不修改任何既有 Python/YAML、任务注册、依赖环境或 PPO/SAPG 训练入口。
- 直接构造局部子类以采集自动重置前的状态，不修改原任务类，不做全局 monkey patch。
- 输出只写到用户指定的新目录。已有含 `metadata.json` 的运行目录拒绝复用，断点恢复也输出到新目录。
- 使用现有 `.venv_isaacsim`，不安装/升级依赖。默认不启动 wandb、不上传数据。
- 运行临时资产写在各自输出目录的 `runtime_tmp/`，避开远程 `/tmp` 配额问题。不清理其他任务的文件。
- `latest.pt` 保存模型和优化器（默认省略 replay；可用 `--checkpoint-replay` 保存）；不覆盖或加载原 PPO/SAPG/B2m 权重。

## 启动

在 `/home/pinhao/simtoolreal` 执行：

```bash
# RTX 3060 12 GB 小规模配置；训练 10,000 次迭代。
bash supermimic_stl/run.sh --num-envs 16 --policy-batch 128 --out outputs/supermimic_stl/train_v1

# 改用现有 Tianji/Wuji 单臂任务（重新训练，不是加载 B2m）。
bash supermimic_stl/run.sh --robot tianji-wuji \
  --out outputs/supermimic_stl/train_tianji_v1

# 只用锤子工具分布做更小的起步实验。
bash supermimic_stl/run.sh --tool-types hammer \
  --out outputs/supermimic_stl/train_hammer_v1
```

默认 12,288 环境、六类工具、每类基础采样数 1（生成器可能为同类生成多个几何变体），world MLP `512×3`、actor `256×2`、critic `256×2`。使用 8 步 autoregressive 模型拟合和 32 步 imagination。完整参数见 `bash supermimic_stl/run.sh --help`；可通过 `SUPER_MIMIC_PYTHON` 指定另一已有兼容环境。

训练器先采集 128 步随机探索并固定状态归一化统计，预训练模型 100 次，随后每迭代采样 16 步、拟合模型 10 次、更新策略 1 次。world batch 为 128，policy batch 为 12,288；经验池存每环境最近 1,024 步。所有规模参数可调整。

## 恢复、评估与模型验证

```bash
# iterations 表示额外训练次数。环境重新 reset，replay 在此处显式断开。
# 训练恢复时 num-envs / replay-steps 必须与保存时相同；无 replay 时重新采集 warmup。
bash supermimic_stl/run.sh \
  --resume outputs/supermimic_stl/train_v1/latest.pt \
  --out outputs/supermimic_stl/train_v1_continued --iterations 10000

# 确定性策略在真实 PhysX 中评估，不在世界模型中自评。
# checkpoint 自动恢复机器人、工具池、网络和训练时容差等配置。
bash supermimic_stl/run.sh --mode eval \
  --resume outputs/supermimic_stl/train_v1/latest.pt \
  --out outputs/supermimic_stl/eval_v1 --num-envs 4 \
  --eval-episodes 32 --eval-max-steps 30000

# 重新采集未参与拟合的数据，不进行任何优化器更新。
bash supermimic_stl/run.sh --mode validate \
  --resume outputs/supermimic_stl/train_v1/latest.pt \
  --out outputs/supermimic_stl/model_validation_v1 --num-envs 4 \
  --collect-steps 128

mkdir -p outputs/supermimic_stl/test_tmp
TMPDIR="$PWD/outputs/supermimic_stl/test_tmp" \
  .venv_isaacsim/bin/python -m unittest supermimic_stl.test_core -v
```

评估采用每环境配额，分别报告完成回合数、平均奖励、平均达到目标数和达到全部目标的回合比例。触及步数上限时写明 `incomplete_step_cap`，不把未完成回合计作成功或完整评估。这里评估的是原随机目标任务，**不是 DexToolBench 24 条记录轨迹的评估，也不是实际工具功能测试**。

## 与论文设计的对应

| 论文设计 | 本实现 |
|---|---|
| 仿真采样 → 模型拟合 → 策略改善 | `train.py` 每迭代循环；缓冲区记录实际执行的加噪、裁剪后动作 |
| k-step autoregressive 模型拟合（k=8） | `Learner.fit_world`；跨 reset 的后续样本屏蔽 |
| 可微模型 + reward / observation / termination heads | residual MLP 预测增强状态、缩放奖励、物理终止和目标切换；状态显式含 clean actor/critic 观测，observation head 为取片段 |
| 冻结模型、BPTT 计算策略梯度（H=32） | `Learner.improve`；冻结权重但保留输入／动作 Jacobian |
| 短视野终值 critic + GAE / lambda-return | gamma=.99、lambda=.95；每次 rollout 固定 targets，5 epochs；critic minibatch 随规模受显存上限约束 |
| 可微软终止 survival weight | sigmoid 终止概率累计相乘；时间上限单独截断并 bootstrap pre-reset value |
| 收集 action noise、imagination 再加 sensor noise | action std=.1，normalized actor obs noise std=.01；两条执行路径一致 |
| 梯度控制、优化器 | norm clip 1；actor 再逐元素 clip .1；actor Adam beta=(.9,.95)，其余 Adam |
| 学习率日程 | 250,000 次迭代 cosine 衰减至初始值的 .03，随后固定 |

额外的任务适配：

1. **目标随机采样不作为可预测物理量回归。** 模型额外学习当前动作导致目标切换的概率。拟合使用实际切换标签和记录的新目标；想象中使用 soft switch 和独立采样的目标随机游走。目标关键点相对位置由预测物体位姿及新目标解析重建。这是离散切换的可微松弛，会带来近似偏差，非原仿真切换的精确梯度。
2. **捕捉 pre-reset 状态。** 子类在 `_get_rewards` 中采集状态，位于 `_get_dones` 后、DirectRLEnv 自动 reset 前。世界模型监督不使用返回的 reset observation。超时和物理终止分开保存；训练采样遇到超时还会断言 terminal timer 与 post-reset timer 不同。
3. **增强状态。** 除 clean actor/critic 观测，包含物体完整 root state、目标 pose、原始关节位置/速度、上一控制目标、抬升参考高度、桌面高度、近目标计数、成功数、时间计数和关键点偏移，避免只拿 actor observation 当作完整动力学状态。
4. **工具 context。** 按实际 spawn 的资产索引记录 one-hot identity、物体质量、尺寸；actor 不接收这些额外特权信息。checkpoint 校验工具 URDF 哈希和状态 schema。该 one-hot 模型不宣称能对任意未见工具预测动力学。
5. **奖励不另造。** 学习原环境返回奖励的 `.01` 缩放版本，原环境奖励和成功判据不变。

## 与论文和完整 SimToolReal 的差异 / 当前边界

- 这是单任务 goal-conditioned feedforward actor，不含论文的动作库、motion encoder/shared decoder、多接口训练、10 帧观测历史、困难动作 reset 采样或分布式 H100 训练；这些并非本工具任务训练循环所必需。
- v1 使用 **clean/no-delay dynamics**：关闭原任务的动作／观测延迟、物体 pose noise、随机外力/扭矩和摩擦倍率随机化；保留物体和关节初态、桌面高度、工具分布随机化。在真实仿真和想象中加入相同的独立观测噪声。没有宣称完整 sim-to-real robustness。
- 成功容差固定（默认 .075，原任务起始容差），不会从 replay 混合不同 curriculum 奖励/终止定义。可设置 `--success-tolerance .01` 直接训练更严格版本；后续 checkpoint 不可用不同容差“继续恢复”。
- 原任务初始目标是 absolute、成功后默认 delta。完整保留此行为；没有把任务改为单目标 episode。默认最多达到 50 个目标。
- 学到的接触模型可能被策略利用，必须以真实 PhysX 成功率和 held-out 预测误差判断效果。训练 MSE 降低不是成功证明；没有把论文的加速数字外推到工具操作。
- 模型验证在新采样数据上使用记录的目标切换条件，另报 switch BCE；它不是无限长自由 rollout 误差，也不能证明接触梯度正确。
- 同步保存模型、actor/critic、归一化、优化器、经验池、迭代数、采样数和 Python/NumPy/Torch/CUDA RNG。恢复时仿真重新起局，因此不是逐帧完全相同的重放恢复。

## 文件和输出

- `core.py`：无 Isaac 依赖的模型、经验池、解析目标处理、BPTT / critic。
- `adapter.py`：两个已有任务的独立适配及 pre-reset capture。
- `train.py` / `run.sh`：训练、恢复、评估、独立新数据模型验证。
- `policy.py`：`load_actor('.../actor.pt', device='cuda:0')` 返回接受原任务 clean policy observation 的确定性策略。
- `test_core.py`：梯度、模型冻结、目标投影、环形 replay、跨 reset 屏蔽、超时 bootstrap 和导出一致性测试。
- `metadata.json` / `args.json`：资产指纹、schema、源代码哈希、配置。
- `metrics.jsonl` / `result.json`：真实奖励/目标数、训练损失、actor 梯度范数、步数和运行时间。
- `latest.pt` / `actor.pt`：完整恢复文件和独立推理导出。
- `evaluation.json` / `world_validation.json`：对应独立运行结果。

实际短程验证结果见本目录 `VALIDATION.md`。没有默认启动无人看管的长时间训练。


## Unicorn fixed-object benchmark (2026-10-28)

`cluster.sbatch supermimic` and `cluster.sbatch sapg` launch isolated experiments.
No original task, SAPG trainer, or default configuration file is changed.
Both use the exact bundled `assets/fixed_hammer.urdf`, native 29 joint actions,
action smoothing 0.1, identical task/reward settings, fixed initial object/joints/
table, no sensor noise/latency/impulses/friction/scale/reset randomization,
seed 42, tolerance 0.075, max 50 goals, 600-step no-progress timeout.
Random goal sampling and policy exploration remain task/algorithm behavior.
The original task YAML is applied before shared benchmark overrides for both.

| Setting | SuperMimic-STL | Vanilla SAPG |
|---|---:|---:|
| Parallel real environments | 12,288 | 24,576 (user requested default) |
| Epochs | 40,000 | 40,000 |
| Real steps/environment/epoch | 16 | 16 |
| Real transitions/epoch | 196,608 | 393,216 |
| Imagined trajectories/policy update | 12,288 | none |
| Imagined trajectory horizon | 32 | none |
| Imagination microbatch | 512, accumulated into one actor step | n/a |

SuperMimic uses CPU replay (1,024 time steps), model batch 1,024, 10 model
updates and 1 policy update per epoch, 128 random-action warmup steps, 100
model pretraining updates. Only requested policy trajectory counts changed;
this remains a SuperMimic-inspired task adaptation, not an official reproduction.
Network sizes/recurrence and exploration differ between algorithms; this is
an end-to-end method comparison, not a controlled optimizer-only ablation.
Replay is omitted from checkpoints by default to avoid repeated ~50+ GB writes;
`--checkpoint-replay` includes it. Replay-free resumes refill warmup without
recalibrating the restored model. Periodic actors are retained every 2,000 epochs.
Both jobs request one A6000, 16 CPUs, 256 GB RAM, seven days. They do not requeue
automatically: failed/interrupted runs must be investigated before resuming.

Each job evaluates the final deterministic policy on seeds 101, 202, 303,
120 complete episodes per seed, 12 environments, same fixed asset and tolerance.
SAPG evaluation selects its zero-exploration condition and resets LSTM state at
episode boundaries. `python -m supermimic_stl.compare RUN_SM RUN_SAPG` aggregates
external task return, goals/episode, any-goal and all-goal success rates.
Compare training epoch, actual simulator transitions and elapsed wall time;
SAPG receives twice the real samples at the same epoch. One training seed does
not establish statistical significance. Training has not finished at submission.
