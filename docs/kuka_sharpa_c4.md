# KUKA + Sharpa C4 等变循环策略

基于 `wuji/main` 的 KUKA + Sharpa SE3 实现（`3031bb7f` 引入，已拉取至
`85de6076`），动作仍为 `[dx, dy, dz, rx, ry, rz, hand_0, ..., hand_21]`。
前六维是世界轴下的平移和旋转向量增量，沿用原有 differential IK、尺度、
关节限制和手部指令语义。新网络需要从头训练。

## 群作用和输入

C4 是绕每个环境局部原点的世界 +Z 轴旋转 0/90/180/270 度。
regular tensor 的形状是 `(..., 4, channels)`；旋转 k 个四分之一圈对应
`torch.roll(feature, k, dims=-2)`。

| 输入 | 转换 |
| --- | --- |
| joint position、velocity、previous joint targets | 每个标量为 `C4.irrep(0)` |
| palm/object/goal position | 使用减去 `env_origins` 的 XYZ，XY 为 `irrep(1)`，Z 为 `irrep(0)` |
| palm/object/goal orientation | xyzw quaternion 转为旋转矩阵前两列，各列为 `irrep(1) + irrep(0)`；q 和 -q 的特征相同 |
| fingertip、keypoint、线速度和角速度 | 每个 XYZ 向量为 `irrep(1) + irrep(0)` |
| bounding box coordinates | 用 object-local grasp-box 尺寸和 object pose 构造八个固定物体角点编号的世界系角点，每点为 `irrep(1) + irrep(0)` |
| object-local box dimensions、距离、进度等标量 | 每个标量为 `irrep(0)` |

这里的 bounding box 与原来的 `object_scales` 使用同一个 grasp box，并非完整工具的
collision mesh 包围盒。角点编号跟随物体，不能用旋转后重新按世界轴编号的 AABB
min/max 直接替换。goal pose 作为显式输入加入，原来的 goal-relative keypoints 也保留。

新任务 policy/critic 原始观测维度为 **174/196**。网络的字段顺序由
`network.obs_fields` 指定，必须与环境的 `obs.obs_list` 一致。
`ObservationEncoder` 输出一个拼接的 **180 维**张量：90 个 invariant 标量，
后接 30 个 XYZ 向量（含三个姿态的六个旋转矩阵列）。对应 escnn `FieldType` 为
`90 × irrep(0) + 30 × (irrep(1) + irrep(0))`。没有四槽复制或逆旋转展开；
原先的 720 维展开输入已替换。`irrep(1)` 在 C4 下是标准二维 XY 旋转矩阵，
`irrep(0)` 是一维常数表示。

## 网络

`simtoolreal_shared/c4.py` 使用 `escnn==1.0.11`，以 `cyclic_group(4)` 和
`no_base_space(C4)` 声明纤维表示。拼接输入通过 `escnn.nn.Linear` 从混合 irreps
映射到 45 个 regular field（180 个标量），再送入 LSTM。该投影不加激活，
避免在 LSTM 前引入原模型没有的非线性。

`C4LSTM` 的输入/循环门映射全部使用 `escnn.nn.Linear`，门输出为四组 regular
field。sigmoid、tanh 和逐元素乘法均与 regular 表示的循环置换可交换，
因此 h 和 c 都是 regular feature。没有普通稠密线性层或手写 circular convolution
代替 escnn 的等变权重约束。
支持多层、非零初态和逐环境逐时间步 reset；done 在处理对应输入前清零 h/c。

默认 256 regular channels，即每层 h、c 各 1024 个标量，与原始 policy 相同。
LSTM 后使用等变 LayerNorm：对全部 1024 个标量做归一化，但 affine scale/bias 在
每个 regular field 的四个槽之间共享。随后是四层 escnn Linear + ELU，标量宽度
`[1024, 1024, 512, 512]`，对应 regular field 数 `[256, 256, 128, 128]`。
这些逐元素激活与 regular 置换可交换。该顺序对应原模型的
`LSTM -> LayerNorm -> MLP -> action/value`。rl_games 边界将状态打包为
`(layers, batch, 4 * channels)`，内部恢复显式的群维。保留循环状态，不在 LSTM 前后
使用任意的普通全连接混合群维。

actor 同样使用 `escnn.nn.Linear`，从 regular field 映射到
`2 × (irrep(1) + irrep(0)) + 22 × irrep(0)`，对应两个等变三维向量和
22 个 invariant 手部动作。critic 用 escnn Linear 输出一个 `irrep(0)`。Gaussian
的平移 x/y 标准差共享，旋转 x/y 标准差共享，z 和各手部通道独立可学习，保证
整个对角 Gaussian 分布的等变性。逐坐标动作 clamp 与 C4 的符号置换也可交换。

训练配置关闭 `normalize_input`，因为普通逐坐标 running mean/std 会破坏等变性；
builder 会拒绝开启此选项。value normalization 可以使用。新配置采用 PPO 和共享的
invariant critic；原有 SAPG/asymmetric critic 配置没有套用到这条网络。

保证范围是网络的输入/状态/输出群作用。固定机器人基座、关节限制、IK 和场景
不一定构成 C4 对称的物理系统，因此不声称旋转后的物理轨迹完全相同。对非零
循环状态检查等变性时，必须同时旋转 h 和 c。

escnn 按 field-major 顺序保存 regular 特征，即每个 field 的四槽连续排列；
LSTM 显式张量为 `(…, 4, channels)`，代码在边界进行转置和打包，验证脚本
直接与 escnn 表示矩阵比较，防止两种布局混淆。escnn 的 eval 展开矩阵作为
派生缓存不进入 checkpoint；载入后重新生成，支持 train/eval 间严格加载。
扩大网络后的参数布局与之前手写 C4 模型及 64 regular channels 的小型 escnn
checkpoint 不兼容，旧 checkpoint 和视频仍保留。回放旧 escnn checkpoint 时应使用
其保存的 Hydra config，而非当前默认网络配置。

## 运行

沿用本机 Isaac Sim 安装所需的 EULA 和兼容动态库设置（见安装文档）。
安装 escnn（其 `py3nj` 依赖在本机使用现有 gfortran 编译）：

```bash
uv pip install --python .venv_isaacsim/bin/python 'escnn==1.0.11'
```

项目也提供 `equivariance` optional dependency。旧网络不会主动导入 escnn。

```bash
.venv_isaacsim/bin/python scripts/verify_c4_equivariance.py
.venv_isaacsim/bin/python scripts/verify_c4_equivariance.py --device cuda --full-size
.venv_isaacsim/bin/python isaacsimenvs/tests/test_c4_se3_env_smoke.py --headless
```

完整训练入口的小规模检查：

```bash
.venv_isaacsim/bin/python isaacsimenvs/train.py \
  --task Isaacsimenvs-SimToolReal-C4-SE3-Direct-v0 --headless \
  env.scene.num_envs=2 env.assets.num_assets_per_type=1 \
  agent.params.config.max_epochs=1 \
  agent.params.config.horizon_length=4 agent.params.config.seq_length=4 \
  agent.params.config.minibatch_size=8 agent.params.config.mini_epochs=1 \
  hydra.run.dir=outputs/c4_escnn_baseline_ppo_smoke
```

正式训练可增大环境数、minibatch 和 epoch 数，保持 minibatch 与序列长度、采样总量
兼容。默认使用 `SimToolRealC4PPO.yaml`。

验证脚本检查四个群元素、两层 LSTM、非零 h/c、reset、序列分段、姿态和角点输入、
quaternion 符号不变性、混合 irrep 输入和 regular 投影、动作/value/Gaussian，
以及参数更新和 eval 缓存下的等变性。还覆盖 train/eval 四种 checkpoint 保存/加载组合。
真实环境 smoke test 检查策略推理、28 维动作、PhysX IK、关节限制、有限值、
机械臂实际运动、单环境 reset，以及通过 rl_games model wrapper 的反向传播和参数更新。

## 之前手写 C4 checkpoint 的 rollout 视频

`outputs/c4_demos_epoch1/index.html` 汇总 hammer、screwdriver 和 marker 三段
约 10 秒视频。使用迁移 escnn 之前的 1-epoch smoke checkpoint 的 deterministic mean，
真实执行 PhysX + SE3 IK，并记录关节、物体和 goal 位姿；视频由原始 URDF mesh
离线渲染，不依赖本机不可用的 RTX 相机。它们用于查看当前策略行为，尚非收敛的
抓取示范。左侧为覆盖整段运动范围的固定整臂视角，右侧为手部近景，青色框表示 goal。

旧视频可直接播放或用已保存的轨迹 JSON 重新渲染。旧 checkpoint 不能装入新的 escnn
网络。录制新模型时，向 `scripts/capture_c4_rollouts.py` 传入新的 escnn checkpoint
及对应 Hydra config，并使用新的输出目录；`scripts/render_c4_rollouts.py` 不依赖网络，
可渲染两版模型的轨迹。

录制沿用保存的环境配置，将环境数改为六个、工具池改为三个类型各两个变体，并固定
seed 11、工具顺序，从每类中选一个环境。默认录制 600 个 60 Hz 策略步，每两步采样，
加上初始帧得到 301 帧。轨迹 JSON 保留动作和 reset 信息，`manifest.json` 记录本次
checkpoint SHA256 和视频来源。

## 参数量与单步计算量

运行 `.venv_isaacsim/bin/python scripts/compare_c4_policy_cost.py`，结果保存到
`outputs/c4_policy_cost.json`。脚本实际构建当前配置的网络，统计可训练参数，
并将逐层矩阵 MAC 数与 PyTorch `FlopCounterMode` 的结果核对。

| 模型 | 可训练参数 | 每步矩阵 FLOPs |
| --- | ---: | ---: |
| 当前 escnn C4：1024 标量 LSTM + 四层等变 MLP | 1,969,028 | 15,724,832 |
| 对照非等变：相同输入、1024D LSTM、投影/归一化/MLP/输出结构 | 7,875,965 | 15,724,832 |
| 原始非等变：140D 输入、1024D LSTM、四层 MLP | 7,679,545 | 15,332,352 |
| 原始模型额外的 asymmetric central critic | 2,004,481 | 4,002,816 |

口径：batch=1、sequence=1、FP32、eval forward，包含动作均值和 value 输出；
一个 MAC 计两个 FLOPs。参数包含 Gaussian log standard deviation；不计缓存矩阵
和其他 buffer。FLOPs 只计矩阵乘法，不含 bias、逐元素运算、激活、归一化、观测编码、
采样、反向传播、环境或 IK。前三行不含额外 central critic；原始训练系统若两网都计算，
合计 9,684,026 参数和 19,335,168 单步矩阵 FLOPs。当前 C4 没有额外 central critic。

当前 C4 与原始 policy 的 LSTM 隐状态和后续 MLP 标量宽度相同。矩阵 FLOPs
比原始模型高约 2.56%，因为输入编码为 180D（原始为 140D），且多一个
180D 混合 irrep 到 180D regular 的投影。这个投影保留了用户要求的输入表示。
可训练参数比原始 policy 少约 74.36%。普通 LayerNorm 的 affine 参数不能直接
使用；当前按 regular field 共享四槽参数，保证等变性。

同结构非等变对照的矩阵 FLOPs 与 C4 相同，C4 约减少 75% 参数。escnn eval 将
受约束的权重展开并缓存为相同大小的稠密矩阵，所以没有四倍的矩阵计算节省。
训练模式还需 basis expansion，未计入本表。实际 latency 还取决于 kernel、batch、
精度和内存访问，本脚本没有测量 latency。

历史的小型 escnn 网络（64 regular、无后续 MLP）为 145,075 参数、1,155,584
单步矩阵 FLOPs；这些旧结果不再代表当前默认配置。

扩大后的模型已通过 CPU float64/float32、完整宽度 CUDA float64/float32 等变验证，
CUDA 最大误差分别为 2.13e-14 / 7.63e-6（包括两层 LSTM、MLP、LayerNorm、
Gaussian、更新参数及 checkpoint 检查）。默认一层生产配置还通过真实环境
2 env × 16 step smoke test，覆盖 IK/关节限制、局部 reset 和 backward+optimizer；
最大关节单步运动 0.0182 rad，等变误差 3.81e-6。
