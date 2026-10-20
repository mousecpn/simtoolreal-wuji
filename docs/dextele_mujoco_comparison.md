# DexTele MuJoCo 与本仓库 Isaac Sim 的 SimToolReal 对照

本次按本机工作区源码比较：DexTele HEAD `fbc3d48`，SimToolReal HEAD `313d5aea`，并包含当前未提交的 Tianji/Wuji 单臂、双臂实现。初版结论来自静态调用链检查及一个 CPU 奖励数值核验；后续按用户指定项目更新了 Isaac Sim 单臂 Tianji/Wuji 默认值，见下表及训练文档。未测量跨引擎成功率。

## 比较范围

- DexTele 主体：`/home/pinhao/DexTele/third_party_patches/wuji-mjlab/simtoolreal/`。这是 Tianji 右臂 + Wuji Hand 1 的 MuJoCo Warp/mjlab 移植，训练入口是 `train_rl_games.py`。另有 `rl/wuji_reorient_env.py`、`rl/wuji_tool_env.py` 等使用部分 SimToolReal 配方的环境，不应混为同一实现。
- 本仓库原版：`isaacsimenvs/tasks/simtoolreal/`，Kuka iiwa + Sharpa，29 动作，actor/critic 为 140/162 维。
- 同机器人主要比较对象：`isaacsimenvs/tasks/tianji_wuji/`，Tianji 右臂 + Wuji，27 动作，134/156 维。
- 双臂版本：`isaacsimenvs/tasks/tianji_wuji_dual/`，54 动作，268/309 维，两件工具、两个目标。这已是不同的 MDP。

核心结论：三个单臂版本共享 SimToolReal 的基本奖励和控制思路，但场景、控制器、观测定义、目标分布、成功判定和训练配置仍有差异，不能视作仅替换物理引擎。DexTele 的 Tianji/Wuji 移植在臂 stiffness、外展限制、网络和物体池规模上更接近 Kuka/Sharpa 配方；本仓库 Tianji/Wuji 已对齐指定的控制与训练参数，但仍保留不同的场景、观测和部分动力学配置。

## 三个单臂版本的默认配置

表中的 Isaac Tianji/Wuji 为指定参数对齐后的当前默认值；双臂配置未随此次调整。以下 MuJoCo 指 `SimToolRealCfg()` / `train_rl_games.py` 的默认 procedural 任务；额外命令行参数会改变实际实验。

| 项目 | 本仓库 Isaac Sim Kuka/Sharpa | DexTele MuJoCo Tianji/Wuji | 本仓库 Isaac Sim Tianji/Wuji |
|---|---|---|---|
| 机器人 | Kuka iiwa14 + 左 Sharpa，7+22关节 | Tianji右臂 + Wuji Hand 1，7+20关节 | Tianji右臂 + Wuji右手，7+20关节 |
| 物理 / 策略频率 | 120 / 60 Hz，decimation=2 | 相同 | 相同 |
| 动作 | 29维：7臂增量+22手绝对位置 | 27维：7臂增量+20手绝对位置 | 27维：7臂增量+20手绝对位置 |
| 平滑 / 臂增量系数 | 臂/手 EMA=0.1，speed=1.5 | 相同 | 相同 |
| 臂 stiffness | `[600,600,500,400,200,200,200]` | 与 Kuka 数值相同，但不是相同动力学 | 与DexTele相同 |
| 臂 damping | 约 `[27.03,27.03,24.67,22.07,9.75,9.15,9.15]` | 按 ζ=0.3 和 Tianji armature 计算，约 `[6.49,6.49,5.92,5.30,2.31,2.31,2.31]` | `[8,8,8,8,4,4,4]` |
| 臂 armature | ImplicitActuator 未显式设置 | 前四项0.195、后三项0.074 | ImplicitActuator 未显式设置 |
| 臂 URDF/actuator effort | URDF各关节300 | Tianji `[108,108,66,66,18,18,18]` | canonical Tianji 同样的 effort 范围 |
| 手执行器 | Sharpa逐关节 stiffness=0.9～13.2、damping≈0.0278～0.4085；逐关节 armature | Wuji mjlab XML actuator，包含其辨识参数 | canonical sim_infra MJCF 转 URDF/USD；逐关节stiffness与DexTele相同（约0.1803～0.6859）；damping=0.01、armature=0.001 |
| 食/中/无名/小指外展限位 | restricted Sharpa URDF，±0.03491 rad（约2°） | 默认限制为同样的±2°；`--hand_abduction` 放开 | 默认限制为±0.03491 rad；可通过robot.restrict_abduction关闭 |
| 场景 | 一条主动 Kuka 臂和左 Sharpa 手 | 默认 DexTele 场景，含静态左臂/手和立柱 | 单右臂场景 |
| 桌面 | 0.475×0.4 m，顶面 z≈0.53 | 1.2×0.6 m，顶面 z≈0.64 | 0.475×0.4 m，顶面z≈0.64（只对齐高度） |
| 底座 / 四元数 | `(0,0.8,0)`，单位四元数 | E坐标 `(0.16,0.55,0.87)`，yaw=-90° | `(-0.48634,0.28420,1.01)`，单位四元数 |
| 初始臂姿态 | `[-1.571,1.571,0,1.376,0,1.485,1.308]` | `DT_ARM_Q0`，IK求得 | `[-0.712,-0.963,0,-0.85,-0.356,0,0]` |
| reset 位置噪声系数，臂/手 | 0.1 / 0.1 | 0.1 / 0.1；拒绝穿桌采样，最多10次，失败回q0 | 0.1 / 0.1；无相同拒绝采样 |
| actor / critic 观测 | 140 / 162 | 134 / 156 | 134 / 156 |
| palm_pos | `iiwa14_link_7` + 局部偏移 `(0,-0.02,0.16)` | palm link + 偏移 `(0.025,0,0.058)` 的掌心site | palm link原点，偏移为0 |
| 指尖取点 | Sharpa DP body + 局部偏移 `(0.02,0.002,0)` | Wuji tip sites，按食中无名拇小顺序 | Wuji固定tip links；按正则发现顺序，未显式排成MuJoCo顺序 |
| object_scale 观测噪声 | 默认 `[1,1]` | `[0.9,1.1]` | 默认 `[1,1]` |
| force/torque 质量因子 | 各环境自身质量 | 默认env0质量，可切换为per_env | 各环境自身质量 |
| 成功保持 | 默认累计10步，可间断 | 连续10步 | 连续10步 |
| 达标 bonus | 每个近目标步+100 | 成功时一次+1000 | 成功时一次+1000 |
| procedural 池 | 12分布×100=1200 | 12分布×100=1200 | 12分布×100=1200 |
| actor 网络 | LSTM1024 + MLP `[1024,1024,512,512]` | 与Kuka版相同规模 | LSTM1024 + MLP `[256,256,128]` |
| critic 网络 | MLP `[1024,1024,512,512]` | 相同规模 | MLP `[256,256,128]` |
| 默认环境数 / minibatch | 24576 / 98304 | 24576 / 98304 | 24576 / 98304 |
| 默认SAPG分组 | block size=4096，共6块 | 默认6块，每块4096 | block size=4096，共6块 |

观测维度不包含 learner 额外的 SAPG 探索系数条件。reset 位置噪声是从 q0 向关节限位内均匀采样位置插值的系数，不是直接的弧度半幅。effort 表中分别注明了 URDF 或 actuator 来源，不代表已测量运行时饱和行为；同样的 stiffness 也不代表不同机器人能产生相同响应。手kp已从实际XML逐项核对，但手damping和armature并未跟随对齐。

来源：MuJoCo 的 `config.py`、`robot.py`、`env.py`、`objects.py`、`train_rl_games.py`；Isaac 的 `tianji_wuji/{config,assets,scene,actions,observations,reset,termination}.py`、`simtoolreal/utils/{scene_utils,obs_utils,reset_utils,action_utils,reward_utils,termination_utils}.py`、`simtoolreal/simtoolreal_env_cfg.py`、`cfg/task/SimToolReal.yaml`、`cfg/train/{SimToolReal,TianjiWuji}{PPO,SAPG}.yaml` 和 restricted Sharpa URDF。

## 加入 Kuka/Sharpa 后，如何归因差异

1. **本仓库 Kuka/Sharpa → 本仓库 Tianji/Wuji**：物理引擎及共享任务配方保持同一路径，但改变了机器人、关节维度、手控制增益、底座/q0、掌心/指尖定义、桌高、成功规则和MLP规模；指定的臂stiffness、reset噪声、物体池、LSTM、环境数和minibatch已对齐DexTele。这是机器人迁移加训练配置修改，不能视作仅替换URDF。
2. **本仓库 Kuka/Sharpa → DexTele Tianji/Wuji**：同时改变引擎、机器人和场景；不过 DexTele 保留了 Kuka 配方的臂stiffness、约±2°外展限制、1200物体、大网络和24576环境。它还沿用旧 Isaac Gym 论文启动配置中的连续成功、scale noise和env0扰动力质量因子，因而与本仓库当前 Isaac Sim 默认设置不同。
3. **DexTele Tianji/Wuji → 本仓库 Tianji/Wuji**：机器人型号相近，但资产、部分执行器参数、场景、任务分布和MLP规模仍不同，也不能作为纯引擎对照。

特别注意：本表 Kuka/Sharpa 一栏指当前 **Isaac Sim** 版本。论文训练使用的 **Isaac Gym** Kuka/Sharpa 及其 `launch_training.py` 覆盖项，是 DexTele 移植的来源；这两套 Kuka 配置不能合并成一栏。前述连续成功、尺寸观测噪声和质量因子差异，恰好说明了这一点。

## 目标采样也不同

三个版本均为首个绝对目标、后续默认在上个目标附近作 ±0.1 m / ≤90° 的随机变化。两个 Isaac Sim 版本使用同一个 `goal_sampling.py`，共享以下目标采样差异：

1. MuJoCo 默认 DexTele 目标框是 x∈[-0.35,0.35]、y∈[-0.15,0.25]、z∈[0.71,1.06]；两个Isaac单臂都使用原版 x∈[-0.35,0.35]、y∈[-0.2,0.2]、z∈[0.6,0.95]。坐标系及机器人朝向也不同，不能只减去桌高就认为完全对齐。
2. MuJoCo 首目标额外保证 z≥物体初始 z−0.05+liftingBonusThreshold，即初始 z+0.10 m。Isaac 的 `sample_absolute_goal_pose` 只按目标框均匀采样，没有这道约束，因而允许更低的首目标。
3. MuJoCo 的旋转轴来自归一化 U(0,1)^3，使用右乘 `q*dq`，并规范到 qw≥0；Isaac 使用归一化高斯轴、左乘 `dq*q`，没有同样的符号规范。实际旋转分布和策略看到的四元数表示均有差别。物体旋转观测噪声也使用这两种不同的扰动方式。

来源：MuJoCo `env.py:_sample_delta_quat_xyzw/reset_target_pose`；Isaac `simtoolreal/utils/goal_sampling.py` 和 `tianji_wuji/observations.py:_perturb_quat`。

## 奖励主体相同，但 Isaac 成功步存在调用顺序问题

两边默认都包含：抬升前指尖距离的历史最优进度×50；抬升 shaping×20；一次性 lift bonus=300；抬升后关键点历史最优进度×200；臂/手 L1 关节速度惩罚 0.03/0.003。这里名为 action penalty 的项实际惩罚关节速度。关键点奖励和成功条件默认使用相同的固定尺寸关键点，容差从 0.075 收紧至 0.01，距离阈值还乘 keypoint_scale=1.5。当前Kuka版默认累计10步，两个Tianji单臂版本均为连续10步。

MuJoCo 在本步算完奖励后，于下一次 `pre_physics_step` 切换目标。Isaac 的调用链则是：

`_get_dones → compute_terminations → reset_goal_trackers → _get_rewards`

`reset_goal_trackers` 先把 `_closest_keypoint_max_dist` 和 `_closest_fingertip_dist` 清为 -1，再立即换目标；后续奖励仍使用旧目标的缓存距离，导致成功那一步的关键点进度奖励被清掉。以旧历史最优距离 0.02 m、本步 0.01 m 为例，正常进度奖励是 `200*(0.02-0.01)=2`；清成 -1 后得到 0。本次直接调用仓库纯 Torch 奖励函数，CPU 数值核验得到 2.0 与 0.0。成功 bonus 使用保留的 `_is_success`，这不意味着 +1000 本身也丢失。

Isaac 还会在切换目标时清掉指尖历史最优，而 MuJoCo 的目标切换没有相同的指尖记录重置。相同调用顺序也出现在原 Kuka 版和双臂版；本次参数对齐没有修复这一奖励调用顺序问题。

来源：`tianji_wuji/env.py`、`termination.py:compute_terminations`、`reset.py:reset_goal_trackers` 和共享 `simtoolreal/utils/reward_utils.py`。

## 物理引擎和资产方面

MuJoCo 是 mjlab 场景 + MuJoCo Warp 仿真，MDP 和 step/reset 由移植环境自己管理。两个 Isaac 版本均为 Isaac Lab `DirectRLEnv` + PhysX TGS；Kuka/Sharpa 从仓库 URDF 转为 USD，Tianji/Wuji 经 canonical MJCF→URDF→USD。两者的接触求解、驱动器和碰撞资产处理不同，数字相同的摩擦系数也不能保证相同接触行为。

MuJoCo 专门用指尖系数 1.0 和接触 priority 去近似 PhysX 指尖1.5、物体0.5 的平均组合，并使用手部碰撞分组、禁用臂自碰撞。两个 Isaac 版本均使用指尖材质1.5、物体0.5，启用自碰撞；Kuka/Sharpa 按专用相邻 link 映射过滤，Tianji/Wuji 按 canonical 碰撞排除和相邻 link 过滤。这些需要比较最终编译出的碰撞对及接触，不能仅比较开关。

MuJoCo 逐物理步手动施加物体角阻尼0.5和角速度上限64 rad/s，并增加非有限数值的 reset。这是对旧 Isaac Gym 行为的显式模拟；Isaac Tianji 的对象生成处没有显式设置相同的角阻尼和角速度上限，不能假设运行时默认值相等。

两边 procedural 物体都使用长方体/胶囊工具；Isaac 虽然先生成 cylinder URDF，但转换处显式 `replace_cylinders_with_capsules=True`，因此不能把“圆柱对胶囊”列为已确认的形状差异。MuJoCo 通过启动事件修改每个环境的几何/质量/惯量，Isaac 预先生成多个 URDF/USD 再分配给环境。

## DexTele 的额外任务与本仓库双臂任务

DexTele `taskSet=A` 包含 cube、cuboid、cylinder、screwdriver、hammer、pickup、pen 七类任务，另有掌内起始、桌边悬空起始、已抓持状态库、任务专用目标序列、每任务容差课程和可选平滑惩罚。它们是可选分支，不能当作 procedural 默认全部开启；本仓库 Tianji 单臂未实现相同的 A 任务体系。

本仓库双臂版则是两套主动臂/手、两个独立工具及目标。奖励对两件工具求和，并对完整目标配对发放成功 bonus；任何一件掉落可导致 episode 结束。DexTele 默认左臂只是静态障碍，无法直接对照双臂训练结果。

## 如何做公平的跨引擎实验

优先选两边的单臂 procedural 任务；固定机器人资产版本、坐标变换、底座/q0、桌面、目标框和相同的物体池，再对齐关节限位、PD/armature、掌心/指尖取点与顺序、连续成功规则、目标旋转分布、scale noise 和扰动力的质量因子。先解决成功步奖励顺序问题，再对齐尚未统一的MLP和总训练transitions；LSTM、默认并行环境、minibatch和SAPG block数现已对齐。

验证顺序应为同姿态 FK 和观测向量对照、同状态奖励/成功标签对照、固定目标的短动作序列响应，最后才比较训练曲线和成功率。本次源码对照能确认上述配置与语义差异，不能据此断言哪个引擎训练效果更好，或两个 checkpoint 能直接互换。
