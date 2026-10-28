# 实际验证记录

日期：2026-10-28，Asia/Shanghai。机器：`pinhao@100.94.225.98`，RTX 3060 12 GB，使用 `/home/pinhao/simtoolreal/.venv_isaacsim`。

## 结论

独立训练、模型反传、checkpoint 恢复、actor 导出、真实 PhysX 评估和新数据模型验证已跑通。原有代码保持不变。**尚未训练出完成任务的策略；本记录不构成成功率或加速优势证明。**

## 自动测试和隔离

- 新增 8 项核心测试通过：BPTT actor 更新、冻结模型权重但保留输入梯度、目标投影、目标随机游走、replay wrap、跨 reset 屏蔽、超时 bootstrap / 物理终止差异、独立 actor 导出一致性。
- 原仓库 27 项测试、814 个子测试通过。合并执行结果：`35 passed, 814 subtests passed`。
- 对开始工作前的 259 个既有源文件逐一比对 SHA-256，内容均未改变。
- `git diff --binary` 与开始工作前完全一致，保留了原有未提交修改。
- 原有测试和首次场景启动曾被远程 `/tmp` 配额阻断，改为本次输出目录下的独立临时目录后通过；未删除旧文件，也未改全局临时目录配置。

## 训练和恢复

| 运行 | 设置 / 结果 |
|---|---|
| KUKA/Sharpa smoke | 4 环境、2 次迭代、128 条转移；模型、actor、critic 均产生有限损失和非零 actor 梯度；触发 4 个 timeout |
| KUKA checkpoint 恢复 | 从 iteration 2 / 128 条转移恢复到 iteration 4 / 192 条转移；成功恢复模型、优化器、replay、归一化和 RNG；新 episode 边界显式隔离 |
| Tianji/Wuji smoke | 4 环境、2 次迭代、128 条转移；4 次 timeout 均验证 terminal timer=16、返回的 reset state timer=0 |
| 六类工具 pilot | 16 环境、50 次迭代、8 步世界模型拟合、32 步 BPTT；14,848 条转移，包含 warmup；训练循环约 51.2 秒，不含 Isaac 启动和资产转换 |

六类工具 pilot 记录 14 个物理终止、13 个超时和 **0 个目标成功**。没有出现 NaN/Inf。末次 `world_loss=0.2542`、`actor_grad_norm=1.3808`；这些是训练诊断，不能解释为策略已收敛。该 pilot 在添加最终的计时断言、日程和代码整理前运行；运行目录保存了当时源代码哈希，最终模块另通过恢复、Tianji smoke 和核心测试验证。

## 真实仿真评估

- 短 episode 的恢复兼容性测试：2 环境、4 回合，4/4 完成评估，均为 16 步超时；目标成功数 0。
- 六类工具 pilot 的正常 episode 评估：4 环境、8 回合，8/8 完成评估，共执行 1,200 个向量步。原任务默认每目标最多 600 步、最多 50 个目标。
- pilot 平均回报 59.8141，平均达到目标数 0，全部目标成功率 0%。不把抬升奖励等同于目标完成。
- 两项都运行在 Isaac Lab / PhysX 中，未使用世界模型替代评估物理。

## 世界模型新数据检查

从 pilot checkpoint 在 4 个环境重新采集 32 步数据，没有优化器更新。记录目标切换条件下，32 个采样窗口的 normalized state MSE：

| rollout 长度 | MSE |
|---|---:|
| 1 | 0.07245 |
| 4 | 0.10680 |
| 8 | 0.16759 |

这是增强状态经固定训练归一化后的平均误差，不是米／弧度单位，不代表接触 Jacobian 的准确性。该小样本没有正例目标成功，低 switch BCE 不能证明已学会成功判定。验证使用记录的外生目标切换，不是完全自由预测目标切换的长期 rollout。

## 成功切换和 reset 的额外接口测试

`check_adapter.py` 在自己的新建 KUKA/Sharpa 环境中，把目标放到物体当前位姿并预置停留计数，验证一次真实 goal-hit 分支：成功数变为 1，目标发生随机切换，episode 继续。随后触发纯时间上限，验证 pre-reset timer=16、post-reset timer=0。这是有意构造的接口检查，**不是学到的策略成功**。

可重跑：

```bash
cd /home/pinhao/simtoolreal
source .envrc
OMNI_KIT_ACCEPT_EULA=YES .venv_isaacsim/bin/python -m supermimic_stl.check_adapter \
  --out outputs/supermimic_stl/adapter_check_repeat
```

## 远程证据目录

根目录：`/home/pinhao/simtoolreal/outputs/supermimic_stl/`

- `pilot_20261028/`：metrics、metadata、完整 checkpoint、actor 导出。
- `resume_verified_20261028/`：恢复后的结果和 checkpoint。
- `smoke_tianji_20261028/`：Tianji/Wuji smoke 结果。
- `eval_smoke_20261028/evaluation.json`：短 episode 接口评估。
- `eval_pilot_20261028/evaluation.json`：正常 episode 的 8 回合评估。
- `validate_pilot_20261028/world_validation.json`：新数据模型诊断。
- `adapter_check_20261028/adapter_check.json`：成功切换与 pre-reset 检查。
- `isolation_verification.json`：既有文件和 tracked diff 比对。

本地工作区 `experiments/supermimic_simtoolreal_20261028/` 保存代码副本、说明和 JSON 证据；大体积 checkpoint 只保留在远程运行目录。所有验证进程已结束，不留后台长期训练。
