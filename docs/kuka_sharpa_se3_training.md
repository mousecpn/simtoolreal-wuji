# KUKA + Sharpa SE3 control

Task: `Isaacsimenvs-SimToolReal-SE3-Direct-v0`. Train a fresh policy with
28 actions: `[dx, dy, dz, rx, ry, rz, hand_0, ..., hand_21]`.
The 22 hand actions keep the original canonical ordering and absolute joint
position semantics. Joint-state and previous joint-target observations retain
29 entries; actor/critic observations remain 140/162 dimensions.

The first six normalized actions specify a relative palm-center pose target
in world axes, re-anchored at the measured pose every policy step (60 Hz).
Translation scales by 0.01 m per axis and rotation-vector increments by
0.05 rad per axis: `p_des = p + dp`, `R_des = Exp(rotvec) R`.
These are increments, not absolute world positions or Euler angles.

Rollout converts the increment to seven arm joint targets using a batched
PhysX geometric Jacobian and damped least-squares differential IK
(`lambda=0.05`). The Jacobian includes the palm-center offset used by the
observations. Commands are `q_measured + dq`, with `dq` limited to 0.1 rad
per joint per policy step, then clipped to joint limits. This is one local
IK update per step, not an iterative global pose solver. Arm joint smoothing
and joint velocity action scales belong to the legacy joint mode; SE3 uses
its Cartesian scales and IK limits. Hand smoothing still applies.

Submit:

```bash
sbatch scripts/train_kuka_sharpa_se3_nodr_fang.sbatch
```

The job first runs a GPU smoke test of nonzero actions, DLS numerics, reset,
command limits and finite observations/rewards. Training proceeds only if it
passes. Training uses SAPG, seed 42, 24,576 environments, six exploration
blocks of 4096 environments, 100,000 epochs and checkpoint saves every
10,000 epochs. One A6000 GPU on `fang`, 128 GB RAM, seven-day time limit.

No DR: disable all observation/action/object-state delays, pose and velocity
noise, scale-observation noise, external disturbances, friction variation,
robot/table reset perturbations and student camera/depth augmentation.
Object start pose is fixed at `(0, 0, 0.63, 1, 0, 0, 0)` (xyz + wxyz).
Tool asset variety and goal sampling remain the task distribution.

Logs: `logs/kuka-sharpa-se3-sapg-nodr-<JOBID>.log`.
Outputs: `outputs/slurm-<JOBID>-kuka-sharpa-se3-sapg-nodr/`.
Initial submission: **441317**.
