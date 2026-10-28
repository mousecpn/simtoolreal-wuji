# SAPG architecture alignment validation

2026-10-28, pinhao RTX 3060 12 GB, isolated checkout `simtoolreal-sm-aligned`.

- Directly constructs actor and central critic with the vendored rl_games ModelBuilder and original SimToolRealSAPG.yaml.
- Compared every parameter and normalization-buffer shape with an actual vanilla SAPG checkpoint: actor 7,811,468 parameters; central critic 2,037,441. Both match. This counts the native actor model including its unused local value head.
- 17 unit tests passed, including native forward equivalence, differentiable frozen normalization, recurrent replay/reset/burn-in, CUDA hidden contiguity, exported recurrent policy, and actor/world gradient isolation.
- Actual Isaac Lab training with 12 physical environments, full 12,288 imagined trajectories, horizon 32, microbatch 512: finite losses, actor gradient norm 0.03568, one accumulated actor step and eight critic optimizer steps. Update/collection loop took 10.68 seconds; peak PyTorch GPU allocation 2,996,888,576 bytes (does not include every simulator allocation). This is a functional check, not a production-throughput benchmark.
- Previous aligned actor export completed 12 physical evaluation episodes. Zero goal successes after the smoke training; no performance claim.

Evidence under `/home/pinhao/simtoolreal/outputs/supermimic_stl/`:

- `aligned_full_final_20261028/result.json`
- `aligned_full_final_20261028/parameter_alignment.json`
- `aligned_eval_v4_20261028/evaluation.json`

Original SAPG/task source files are unchanged. All new implementation is isolated in `supermimic_stl/`. Legacy actor checkpoints remain supported.

Architecture and parameterization match, but the training algorithms still differ: imagined pathwise actor gradients versus SAPG PPO, imagined lambda targets, real-only RMS update timing, cosine versus adaptive-KL learning-rate schedules, FP32 versus native AMP, and off-policy recurrent checkpoints plus 16-step burn-in. The chosen real environment counts also differ (12,288 versus 24,576). Compare success against real samples and wall time as well as epochs.
