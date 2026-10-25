#!/usr/bin/env python3
"""Plot TensorBoard rewards for one or more Slurm training jobs.

Example:
    .venv_isaacsim/bin/python scripts/print_reward_curve.py 441317 532210 530967
"""

from __future__ import annotations

import argparse
import csv
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Isaac Sim's TensorBoard install may not provide this optional module.
sys.modules.setdefault("tensorboard.compat.notf", types.ModuleType("tensorboard.compat.notf"))
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = ["#2563eb", "#ea580c", "#059669", "#7c3aed", "#db2777", "#0891b2"]


def find_summaries(outputs: Path, job: str) -> Path:
    matches = sorted(outputs.glob(f"slurm-{job}*/**/summaries"))
    if not matches:
        raise FileNotFoundError(f"job {job}: no TensorBoard summaries under {outputs}")
    if len(matches) > 1:
        raise RuntimeError(f"job {job}: multiple summaries directories found: {matches}")
    return matches[0]


def read_curve(outputs: Path, job: str):
    summaries = find_summaries(outputs, job)
    accumulator = EventAccumulator(str(summaries), size_guidance={"scalars": 0})
    accumulator.Reload()
    if "rewards/iter" not in accumulator.Tags().get("scalars", []):
        raise RuntimeError(f"job {job}: scalar rewards/iter was not found")
    values = accumulator.Scalars("rewards/iter")
    if not values:
        raise RuntimeError(f"job {job}: rewards/iter is empty")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", nargs="+", help="Slurm job IDs, for example 441317 532210 530967")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Repository root")
    parser.add_argument("--output", type=Path, default=None, help="PNG/PDF prefix; default outputs/reward-curves/reward_curve_<jobs>")
    parser.add_argument("--labels", nargs="*", help="Optional legend labels, one per job")
    parser.add_argument("--ema", type=float, default=0.95, help="EMA weight for the previous smoothed value (default: 0.95)")
    args = parser.parse_args()
    if not 0.0 <= args.ema < 1.0:
        parser.error("--ema must be in [0, 1)")
    if args.labels and len(args.labels) != len(args.jobs):
        parser.error("--labels must contain exactly one label per job")

    output_dir = args.root / "outputs" / "reward-curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = args.labels or args.jobs
    with ThreadPoolExecutor(max_workers=len(args.jobs)) as executor:
        curves = list(executor.map(lambda job: read_curve(args.root / "outputs", job), args.jobs))

    for job, values in zip(args.jobs, curves):
        csv_path = output_dir / f"{job}_reward_iter.csv"
        with csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["environment_steps", "reward"])
            writer.writerows((value.step, value.value) for value in values)
        latest = values[-1]
        peak = max(values, key=lambda value: value.value)
        print(
            f"{job}: points={len(values)}, latest_step={latest.step}, "
            f"latest_reward={latest.value:.2f}, peak_reward={peak.value:.2f}, "
            f"peak_step={peak.step}",
            flush=True,
        )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for index, (job, label, values) in enumerate(zip(args.jobs, labels, curves)):
        x = [value.step / 1e9 for value in values]
        y = [value.value for value in values]
        smooth = []
        for value in y:
            smooth.append(value if not smooth else args.ema * smooth[-1] + (1.0 - args.ema) * value)
        color = COLORS[index % len(COLORS)]
        ax.plot(x, y, color=color, alpha=0.18, linewidth=0.6)
        ax.plot(x, smooth, color=color, linewidth=1.7, label=f"{job}: {label} (EMA {args.ema:g})")
    ax.set(
        xlabel="Environment steps (billions)",
        ylabel="Reward (rewards/iter)",
        title="Training reward: " + " vs ".join(args.jobs),
        xlim=(0, None),
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()

    if args.output is None:
        prefix = output_dir / ("reward_curve_" + "_".join(args.jobs))
    else:
        prefix = args.output.with_suffix("")
        prefix.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(prefix.with_suffix(f".{extension}"), dpi=180)
    print(f"PNG: {prefix}.png")
    print(f"PDF: {prefix}.pdf")


if __name__ == "__main__":
    main()
