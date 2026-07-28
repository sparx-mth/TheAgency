"""Turn ``metrics.jsonl`` into the figures that explain what the run did.

    python -m ...world_goal.plots --run ~/navdp_world_goal/run1

``train.py`` calls this automatically when it finishes, and it can be run on a
half-finished run at any time -- the log is append-only line-delimited JSON, so a
partial file plots fine.

Three figures, and the split between them is the point:

``training_curves``  every loss term, train against validation, on its own axes.
                     A single "total" curve going down tells you almost nothing
                     when it is a weighted sum of five things pulling in
                     different directions; the interesting failure -- clearance
                     improving while goal-reaching quietly rots -- is invisible
                     unless the terms are separated.
``training_health``  learning rate, gradient norm, throughput, GPU memory. This
                     is what you look at when the loss curve is strange.
``training_nav``     the *navigation* metrics measured on validation --
                     clearance, collision rate, and how far the trajectory ends
                     from the goal against how far the expert's ends. These are
                     metres and percentages rather than loss units, so they say
                     whether the flying got better, which the loss does not.

matplotlib only (it is in the ``navdp`` env; TensorBoard is not). Both PNG for
looking at and SVG for embedding in the HTML report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

TERMS = (("act", "diffusion epsilon-MSE"), ("waypoint", "waypoint L1 (m)"),
         ("clearance", "clearance hinge"), ("goal", "goal-distance match"),
         ("critic", "critic value MSE"))
TRAIN_COLOR = "#8a8f98"
VAL_COLOR = "#2a78d6"
ACCENT = "#eb6834"


def load_metrics(run_dir) -> Tuple[List[dict], List[dict]]:
    """Read ``metrics.jsonl`` into ``(train_records, val_records)``."""
    path = Path(run_dir).expanduser() / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no metrics.jsonl in {run_dir} -- has training run?")
    train, val = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        (val if record.get("phase") == "val" else train).append(record)
    return train, val


def series(records: Sequence[dict], key: str) -> Tuple[List[int], List[float]]:
    """``(steps, values)`` for one key, skipping records that lack it."""
    steps = [r["step"] for r in records if key in r]
    values = [float(r[key]) for r in records if key in r]
    return steps, values


def _panel(axis, title: str, ylabel: str = "") -> None:
    axis.set_title(title, fontsize=9)
    axis.set_xlabel("step", fontsize=8)
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=8)
    axis.tick_params(labelsize=7)
    axis.grid(alpha=0.25, linewidth=0.5)


def curves_figure(train: List[dict], val: List[dict]):
    """Total plus one panel per loss term, train against validation."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    flat = axes.ravel()

    steps, values = series(val, "total")
    if steps:
        flat[0].plot(steps, values, color=VAL_COLOR, linewidth=1.6, label="validation")
        best = min(range(len(values)), key=lambda i: values[i])
        flat[0].plot([steps[best]], [values[best]], "o", color=ACCENT, markersize=6,
                     label=f"best {values[best]:.4f} @ {steps[best]}")
        flat[0].legend(fontsize=7)
    _panel(flat[0], "total objective (validation)", "loss")

    for axis, (term, label) in zip(flat[1:], TERMS):
        tx, ty = series(train, f"train/raw/{term}")
        vx, vy = series(val, f"raw/{term}")
        if tx:
            axis.plot(tx, ty, color=TRAIN_COLOR, linewidth=1.0, label="train")
        if vx:
            axis.plot(vx, vy, color=VAL_COLOR, linewidth=1.6, label="validation")
        if tx or vx:
            axis.legend(fontsize=7)
        _panel(axis, label, term)
    figure.suptitle("NavDP world-goal fine-tune: loss terms", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    return figure


def health_figure(train: List[dict]):
    """Learning rate, gradient norm, throughput and memory."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for axis, (key, title, ylabel) in zip(axes, (
            ("lr_head", "learning rate (head)", "lr"),
            ("grad_norm", "gradient norm (pre-clip)", "L2"),
            ("samples_per_s", "throughput", "samples/s"),
            ("gpu_mem_gb", "peak GPU memory", "GB"))):
        steps, values = series(train, key)
        if steps:
            axis.plot(steps, values, color=VAL_COLOR, linewidth=1.3)
        _panel(axis, title, ylabel)
    figure.tight_layout()
    return figure


def navigation_figure(val: List[dict]):
    """The metres-and-percent view: is the flying actually getting better?"""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.4))

    steps, values = series(val, "metric/min_clear_m")
    if steps:
        axes[0].plot(steps, values, color=VAL_COLOR, linewidth=1.5)
    axes[0].axhline(0.0, color=ACCENT, linewidth=0.9, linestyle="--", label="wall")
    axes[0].legend(fontsize=7)
    _panel(axes[0], "mean worst-clearance of predictions", "metres")

    steps, values = series(val, "metric/collide_frac")
    if steps:
        axes[1].plot(steps, [100.0 * v for v in values], color=ACCENT, linewidth=1.5)
    _panel(axes[1], "predictions entering geometry", "% of samples")

    steps, values = series(val, "metric/goal_gap_m")
    expert_steps, expert = series(val, "metric/goal_gap_expert_m")
    if steps:
        axes[2].plot(steps, values, color=VAL_COLOR, linewidth=1.5, label="policy")
    if expert_steps:
        axes[2].plot(expert_steps, expert, color=TRAIN_COLOR, linewidth=1.2,
                     linestyle="--", label="expert (target)")
        axes[2].legend(fontsize=7)
    _panel(axes[2], "distance from goal at the horizon", "metres")

    figure.tight_layout()
    return figure


def render(run_dir, formats: Sequence[str] = ("png", "svg")) -> Dict[str, Path]:
    """Write every figure for a run. Returns ``{name: png path}``."""
    import matplotlib
    matplotlib.use("Agg")

    run_dir = Path(run_dir).expanduser()
    train, val = load_metrics(run_dir)
    figures = {"training_curves": curves_figure(train, val),
               "training_health": health_figure(train),
               "training_nav": navigation_figure(val)}
    written: Dict[str, Path] = {}
    for name, figure in figures.items():
        for extension in formats:
            path = run_dir / f"{name}.{extension}"
            figure.savefig(path, dpi=140, bbox_inches="tight")
            if extension == "png":
                written[name] = path
        figure.clf()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True, help="a training output directory")
    args = parser.parse_args()
    for name, path in render(args.run).items():
        print(f"[plots] {name} -> {path}", flush=True)


if __name__ == "__main__":
    main()
