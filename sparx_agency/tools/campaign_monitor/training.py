"""How far a fine-tune has got, read off ``metrics.jsonl``.

:class:`world_goal.logger.RunLogger` writes ``run.json`` before the first step
and appends one JSON object per logged step to ``metrics.jsonl`` with line
buffering, so both files are readable while the run is still going. That is the
whole contract this module depends on — it never imports torch and never
touches the training process.

Progress is measured in optimiser steps. The total is not written anywhere, so
it is reconstructed from the resolved config in ``run.json``: epochs times the
steps per epoch, which the trainer notes once it knows the dataset size.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_STEPS_PER_EPOCH = re.compile(r"(\d[\d,]*)\s+steps?\s+per\s+epoch", re.IGNORECASE)
_TOTAL_STEPS = re.compile(r"(\d[\d,]*)\s+(?:optimiser|optimizer|total)\s+steps", re.IGNORECASE)


@dataclass
class TrainingProgress:
    """Everything the dashboard shows about a running fine-tune."""

    run_dir: Path
    exists: bool = False
    step: int = 0
    total_steps: Optional[int] = None
    epoch: float = 0.0
    total_epochs: Optional[int] = None
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    best_val_loss: Optional[float] = None
    min_clear_m: Optional[float] = None
    collide_pct: Optional[float] = None
    wall_s: float = 0.0
    started: Optional[str] = None
    finished: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)

    @property
    def fraction(self) -> Optional[float]:
        """Completed share of the run, or ``None`` before the total is known."""
        if not self.total_steps:
            return None
        return min(1.0, self.step / self.total_steps)

    @property
    def steps_per_second(self) -> Optional[float]:
        """Average optimiser steps per wall-clock second so far."""
        if self.wall_s <= 0 or self.step <= 0:
            return None
        return self.step / self.wall_s

    def eta_seconds(self) -> Optional[float]:
        """Seconds of training left at the average rate."""
        rate = self.steps_per_second
        if not rate or not self.total_steps:
            return None
        remaining = self.total_steps - self.step
        return remaining / rate if remaining > 0 else 0.0


def _tail_records(path: Path, limit: int = 400) -> List[Dict]:
    """The last ``limit`` well-formed records of a JSONL file.

    The final line of a file being appended to may be half-written, and one
    truncated line must not cost the whole reading; malformed lines are skipped.
    """
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return []
    records = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records


def _total_steps_from(record: Dict, notes: List[str]) -> Optional[int]:
    """Total optimiser steps, from the run record or the notes the trainer left.

    ``--max-steps`` wins when it was given; otherwise the trainer's own note is
    the most reliable source, since only it knows how many samples survived
    admission filtering. The epochs-times-steps-per-epoch product is the
    fallback.
    """
    config = record.get("config") or {}
    optim = config.get("optim") or {}
    for key in ("max_steps", "total_steps"):
        for holder in (record, config, optim):
            value = holder.get(key)
            if value:
                return int(value)

    joined = " ".join(notes)
    total = _TOTAL_STEPS.search(joined)
    if total:
        return int(total.group(1).replace(",", ""))

    per_epoch = _STEPS_PER_EPOCH.search(joined)
    epochs = optim.get("epochs")
    if per_epoch and epochs:
        return int(per_epoch.group(1).replace(",", "")) * int(epochs)
    return None


def read(run_dir: Path) -> TrainingProgress:
    """Read a run directory, tolerating one that has not started yet."""
    progress = TrainingProgress(run_dir=run_dir)
    record_path = run_dir / "run.json"
    if not record_path.is_file():
        return progress
    progress.exists = True

    try:
        record = json.loads(record_path.read_text())
    except (OSError, ValueError):
        return progress

    progress.started = record.get("started")
    progress.finished = record.get("finished")
    progress.notes = list(record.get("notes") or [])
    config = record.get("config") or {}
    progress.total_epochs = (config.get("optim") or {}).get("epochs")
    progress.total_steps = _total_steps_from(record, progress.notes)

    for name in ("best.pth", "last.pth", "milestone_25.pth", "milestone_50.pth",
                 "milestone_75.pth", "navdp-world-goal.ckpt"):
        if (run_dir / name).is_file():
            progress.checkpoints.append(name)

    for entry in _tail_records(run_dir / "metrics.jsonl"):
        progress.step = max(progress.step, int(entry.get("step", 0)))
        progress.epoch = max(progress.epoch, float(entry.get("epoch", 0.0)))
        progress.wall_s = max(progress.wall_s, float(entry.get("wall_s", 0.0)))
        if entry.get("phase") == "train":
            progress.train_loss = entry.get("total", progress.train_loss)
        elif entry.get("phase") == "val":
            progress.val_loss = entry.get("total", progress.val_loss)
            progress.min_clear_m = entry.get("metric/min_clear_m", progress.min_clear_m)
            collide = entry.get("metric/collide_frac")
            if collide is not None:
                progress.collide_pct = 100.0 * float(collide)
            if progress.val_loss is not None:
                progress.best_val_loss = (progress.val_loss if progress.best_val_loss is None
                                          else min(progress.best_val_loss, progress.val_loss))
    return progress


def is_running(run_dir: Path, stale_after_s: float = 300.0) -> bool:
    """Whether the run looks live, judged by how recently metrics were appended."""
    metrics = run_dir / "metrics.jsonl"
    if not metrics.is_file():
        return False
    try:
        return time.time() - metrics.stat().st_mtime < stale_after_s
    except OSError:
        return False


@dataclass
class StageProgress:
    """One offline pipeline stage, and whether its output exists yet."""

    name: str
    output: Path
    done: bool
    detail: str = ""


def pipeline_stages(out_dir: Path, dataset: Path, run: str) -> List[StageProgress]:
    """The six offline stages, in order, with the artefact each one produces.

    ``run_pipeline.sh`` skips a stage whose output is already present, so the
    existence of these files is exactly what decides whether work is repeated.
    """
    run_dir = out_dir / run
    features = dataset.with_name(dataset.name + "_features")
    checks = [
        ("dataset", dataset / "index.json"),
        ("features", features / "meta.json"),
        ("train", run_dir / "best.pth"),
        ("evaluate", run_dir / "evaluation.json"),
        ("export", run_dir / "navdp-world-goal.ckpt"),
        ("report", run_dir / "report.html"),
    ]
    stages = []
    for name, output in checks:
        done = output.exists()
        detail = ""
        if done:
            try:
                size = output.stat().st_size
                detail = f"{size / 1e6:.1f} MB" if size > 1e6 else f"{size / 1e3:.0f} kB"
            except OSError:
                detail = ""
        stages.append(StageProgress(name=name, output=output, done=done, detail=detail))
    return stages
