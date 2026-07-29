"""Recording what happened, in a form that is still readable next month.

Two artefacts, both plain text:

``run.json``      everything needed to reproduce or explain the run -- the fully
                  resolved config, the git commit, package and driver versions,
                  the dataset fingerprint, parameter counts, the exact argv.
                  Written at the start so it exists even if the run is killed,
                  and updated at the end with the outcome.
``metrics.jsonl`` one JSON object per logged step: every loss term both raw and
                  weighted, both learning rates, gradient norm, throughput, GPU
                  memory, and the navigation metrics measured on validation.
                  Append-only and line-delimited, so a half-finished run still
                  plots and a running one can be tailed.

Nothing here depends on TensorBoard or wandb (neither is installed in the
``navdp`` env). :mod:`.plots` reads ``metrics.jsonl`` and produces the figures;
if TensorBoard ever is installed, ``--tensorboard`` mirrors the same records
into it without changing anything else.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional


def git_commit() -> str:
    """Short commit of the working tree, or a marker when unavailable."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5)
        suffix = "-dirty" if dirty.stdout.strip() else ""
        return (out.stdout.strip() or "unknown") + suffix
    except Exception:                                    # noqa: BLE001 - diagnostics only
        return "unknown"


def environment() -> Dict:
    """Versions that change results, captured so a rerun can be compared."""
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception:                                    # noqa: BLE001
        info["torch"] = "unavailable"
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except Exception:                                    # noqa: BLE001
        pass
    return info


class RunLogger:
    """Console table + JSONL metrics + a reproducibility record."""

    COLUMNS = ("step", "epoch", "lr", "train", "v_act", "v_wp", "v_clr", "v_goal",
               "v_crit", "val", "clear_m", "coll%", "note")
    WIDTHS = (7, 6, 9, 8, 7, 7, 7, 7, 7, 8, 8, 6, 22)

    def __init__(self, out_dir, config: Dict, extra: Optional[Dict] = None,
                 tensorboard: bool = False) -> None:
        self.out = Path(out_dir).expanduser()
        self.out.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.out / "metrics.jsonl"
        self.metrics = self.metrics_path.open("a", buffering=1)
        self.started = time.time()
        self.record = {
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "argv": sys.argv,
            "git": git_commit(),
            "environment": environment(),
            "config": config,
            **(extra or {}),
        }
        self._save_record()
        self.writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(str(self.out / "tb"))
            except Exception as error:                    # noqa: BLE001
                print(f"[log] tensorboard unavailable ({error}); JSONL only", flush=True)

    def _save_record(self) -> None:
        (self.out / "run.json").write_text(json.dumps(self.record, indent=2, default=str))

    # ------------------------------------------------------------------ output
    def note(self, message: str) -> None:
        """A free-text line, echoed to the console and kept in ``run.json``."""
        print(f"[log] {message}", flush=True)
        self.record.setdefault("notes", []).append(message)
        self._save_record()

    def header(self) -> None:
        line = "  ".join(name.rjust(width) for name, width in
                         zip(self.COLUMNS, self.WIDTHS))
        print(line, flush=True)
        print("-" * len(line), flush=True)

    def log(self, phase: str, step: int, epoch: float, values: Dict) -> None:
        """Append one metrics record. ``phase`` is ``train`` or ``val``."""
        entry = {"phase": phase, "step": int(step), "epoch": round(float(epoch), 4),
                 "wall_s": round(time.time() - self.started, 2)}
        entry.update({key: (round(float(value), 6) if isinstance(value, (int, float))
                            else value) for key, value in values.items()})
        self.metrics.write(json.dumps(entry) + "\n")
        if self.writer is not None:
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{phase}/{key}", float(value), step)

    def row(self, step: int, epoch: float, lr: float, train: float,
            val: Optional[Dict], note: str = "") -> None:
        """One aligned console line, matching :attr:`COLUMNS`."""
        if val is None:
            cells = [step, f"{epoch:.2f}", f"{lr:.2e}", f"{train:.4f}"] + ["-"] * 8 + [note]
        else:
            cells = [step, f"{epoch:.2f}", f"{lr:.2e}", f"{train:.4f}",
                     f"{val.get('raw/act', 0):.4f}", f"{val.get('raw/waypoint', 0):.4f}",
                     f"{val.get('raw/clearance', 0):.4f}", f"{val.get('raw/goal', 0):.4f}",
                     f"{val.get('raw/critic', 0):.4f}", f"{val.get('total', 0):.4f}",
                     f"{val.get('metric/min_clear_m', 0):.3f}",
                     f"{100 * val.get('metric/collide_frac', 0):.1f}", note]
        print("  ".join(str(c).rjust(w) for c, w in zip(cells, self.WIDTHS)), flush=True)

    def finish(self, summary: Dict) -> None:
        """Close the log and record the outcome."""
        self.record["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.record["duration_s"] = round(time.time() - self.started, 1)
        self.record["summary"] = summary
        self._save_record()
        self.metrics.close()
        if self.writer is not None:
            self.writer.close()
