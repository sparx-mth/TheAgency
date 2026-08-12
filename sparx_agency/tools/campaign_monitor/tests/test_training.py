"""Reading a run directory, including one that was resumed after an outage."""
import json
from pathlib import Path

from sparx_agency.tools.campaign_monitor import training


def _write_run(tmp_path: Path, notes=(), epochs: int = 8) -> Path:
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({
        "started": "2026-08-02T13:13:00",
        "config": {"optim": {"epochs": epochs}},
        "notes": list(notes),
    }))
    return run_dir


def _write_metrics(run_dir: Path, rows) -> None:
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n")


def test_rate_is_measured_over_the_window_not_the_whole_run(tmp_path):
    """A resumed run's step counter outruns its wall clock; the rate must not.

    ``step / wall_s`` divides steps accumulated before the outage by seconds
    counted only since the resume, which reported 11.4 steps/s for a machine
    doing 3.6 and an ETA a third of the truth.
    """
    run_dir = _write_run(tmp_path, notes=["resumed from best.pth at step 45000"])
    _write_metrics(run_dir, [
        {"phase": "train", "step": 45500, "wall_s": 100.0, "epoch": 4.4},
        {"phase": "train", "step": 66000, "wall_s": 5760.0, "epoch": 6.41},
    ])

    progress = training.read(run_dir)

    assert progress.step == 66000
    assert abs(progress.steps_per_second - (20500 / 5660.0)) < 1e-6


def test_whole_run_average_is_the_fallback_when_there_is_one_record(tmp_path):
    """A run with nothing to difference still gets a rate rather than none."""
    run_dir = _write_run(tmp_path)
    _write_metrics(run_dir, [{"phase": "train", "step": 1000, "wall_s": 250.0}])

    progress = training.read(run_dir)

    assert progress.steps_per_second == 4.0


def test_resume_point_is_read_from_the_notes(tmp_path):
    """So the elapsed time can be labelled as time *since* the resume."""
    run_dir = _write_run(tmp_path, notes=["resumed from best.pth at step 45,000"])

    assert training.read(run_dir).resumed_from_step == 45000


def test_a_fresh_run_reports_no_resume(tmp_path):
    assert training.read(_write_run(tmp_path)).resumed_from_step == 0


def test_a_missing_run_directory_is_not_reported_as_started(tmp_path):
    """The reading a mistyped --run produces, which the dashboard must flag."""
    progress = training.read(tmp_path / "nowhere")

    assert not progress.exists
    assert not progress.run_dir.is_dir()


def test_eta_uses_the_windowed_rate(tmp_path):
    run_dir = _write_run(tmp_path, notes=["82,360 optimiser steps"])
    _write_metrics(run_dir, [
        {"phase": "train", "step": 45500, "wall_s": 100.0},
        {"phase": "train", "step": 66000, "wall_s": 5760.0},
    ])

    progress = training.read(run_dir)

    assert progress.total_steps == 82360
    expected = (82360 - 66000) / (20500 / 5660.0)
    assert abs(progress.eta_seconds() - expected) < 1.0
