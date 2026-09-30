"""The multi-story campaign report surfaces SR, SPL and DTG, per episode and overall."""
from __future__ import annotations

import hashlib
import json

import pytest

from sparx_agency.tasks.planning.objnav_benchmark.records import ACTION_NAMES, EpisodeRecord
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.multifloor_dataset import MULTIFLOOR_SCHEMA
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.multifloor_report import write_multifloor_report

SCENE = "Ranchester"
SOURCE = "a" * 64


def _record(index, *, success, spl, dtg, steps):
    return EpisodeRecord(
        benchmark="gibson", split="multistory-development", episode_id="%s/%06d" % (SCENE, index),
        scene_id=SCENE, target_category="toilet", agent="rpt-frontier", success=success, spl=spl,
        soft_spl=spl if success else 0.25, distance_to_goal_m=dtg, path_length_m=12.5,
        observed_path_length_m=12.5, shortest_path_m=8.0, start_distance_to_goal_m=8.0, steps=steps,
        stop_called=success, termination="stop" if success else "step_limit",
        action_counts={name: (steps if name == "MOVE_FORWARD" else 0) for name in ACTION_NAMES}, wall_s=60.0,
        agent_info={"policy": {"building": {"atlas": {"floors": [1, 2], "connections": [1]}}}})


def _campaign(root, records):
    run = root / "frontier" / SCENE
    run.mkdir(parents=True)
    (root / "campaign.json").write_text(json.dumps(
        {"jobs": [["frontier", SCENE]], "source_sha256": SOURCE, "recordings_expected": len(records)}))
    (run / "run.json").write_text(json.dumps(
        {"config": {"source_sha256": SOURCE, "recording": {"episode_ids": [r.episode_id for r in records]}}}))
    (run / "episodes.jsonl").write_text("".join(json.dumps(r.to_row()) + "\n" for r in records))
    (run / "evaluation_diagnostics.jsonl").write_text("".join(
        json.dumps({"episode_id": r.episode_id, "first_cross_floor_action": 40}) + "\n" for r in records))
    for record in records:
        video = run / "recordings" / hashlib.sha256(record.episode_id.encode()).hexdigest()[:12] / "video.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"\0")
    return {"schema": MULTIFLOOR_SCHEMA, "scenes": [SCENE],
            "episodes": [{"scene": SCENE} for _ in records],
            "generation": {"semantic_limit": "reference floor only"}}


def test_report_carries_dtg_beside_sr_and_spl(tmp_path):
    records = [_record(0, success=True, spl=0.8, dtg=0.4, steps=120),
               _record(1, success=False, spl=0.0, dtg=None, steps=500)]
    report = write_multifloor_report(tmp_path, _campaign(tmp_path, records))

    overall = report["statistics"]["frontier"]["overall"]
    assert overall["success_rate"] == pytest.approx(0.5)
    assert overall["spl"] == pytest.approx(0.4)
    assert overall["distance_to_goal_m"] == pytest.approx(0.4)
    assert overall["n_unreachable_end"] == 1
    assert [row["distance_to_goal_m"] for row in report["episodes"]] == [0.4, None]

    text = (tmp_path / "RESULTS.md").read_text()
    assert "| Explorer | Episodes | SR | SR 95% CI | SPL | SPL 95% CI | DTG (m) |" in text
    assert "| frontier | 2 | 0.500 |" in text and "| 0.40 (1 unreachable end) |" in text
    assert "| frontier | Ranchester/000000 | toilet | True | 0.8000 | 0.40 | 120 |" in text
    assert "| frontier | Ranchester/000001 | toilet | False | 0.0000 | n/a | 500 |" in text
    assert "DTG 0.40 m" in (tmp_path / "index.html").read_text()



