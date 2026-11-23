"""The MP3D two-stage campaign's tooling: the committable results summary and the jobs governor."""
from __future__ import annotations

import json

from sparx_agency.core.planning.objnav.types.measurement import TERMINATION_STEP_LIMIT, TERMINATION_STOP
from sparx_agency.tasks.planning.objnav_benchmark.records import ACTION_NAMES, EpisodeRecord
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.jobs_governor import decide
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.protocol import MP3D_V1
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.results_summary import write

DIGEST = "2d025f01" + "0" * 56


def _record(episode_id, scene, category, success):
    counts = dict.fromkeys(ACTION_NAMES, 0)
    counts.update(MOVE_FORWARD=20, STOP=1 if success else 0)
    return EpisodeRecord(
        benchmark=MP3D_V1.benchmark, split=MP3D_V1.split, episode_id=episode_id, scene_id=scene,
        target_category=category, agent="rpt", success=success, spl=0.5 if success else 0.0, soft_spl=0.4,
        distance_to_goal_m=0.05 if success else 3.0, path_length_m=8.0, observed_path_length_m=8.0,
        shortest_path_m=4.0, start_distance_to_goal_m=4.0, steps=sum(counts.values()), stop_called=success,
        termination=TERMINATION_STOP if success else TERMINATION_STEP_LIMIT, action_counts=counts, wall_s=1.0)


def _campaign_root(tmp_path, method=None):
    root = tmp_path / "stage1_qwen2_5_3b-instruct"
    rows = {"sceneA": [_record("sceneA/0", "sceneA", "chair", True), _record("sceneA/1", "sceneA", "bed", False)],
            "sceneB": [_record("sceneB/0", "sceneB", "chair", True), _record("sceneB/1", "sceneB", "chair", False)]}
    run = {"config": {"protocol_id": "mp3d-objectnav-v1-val/2", "seed": 17, "perception": "ground-truth",
                      "method": {"llm": {"model": "qwen2.5:3b-instruct", "reasoning_model": "qwen2.5:3b-instruct"},
                                 "perception": "ground-truth",
                                 "node_oracle": {"prompt": {"version": "facts-qwen2.5-3b-v1", "sha256": DIGEST,
                                                            "tuned_on": "qwen2.5:3b-instruct"},
                                                 "model": {"name": "qwen2.5:3b-instruct", "prompt_tuned_on_it": True}}}}}
    if method is not None:
        run["config"]["method"] = method
    for scene, records in rows.items():
        (root / scene).mkdir(parents=True)
        (root / scene / "episodes.jsonl").write_text("".join(json.dumps(r.to_row()) + "\n" for r in records))
        (root / scene / "run.json").write_text(json.dumps(run))
    (root / "source-commit.txt").write_text("a" * 40 + "\n")
    (root / "source-changes.diff").write_text("")
    (root / "started_utc.txt").write_text("2026-11-20T10:00:00Z\n")      # a resume's launch, not the stage's start
    (root / "finished_utc.txt").write_text("2026-11-20T12:00:00Z\n")
    (root / "campaign_summary.json").write_text(json.dumps({"complete": True, "episodes_published": 2195}))
    return root


def test_the_summary_states_metrics_per_category_and_the_runs_provenance(tmp_path):
    out = tmp_path / "results"
    data = write(_campaign_root(tmp_path), out)
    assert data["title"].endswith("qwen2.5:3b-instruct (oracle) / qwen2.5:3b-instruct (labels), ground-truth perception")
    overall = data["summary"]["overall"]
    assert overall["n_episodes"] == 4 and overall["success_rate"] == 0.5 and overall["spl"] == 0.25
    by_category = {g["key"]: g for g in data["summary"]["by_category"]}
    assert by_category["chair"]["n_episodes"] == 3 and by_category["bed"]["success_rate"] == 0.0
    prov = data["provenance"]
    assert prov["node_oracle"]["prompt"]["sha256"] == DIGEST and prov["llm"]["oracle"] == "qwen2.5:3b-instruct"
    assert prov["source_commit"] == "a" * 40 and prov["source_tree_clean"]
    text = (out / "summary.md").read_text()
    assert "facts-qwen2.5-3b-v1" in text and "2d025f01" in text and "| chair | 3 |" in text
    assert sorted(p.name for p in out.iterdir()) == ["campaign_summary.json", "episodes.csv", "run.json",
                                                     "summary.json", "summary.md"]


def test_default_frontier_report_needs_no_room_oracle_or_labeller(tmp_path):
    method = {"pipeline_variant": "rpt", "perception": "ground-truth",
              "frontier_search": {"model": "qwen2.5vl:7b", "prompt": "boxed-frontier-v2"},
              "llm_identity": {"digest": DIGEST}}
    out = tmp_path / "results"
    data = write(_campaign_root(tmp_path, method=method), out)
    expected = "MP3D mp3d-objectnav-v1-val/2: qwen2.5vl:7b (frontier VLM), ground-truth perception"
    assert data["title"] == expected
    assert (out / "summary.md").read_text().splitlines()[0] == "# " + expected


def test_the_callers_first_start_wins_over_the_launchers_last_launch(tmp_path):
    data = write(_campaign_root(tmp_path), tmp_path / "out", "t",
                 started_utc="2026-11-19T08:00:00Z", finished_utc="2026-11-20T12:00:00Z")
    assert data["title"] == "t"
    prov = data["provenance"]
    assert prov["started_utc"] == "2026-11-19T08:00:00Z" and prov["duration_h"] == 28.0
    assert prov["run_root"].startswith("/")


def test_the_governor_lowers_on_any_mean_over_the_limit_and_never_below_the_floor():
    calm = {"cpu_pct": 40.0, "ram_pct": 50.0, "vram_pct": 80.0, "gpu_pct": 30.0}
    assert decide(10, 16, 1, dict(calm, ram_pct=91.0), 90.0, 0.0, 1800.0) == 9
    assert decide(10, 16, 1, dict(calm, vram_pct=90.5), 90.0, 9999.0, 1800.0) == 9, "over the limit wins over calm"
    assert decide(1, 16, 1, dict(calm, cpu_pct=99.0), 90.0, 0.0, 1800.0) == 1
    assert decide(10, 16, 1, dict(calm, vram_pct=90.0), 90.0, 0.0, 1800.0) == 10, "at the limit is not over it"


def test_the_governor_raises_only_after_a_calm_spell_and_never_past_the_ceiling():
    calm = {"cpu_pct": 40.0, "ram_pct": 50.0, "vram_pct": 70.0, "gpu_pct": 30.0}
    assert decide(10, 16, 1, calm, 90.0, 1799.0, 1800.0) == 10
    assert decide(10, 16, 1, calm, 90.0, 1800.0, 1800.0) == 11
    assert decide(16, 16, 1, calm, 90.0, 99999.0, 1800.0) == 16
