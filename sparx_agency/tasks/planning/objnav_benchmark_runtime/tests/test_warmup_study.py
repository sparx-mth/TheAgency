"""The warm-up study's subset and table (``mp3d/warmup_study.py``, 2026-11-11)."""
from __future__ import annotations

import json

from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d import warmup_study as study


def row(episode_id, success, spl, steps, warm_actions, areas, phase_warm=10, first_seen=None, moves=0):
    curve = [{"action": k, "observed_m2": areas[min(k, len(areas) - 1)], "phase": "warmup" if k < phase_warm else "frontier"}
             for k in range(steps)]
    vantage = {"stats": {"moves": moves}} if moves else None
    return {"episode_id": episode_id, "success": success, "spl": spl, "soft_spl": spl, "steps": steps,
            "distance_to_goal_m": 0.0 if success else 3.0, "stop_called": success, "target_category": "chair",
            "agent_info": {"policy": {"warmup": {"actions": warm_actions, "mode": "vantage" if moves else "spin",
                                                 "vantage": vantage},
                                      "exploration_metrics": {"coverage_curve": curve},
                                      "perception": {"target_first_seen_step": first_seen}}}}


def test_the_subset_is_drawn_round_robin_over_the_scenes_and_is_reproducible():
    ids = ["A/%06d" % k for k in range(10)] + ["B/%06d" % k for k in range(3)] + ["C/%06d" % k for k in range(6)]
    picked = study.select_subset(ids, 9, seed=17)
    assert len(picked) == 9 and picked == sorted(picked) and len(set(picked)) == 9
    by_scene = {s: sum(1 for k in picked if k.startswith(s)) for s in "ABC"}
    assert by_scene == {"A": 3, "B": 3, "C": 3}, "nine over three scenes: three each, whatever their sizes"
    assert study.select_subset(ids, 9, seed=17) == picked
    assert study.select_subset(ids, 9, seed=18) != picked
    assert len(study.select_subset(ids, 50, seed=1)) == 19, "never more than there are"
    small = study.select_subset(ids, 12, seed=3)
    assert sum(1 for k in small if k.startswith("B")) == 3, "a scene with three gives three, the rest fill up"


def test_the_warmup_measures_read_the_record_and_the_table_summarises_them():
    rows = {"A/1": row("A/1", True, 0.5, 60, 10, [1.0 + k for k in range(60)], first_seen=12),
            "A/2": row("A/2", False, 0.0, 500, 14, [2.0 + 0.5 * k for k in range(500)], phase_warm=14, moves=1)}
    m = study.warmup_measures(rows["A/1"])
    assert m["warmup_actions"] == 10 and m["moves"] == 0 and m["area_end_m2"] == 10.0 and m["first_seen"] == 12
    assert m["area_10_m2"] == 11.0 and m["area_50_m2"] == 51.0 and m["warmup_mode"] == "spin"
    m2 = study.warmup_measures(rows["A/2"])
    assert m2["moves"] == 1 and m2["warmup_mode"] == "vantage" and m2["area_end_m2"] == 2.0 + 0.5 * 13
    assert study.coverage_at([], 10) is None and study.coverage_at([{"action": 3, "observed_m2": 4.0}], 10) == 4.0
    measures = study.episode_measures(rows, "nowhere", ["A/1", "A/2"], recordings=False)
    s = study.summarise(measures, ["A/1", "A/2"])
    assert s["warmup_actions"] == 12.0 and s["moves"] == 0.5 and s["first_seen_median"] == 12
    assert s["first_seen_within"] == 0.5 and s["never_seen"] == 0.5
    assert s["area_end_m2"] == (10.0 + 8.5) / 2


def test_the_report_pairs_every_arm_with_the_first_over_the_set_and_renders(tmp_path):
    base = {"A/1": row("A/1", True, 0.5, 60, 10, [1.0] * 60), "A/2": row("A/2", False, 0.0, 500, 10, [1.0] * 500),
            "A/3": row("A/3", True, 0.4, 80, 10, [1.0] * 80)}
    arm = {"A/1": row("A/1", True, 0.6, 50, 12, [2.0] * 50, moves=1), "A/2": row("A/2", True, 0.3, 200, 15, [2.0] * 200, moves=1)}
    for name, rows in (("spin", base), ("arm", arm)):
        d = tmp_path / name / "scene"
        d.mkdir(parents=True)
        with open(d / "episodes.jsonl", "w") as fh:
            for r in rows.values():
                fh.write(json.dumps(r) + "\n")
    data = study.report([("spin", str(tmp_path / "spin")), ("arm", str(tmp_path / "arm"))], ["A/1", "A/2", "A/3"],
                        recordings=False, resamples=200)
    assert data["episodes"] == 2, "the episode the arm did not run is left out of every number"
    assert data["per_arm"]["spin"]["sr"] == 0.5 and data["per_arm"]["arm"]["sr"] == 1.0
    assert data["paired"]["arm"]["to_success"] == ["A/2"] and data["paired"]["arm"]["to_failure"] == []
    assert data["warmup"]["arm"]["moves"] == 1.0 and data["warmup"]["spin"]["moves"] == 0.0
    text = study.render(data)
    assert "| arm | 2 | 100.0 %" in text and "The warm-up itself" in text and "area @10" in text
