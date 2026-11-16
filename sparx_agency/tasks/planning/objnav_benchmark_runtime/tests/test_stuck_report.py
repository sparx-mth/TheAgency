"""The paired report of where the agent got stuck: read from two arms' lean rows, episode by episode."""
from __future__ import annotations

import json

import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.stuck_report import (
    BLIND_RUN_STEPS, closing, main, parse_arm, refused, render, report, sign_test_p, signature)


def row(episode_id, success=True, steps=100, forward=60, blocked=0, spl=0.5, dtg=0.05, stop=True, **counters):
    return {"episode_id": episode_id, "scene_id": episode_id.split("/")[0], "target_category": "chair",
            "success": success, "spl": spl if success else 0.0, "steps": steps, "stop_called": stop,
            "distance_to_goal_m": dtg, "path_length_m": 0.25 * forward,
            "action_counts": {"MOVE_FORWARD": forward, "TURN_LEFT": steps - forward - int(stop), "STOP": int(stop)},
            "agent_info": {"blocked_forward": blocked, "policy": {"target_closing": counters}}}


def arms():
    before = {
        "A/000001": row("A/000001", blocked=2, releases=1, squeezed_standoffs=1),
        "A/000002": row("A/000002", steps=180, forward=90, blocked=31, glimpse_steps=35),     # a blind-step run
        "B/000003": row("B/000003", success=False, steps=500, forward=300, blocked=12, dtg=4.0, stop=False),
        "B/000004": row("B/000004", blocked=1),
        "C/000009": row("C/000009"),                                                          # not in the other arm
    }
    after = {
        "A/000001": row("A/000001", blocked=2, releases=1, retry_standoffs=1),                # identical trajectory
        "A/000002": row("A/000002", steps=120, forward=70, blocked=3, glimpse_steps=2),
        "B/000003": row("B/000003", success=True, steps=240, forward=150, blocked=5, dtg=0.03,
                        stop_approaches=1, stop_approach_steps=2),                            # walked on to the stop distance
        "B/000004": row("B/000004", success=False, steps=130, forward=70, blocked=1, dtg=0.4,
                        stop_approaches=1, stop_approach_steps=1, stop_approach_shortfalls=1),
    }
    return before, after


def test_the_counters_are_read_under_either_name_and_absent_ones_are_zero():
    before, after = arms()
    assert closing(before["A/000001"], "retry_standoffs", "squeezed_standoffs") == 1.0, "the old name"
    assert closing(after["A/000001"], "retry_standoffs", "squeezed_standoffs") == 1.0, "the new one"
    assert closing(before["B/000003"], "stop_approaches") == 0.0, "a counter the arm's code did not have"
    assert closing({"agent_info": {}}, "releases") == 0.0 and closing({}, "releases") == 0.0
    assert refused(before["A/000002"]) == 31.0 and refused({}) == 0.0
    assert signature(before["A/000001"]) == signature(after["A/000001"])
    assert signature(before["A/000002"]) != signature(after["A/000002"])
    # The same steps, the same counts and the same path length, ended somewhere else: another trajectory.
    elsewhere = dict(before["A/000001"], distance_to_goal_m=float(before["A/000001"]["distance_to_goal_m"]) + 0.05)
    assert signature(elsewhere) != signature(before["A/000001"])


def test_the_report_pairs_the_episodes_and_says_what_moved():
    before, after = arms()
    data = report(before, after, resamples=200)
    assert data["episodes"] == 4 and data["only_before"] == 1 and data["only_after"] == 0
    assert data["identical"] == 1, "the one episode the change never acted in"
    lines = {line["name"]: line for line in data["lines"]}
    assert lines["success"]["before"] == 0.75 and lines["success"]["after"] == 0.75 and lines["success"]["diff"] == 0.0
    refused_line = lines["refused forward steps"]
    assert refused_line["before"] == pytest.approx(11.5) and refused_line["after"] == pytest.approx(2.75)
    assert refused_line["up"] == 0 and refused_line["down"] == 2 and refused_line["ci"][0] <= refused_line["diff"] <= refused_line["ci"][1]
    assert lines["closing: standoffs at the relaxed clearance"]["diff"] == 0.0, "one counter under two names"
    assert lines["closing: blind verification steps"]["before"] == 8.75 and lines["closing: blind verification steps"]["after"] == 0.5
    walks, steps, short = (lines[name] for name in ("closing: walks to the stop distance",
                                                    "closing:   forward steps of those walks",
                                                    "closing:   walks that ended short of it"))
    assert (walks["before"], walks["after"]) == (0.0, 0.5), "an arm whose code had no stop criterion reads 0"
    assert steps["after"] == 0.75 and short["after"] == 0.25
    assert lines["ended without a STOP"]["before"] == 0.25 and lines["ended without a STOP"]["after"] == 0.0
    assert data["lost"] == ["B/000004"] and data["won"] == ["B/000003"] and data["mcnemar_p"] == 1.0
    assert data["blind_runs"] == {"before": ["A/000002"], "after": []} and BLIND_RUN_STEPS == 20
    assert data["refused_total"] == {"before": 46.0, "after": 11.0}
    assert data["scenes"]["A"]["episodes"] == 2 and data["scenes"]["B"]["sr"] == [0.5, 0.5]
    with pytest.raises(ValueError, match="share no episode"):
        report({"A/000001": before["A/000001"]}, {"B/000003": after["B/000003"]})


def test_the_text_names_the_arms_the_flips_and_the_runs():
    before, after = arms()
    text = render(report(before, after, resamples=200), ("head", "fixed"))
    assert text.startswith("fixed against head: 4 paired episodes; identical trajectories in 1")
    assert "NOT PAIRED: 1 episodes only in head, 0 only in fixed" in text
    assert "success flips: 1 lost, 1 won (exact McNemar p = 1.000)" in text
    assert "to failure B/000004" in text and "to success B/000003" in text
    assert "blind verification-step runs (>= 20 such steps in an episode): 1 episodes head, 0 fixed" in text
    assert "refused forward steps in all: 46 head, 11 fixed" in text
    assert sign_test_p([0.0, 0.0]) == 1.0 and sign_test_p([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) == pytest.approx(2 / 64)
    assert parse_arm("before=runs/a,runs/b") == ("before", "runs/a,runs/b") and parse_arm("runs/cpu/after/") == ("after", "runs/cpu/after/")


def test_main_reads_the_roots_and_writes_the_numbers(tmp_path, capsys):
    before, after = arms()
    for name, rows in (("before", before), ("after", after)):
        for episode_id, item in rows.items():
            scene = tmp_path / name / episode_id.split("/")[0]
            scene.mkdir(parents=True, exist_ok=True)
            with (scene / "episodes.jsonl").open("a") as stream:
                stream.write(json.dumps(item) + "\n")
    out = tmp_path / "numbers.json"
    assert main(["head=%s" % (tmp_path / "before"), str(tmp_path / "after"), "--resamples", "100", "--json", str(out)]) == 0
    assert "after against head: 4 paired episodes" in capsys.readouterr().out
    assert json.loads(out.read_text())["arms"] == ["head", "after"]
