"""The storey report: same- against cross-storey episodes, paired arms, and the transitions per episode.

Two miniature campaign roots over the same four episodes (three on the start storey, one with its goal a storey
up), one arm with the multi-storey policy's own transition bookkeeping and a recorded trajectory that climbs. No
simulator, no GPU: the rows are written as ``mp3d.run`` leaves them.
"""
from __future__ import annotations

import csv
import json

import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d import storey_report
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.protocol import MP3D_V1
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_mp3d import SCENE, row, write_release

UP = (1.0, 3.5, 2.0)        # a start a storey above the release's goals (which stand at y = 0.5)


@pytest.fixture
def release(tmp_path):
    """Four episodes of one scene: three same-storey chairs, one bed started a storey above its goal."""
    return write_release(tmp_path, MP3D_V1, {SCENE: [row(), row(), row(), row("bed", position=UP)]})


def episode_row(episode_id, category, *, success, spl, dtg, steps=100, stop=True, arrivals=0, completed=0, lean=False):
    policy = {"floor_revisions": arrivals}
    if not lean:
        policy["building"] = {"events": [{"action": 10 * i, "event": "floor_arrival", "source": 0, "destination": 1}
                                         for i in range(arrivals)] + [{"action": 3, "event": "stairs_seen"}],
                              "completed_transitions": [{"phase": "done", "destination": 1}] * completed
                              + [{"phase": "deferred"}] * (1 if completed else 0)}
    return {"episode_id": episode_id, "scene_id": SCENE, "target_category": category, "success": success, "spl": spl,
            "soft_spl": spl, "distance_to_goal_m": dtg, "steps": steps, "stop_called": stop, "wall_s": 10.0,
            "agent_info": {"policy": policy}}


def write_arm(root, rows, trajectories=None):
    scene_dir = root / SCENE
    scene_dir.mkdir(parents=True)
    with (scene_dir / "episodes.jsonl").open("w") as stream:
        for r in rows:
            stream.write(json.dumps(r) + "\n")
    for episode_id, heights in (trajectories or {}).items():
        directory = scene_dir / "shard_0" / "recordings" / episode_id.replace("/", "_")
        directory.mkdir(parents=True)
        (directory / "episode.json").write_text(json.dumps({"episode_id": episode_id}))
        with (directory / "trajectory.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("step", "x", "y", "z"))
            writer.writeheader()
            for step, z in enumerate(heights):
                writer.writerow({"step": step, "x": 0.0, "y": 0.0, "z": z})
    return root


def test_the_storeys_are_read_from_the_published_goals_and_the_survey(release, tmp_path):
    ids = ["%s/%06d" % (SCENE, i) for i in range(4)]
    survey = tmp_path / "levels.csv"
    survey.write_text("episode,level_heights_m,start_level,goal_level\n"
                      + "\n".join("%s,0.5 3.5,%d,0" % (i, 1 if i.endswith("3") else 0) for i in ids) + "\n")
    facts = storey_report.storey_facts(ids, release.episodes_dir, survey, 0.5)
    assert [facts[i]["same_storey"] for i in ids] == [True, True, True, False]
    assert facts[ids[3]]["nearest_view_point_dz_m"] == 3.0 and facts[ids[3]]["survey_same_level"] is False
    assert facts[ids[3]]["start_level"] == 1 and facts[ids[3]]["goal_levels"] == [0]
    assert facts[ids[0]]["survey_same_level"] is True and facts[ids[0]]["levels_m"] == [0.5, 3.5]
    without = storey_report.storey_facts(ids, release.episodes_dir, None, 0.5)
    assert without[ids[3]]["survey_same_level"] is None and without[ids[3]]["goal_levels"] == []


def test_transitions_come_from_the_building_report_and_the_recorded_height(tmp_path):
    full = episode_row("s/000000", "bed", success=True, spl=0.5, dtg=0.1, arrivals=2, completed=1)
    assert storey_report.method_transitions(full) == {"floor_arrivals": 2, "completed_stair_transitions": 1,
                                                       "stair_transition_attempts": 2, "floor_revisions": 2}
    assert storey_report.method_transitions(episode_row("s/000001", "bed", success=False, spl=0.0, dtg=5.0, lean=True)) is None
    root = write_arm(tmp_path / "arm", [full], {"s/000000": [3.5, 3.4, 2.9, 2.0, 1.2, 0.6, 0.5, 0.5, 0.6, 1.5, 2.8, 3.4, 3.5, 0.5]})
    recording = root / SCENE / "shard_0" / "recordings" / "s_000000"
    facts = {"levels_m": [0.5, 3.5], "goal_levels": [0]}
    t = storey_report.trajectory_transitions(recording, facts)
    # 3.5 -> (stairs) -> 0.5 -> (stairs) -> 3.5 -> 0.5: three settled changes; the last height is the goal's level
    assert t["level_changes"] == 3 and t["final_level"] == 0 and t["ended_on_goal_level"] is True
    assert t["max_drop_m"] == 3.0 and t["max_rise_m"] == 0.0
    assert storey_report.trajectory_transitions(None, facts) is None
    assert storey_report.trajectory_transitions(recording, {"levels_m": [], "goal_levels": []}) is None


def test_two_arms_are_paired_by_storey_and_the_climbing_arm_counts_its_transitions(release, tmp_path):
    ids = ["%s/%06d" % (SCENE, i) for i in range(4)]
    survey = tmp_path / "levels.csv"
    survey.write_text("episode,level_heights_m,start_level,goal_level\n"
                      + "\n".join("%s,0.5 3.5,%d,0" % (i, 1 if i.endswith("3") else 0) for i in ids) + "\n")
    off = write_arm(tmp_path / "off", [
        episode_row(ids[0], "chair", success=True, spl=0.6, dtg=0.05),
        episode_row(ids[1], "chair", success=True, spl=0.4, dtg=0.08),
        episode_row(ids[2], "chair", success=False, spl=0.0, dtg=4.0, stop=False, steps=500),
        episode_row(ids[3], "bed", success=False, spl=0.0, dtg=9.0, steps=500, stop=False)],
        {ids[3]: [3.5] * 20})
    on = write_arm(tmp_path / "on", [
        episode_row(ids[0], "chair", success=True, spl=0.6, dtg=0.05),
        episode_row(ids[1], "chair", success=False, spl=0.0, dtg=2.0, steps=300),
        episode_row(ids[2], "chair", success=False, spl=0.0, dtg=4.0, stop=False, steps=500),
        episode_row(ids[3], "bed", success=True, spl=0.3, dtg=0.07, steps=250, arrivals=1, completed=1)],
        {ids[3]: [3.5, 3.4, 2.5, 1.4, 0.6, 0.5, 0.5]})
    data = storey_report.report([("off", str(off)), ("on", str(on))], episodes_dir=release.episodes_dir,
                                levels_csv=survey, resamples=200)
    assert data["episodes"] == 4 and data["group_sizes"] == {"all": 4, "same_storey": 3, "cross_storey": 1}
    assert data["per_arm"]["off"]["cross_storey"]["sr"] == 0.0 and data["per_arm"]["on"]["cross_storey"]["sr"] == 1.0
    assert data["per_arm"]["off"]["same_storey"]["successes"] == 2 and data["per_arm"]["on"]["same_storey"]["successes"] == 1
    pair = data["paired"]["on"]
    assert pair["all"]["to_success"] == [ids[3]] and pair["all"]["to_failure"] == [ids[1]]
    assert pair["cross_storey"]["sr_diff"] == 1.0 and pair["same_storey"]["sr_diff"] == pytest.approx(-1 / 3)
    assert pair["survey_other_level"]["n"] == 1 and pair["by_category"]["bed"]["to_success"] == [ids[3]]
    assert data["by_category"]["on"]["bed"]["cross_storey"]["n"] == 1 and data["by_category"]["on"]["chair"]["cross_storey"]["n"] == 0
    on_t = data["transitions"]["on"]["groups"]
    assert on_t["cross_storey"]["method"]["mean_floor_arrivals"] == 1.0
    assert on_t["cross_storey"]["method"]["completed_stair_transitions"] == 1
    assert on_t["cross_storey"]["trajectory"]["ended_on_goal_level"] == 1 and on_t["cross_storey"]["sr_with_change"] == (1, 1)
    assert on_t["all"]["method"]["arrivals_histogram"] == {0: 3, 1: 1}
    off_t = data["transitions"]["off"]["groups"]
    assert off_t["cross_storey"]["trajectory"]["mean_level_changes"] == 0.0 and off_t["cross_storey"]["sr_without_change"] == (0, 1)
    assert off_t["same_storey"]["trajectory"] is None            # no recording of a same-storey episode
    text = storey_report.render(data)
    assert "| on | 4 | 50.0 %" in text and "## Cross storey" in text and "## Storey transitions per episode" in text
    assert "| on | cross_storey | 1 | 1.00 | 1 | 0 / 1 / 0 / 0 | 1 (2) | 1.00 | 1 | 0 / 1 / 0 / 0 | 1 of 1 | 1 / 1 | 0 / 0 |" in text
    assert "on against off: 1 to success, 1 to failure" in text
    out = tmp_path / "out.json"
    assert storey_report.main(["off=%s" % off, "on=%s" % on, "--episodes-dir", str(release.episodes_dir), "--levels-csv", str(survey),
                               "--resamples", "100", "--json", str(out), "--markdown", str(tmp_path / "out.md")]) == 0
    assert "per_episode" not in json.loads(out.read_text())["transitions"]["on"] and (tmp_path / "out.md").read_text().startswith("Episodes")


def test_a_single_arm_reports_without_pairs_and_disjoint_arms_are_refused(release, tmp_path):
    ids = ["%s/%06d" % (SCENE, i) for i in range(4)]
    root = write_arm(tmp_path / "one", [episode_row(ids[0], "chair", success=True, spl=0.5, dtg=0.1)])
    data = storey_report.report([("one", str(root))], episodes_dir=release.episodes_dir, levels_csv=None, resamples=100)
    assert data["paired"] == {} and data["per_arm"]["one"]["all"]["n"] == 1
    assert "| one | 1 | 100.0 %" in storey_report.render(data)
    other = write_arm(tmp_path / "two", [episode_row(ids[1], "chair", success=True, spl=0.5, dtg=0.1)])
    with pytest.raises(ValueError, match="no episode"):
        storey_report.report([("one", str(root)), ("two", str(other))], episodes_dir=release.episodes_dir, levels_csv=None)
