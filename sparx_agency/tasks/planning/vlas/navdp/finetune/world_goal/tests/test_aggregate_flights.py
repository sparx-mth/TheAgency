"""Joining one-mission-per-process flight results into an arm summary."""
import json

from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal import aggregate_flights


def _write(arm_dir, name, entries):
    arm_dir.mkdir(parents=True, exist_ok=True)
    (arm_dir / name).write_text(json.dumps(entries))


def _result(mission, reached=True, collided=False, clear=1.5):
    return {"mission": mission, "reached": reached, "collided": collided,
            "min_clear_m": clear, "path_len_m": 10.0, "duration_s": 30.0,
            "goal_error_m": 1.49}


def test_one_file_per_mission_is_joined_in_mission_order(tmp_path):
    """The shape a real comparison produces: twenty sessions, twenty files."""
    arm = tmp_path / "trained"
    for index in (2, 0, 1):
        _write(arm, f"results_{index:02d}.json", [_result(index)])

    results = aggregate_flights.load_results(arm)

    assert [r["mission"] for r in results] == [0, 1, 2]


def test_an_all_in_one_session_still_summarises(tmp_path):
    """--mission-index is optional, so results.json has to keep working."""
    arm = tmp_path / "baseline"
    _write(arm, "results.json", [_result(0), _result(1)])

    assert len(aggregate_flights.load_results(arm)) == 2


def test_a_reflown_mission_keeps_the_later_result(tmp_path):
    """A mission re-flown after a PX4 failure must not be counted twice."""
    arm = tmp_path / "trained"
    _write(arm, "results.json", [_result(0, reached=False)])
    _write(arm, "results_00.json", [_result(0, reached=True)])

    results = aggregate_flights.load_results(arm)

    assert len(results) == 1
    assert results[0]["reached"]


def test_summary_counts_reached_and_collisions(tmp_path):
    results = [_result(0, reached=True, collided=True, clear=-0.6),
               _result(1, reached=False, collided=False, clear=0.8),
               _result(2, reached=True, collided=False, clear=1.6)]

    summary = aggregate_flights.summarise("trained", results)

    assert summary["missions"] == 3
    assert summary["reached"] == 2
    assert summary["collisions"] == 1
    assert abs(summary["min_clear_m"] - (-0.6 + 0.8 + 1.6) / 3) < 1e-9


def test_a_corrupt_file_does_not_lose_the_others(tmp_path):
    """A session killed mid-write must not cost the nineteen that finished."""
    arm = tmp_path / "baseline"
    _write(arm, "results_00.json", [_result(0)])
    (arm / "results_01.json").write_text("{ truncated")

    assert len(aggregate_flights.load_results(arm)) == 1


def test_aggregate_writes_one_summary_per_arm(tmp_path):
    _write(tmp_path / "baseline", "results_00.json", [_result(0)])
    _write(tmp_path / "trained", "results_00.json", [_result(0)])

    summaries = aggregate_flights.aggregate(tmp_path)

    assert set(summaries) == {"baseline", "trained"}
    for arm in ("baseline", "trained"):
        stored = json.loads((tmp_path / arm / "summary.json").read_text())
        assert stored["arm"] == arm


def test_an_arm_that_never_flew_is_skipped_not_reported_as_perfect(tmp_path):
    """An empty directory must not become '0/0 reached, 0 collisions'."""
    (tmp_path / "trained").mkdir()

    assert aggregate_flights.aggregate(tmp_path) == {}
