"""The results directory is where a correct run can still be corrupted; these pin every refusal.

A run appends to one file for hours and is resumed after crashes, so the
failures that matter are the quiet ones: two experiments in one file, an
episode counted twice, a truncated line skipped instead of reported. Who may
write the directory, and when a resume enters run.json, are pinned in
``test_logger_lifecycle.py``.
"""
from __future__ import annotations

import json
import re
import sys

import pytest

from sparx_agency.core.planning.objnav.types.measurement import TERMINATION_STOP
from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.errors import (
    HarnessError,
    ResultsError,
)
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark.records import (
    ACTION_NAMES,
    EpisodeRecord,
)
from sparx_agency.tasks.planning.objnav_benchmark.run_info import RUN_SCHEMA

CONFIG = {"benchmark": "hm3d_v2", "split": "val",
          "agent": {"name": "agent_a", "lookahead_m": 0.5},
          "episodes": ["ep1", "ep2", "ep3"]}
UTC_STAMP = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ"


def record(episode_id, agent="agent_a", success=True, info=None):
    """A valid record; only what a test varies is a parameter."""
    counts = dict.fromkeys(ACTION_NAMES, 0)
    counts.update(MOVE_FORWARD=7, STOP=1)
    return EpisodeRecord(
        benchmark="hm3d_v2", split="val", episode_id=episode_id,
        scene_id="s1", target_category="chair", agent=agent,
        success=success, spl=0.75 if success else 0.0, soft_spl=0.8,
        distance_to_goal_m=0.05, path_length_m=1.75,
        observed_path_length_m=1.75, shortest_path_m=1.3125,
        start_distance_to_goal_m=1.3125, steps=8, stop_called=True,
        termination=TERMINATION_STOP, action_counts=counts, wall_s=0.25,
        native_metrics={"success": 1.0 if success else 0.0},
        agent_info={"statuses": {"forward": 7}} if info is None else info)


def fresh(tmp_path, config=CONFIG):
    return MetricsLogger(tmp_path / "run", config, argv=["objnav", "--quick"])


def logged(tmp_path, *episode_ids):
    """A fresh run with these episodes logged."""
    logger = fresh(tmp_path)
    for episode_id in episode_ids:
        logger.log(record(episode_id))
    return logger


def lines(logger):
    return logger.episodes_path.read_text(encoding="utf-8").splitlines()


def run_info(logger):
    return json.loads((logger.run_dir / "run.json").read_text(encoding="utf-8"))


def resume(logger, config=CONFIG):
    """Close ``logger`` -- one writer per directory, as a killed run's is -- and reopen its directory as a resume."""
    logger.close()
    return MetricsLogger(logger.run_dir, config, resume=True)


# -- a fresh run -----------------------------------------------------------------

def test_a_fresh_run_records_how_it_was_produced_before_any_episode(tmp_path):
    """run.json exists from the first second, so even a killed run can be explained."""
    logger = fresh(tmp_path)
    info = run_info(logger)
    assert info["schema"] == RUN_SCHEMA
    assert info["config"] == CONFIG
    assert info["argv"] == ["objnav", "--quick"]
    assert info["git"] and info["python"] and info["platform"]
    assert re.fullmatch(UTC_STAMP, info["started_utc"])
    assert (info["finished_utc"], info["n_episodes"], info["resumes"]) == (None, 0, [])
    assert lines(logger) == [] and logger.records == ()
    assert logger.completed_episode_ids() == frozenset()


def test_the_recorded_argv_defaults_to_the_process_command_line(tmp_path):
    """The exact command is what reproduces a run; it must not need passing by hand."""
    assert run_info(MetricsLogger(tmp_path / "run", CONFIG))["argv"] == sys.argv


def test_a_fresh_run_never_overwrites_a_directory_that_holds_results(tmp_path):
    """Overwriting episodes.jsonl destroys a finished sweep; appending to it mixes two."""
    first = logged(tmp_path, "ep1")
    before = (first.run_dir / "run.json").read_text()
    with pytest.raises(ResultsError, match="never overwritten"):
        fresh(tmp_path, config={"other": True})
    assert len(lines(first)) == 1
    assert (first.run_dir / "run.json").read_text() == before


@pytest.mark.parametrize("config", [
    {"lr": float("nan")}, {"far_m": float("inf")}, {"device": object()},
    {"ids": {1, 2}}], ids=["nan", "inf", "object", "set"])
def test_a_config_that_is_not_strict_json_is_refused_before_anything_is_written(
        tmp_path, config):
    """run.json must read back as written; NaN would be written as a token JSON does not have."""
    with pytest.raises(ResultsError, match="run config is not strict JSON"):
        MetricsLogger(tmp_path / "run", config)
    assert not (tmp_path / "run").exists()


def test_resume_must_be_a_real_bool(tmp_path):
    """``resume="false"`` reads as True and would append to a run meant to start fresh."""
    with pytest.raises(HarnessError, match="bool"):
        MetricsLogger(tmp_path / "run", CONFIG, resume="false")


# -- logging -------------------------------------------------------------------

def test_each_episode_is_one_sorted_strict_json_line(tmp_path):
    """Line-delimited and flushed per episode: a crash loses one episode, and a live run can be tailed."""
    logger = logged(tmp_path, "ep1", "ep2")
    assert lines(logger) == [
        json.dumps(record(i).to_row(), allow_nan=False, sort_keys=True)
        for i in ("ep1", "ep2")]
    assert logger.records == (record("ep1"), record("ep2"))
    assert logger.completed_episode_ids() == frozenset({"ep1", "ep2"})


def test_an_episode_logged_twice_is_refused_and_not_written(tmp_path):
    """A second row for one episode would weigh it double in every mean."""
    logger = logged(tmp_path, "ep1")
    with pytest.raises(ResultsError, match="'ep1' is already in"):
        logger.log(record("ep1", success=False))
    assert len(lines(logger)) == 1


def test_a_record_of_another_agent_is_refused_in_the_same_run(tmp_path):
    """One directory holds one experiment; a second agent's rows would be summarised as the first's."""
    logger = logged(tmp_path, "ep1")
    with pytest.raises(ResultsError, match="one agent on one benchmark split"):
        logger.log(record("ep2", agent="agent_b"))
    assert len(lines(logger)) == 1


def test_a_record_that_is_not_strict_json_is_refused_and_not_written(tmp_path):
    """A NaN written as a bare token would make the line unreadable by the resume that needs it."""
    logger = fresh(tmp_path)
    with pytest.raises(ResultsError, match="strict JSON"):
        logger.log(record("ep1", info={"heading": float("nan")}))
    assert lines(logger) == [] and logger.records == ()


# -- resuming ------------------------------------------------------------------

def test_a_resume_reads_back_exactly_the_records_that_were_logged(tmp_path):
    """Resumed episodes are reused as they are; any drift would change the summary."""
    first = logged(tmp_path, "ep1", "ep2")
    resumed = resume(first)
    assert resumed.records == first.records == (record("ep1"), record("ep2"))
    assert resumed.completed_episode_ids() == frozenset({"ep1", "ep2"})
    resumed.log(record("ep3"))
    assert len(lines(resumed)) == 3


def test_a_resume_with_a_different_config_is_refused_and_names_the_difference(tmp_path):
    """Resuming with a different config would mix two experiments in one file."""
    first = logged(tmp_path, "ep1")
    changed = dict(CONFIG, agent={"name": "agent_a", "lookahead_m": 0.75})
    with pytest.raises(ResultsError, match=r"different config \(agent differ\)"):
        resume(first, changed)


def test_a_resume_compares_the_config_as_json_exactly(tmp_path):
    """``True == 1`` in Python, but they are different experiments; a tuple and a list are the same JSON."""
    first = fresh(tmp_path, config={"flag": True, "size": (64, 48)})
    with pytest.raises(ResultsError, match="different config"):
        resume(first, {"flag": 1, "size": (64, 48)})
    assert resume(first, {"flag": True, "size": [64, 48]}).records == ()


def test_a_resume_of_a_directory_without_a_run_is_refused(tmp_path):
    """Resuming nothing would silently start a fresh run under a flag that promised continuity."""
    with pytest.raises(ResultsError, match="cannot resume"):
        MetricsLogger(tmp_path / "missing", CONFIG, resume=True)


@pytest.mark.parametrize("cut", [1, 20], ids=["newline_lost", "mid_line"])
def test_a_truncated_last_line_is_reported_with_its_number_not_skipped(tmp_path, cut):
    """A skipped line is an episode silently gone from the mean; the message must say how to recover."""
    logger = logged(tmp_path, "ep1", "ep2")
    text = logger.episodes_path.read_text(encoding="utf-8")
    logger.episodes_path.write_text(text[:-cut], encoding="utf-8")
    with pytest.raises(ResultsError, match=r"line 2 is truncated.*delete that line"):
        resume(logger)


@pytest.mark.parametrize("bad,problem", [
    ("{not json", "Expecting"),
    ('{"schema": "objnav_episode/0"}', "another version"),
    (json.dumps(dict(record("ep1").to_row(), agent_info={"x": float("nan")})),
     "NaN is not JSON"),
], ids=["garbage", "old_schema", "nan_token"])
def test_a_malformed_line_before_the_last_is_reported_with_its_number(tmp_path, bad, problem):
    """Not a crash but an edit or a join; skipping it would drop an episode without a word."""
    logger = logged(tmp_path, "ep1", "ep2")
    rows = lines(logger)
    rows[0] = bad
    logger.episodes_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ResultsError, match=r"line 1 is not a valid episode row.*%s.*edited" % problem):
        resume(logger)


def test_a_repeated_line_is_refused_on_resume(tmp_path):
    """Two files concatenated, or an episode re-run by hand, must not be read as two episodes."""
    logger = logged(tmp_path, "ep1")
    with logger.episodes_path.open("a", encoding="utf-8") as handle:
        handle.write(lines(logger)[0] + "\n")
    with pytest.raises(ResultsError, match=r"line 2: episode 'ep1' is already in"):
        resume(logger)


# -- finishing -----------------------------------------------------------------

def test_finish_writes_the_summary_and_marks_the_run_finished(tmp_path):
    """summary.json and run.json's outcome are what a later reader trusts without re-running."""
    logger = logged(tmp_path, "ep1", "ep2")
    summary = summarise(logger.records, resamples=50)
    path = logger.finish(summary)
    assert path == logger.run_dir / "summary.json"
    assert json.loads(path.read_text()) == json.loads(json.dumps(summary.to_dict()))
    info = run_info(logger)
    assert info["n_episodes"] == 2 and info["config"] == CONFIG
    assert re.fullmatch(UTC_STAMP, info["finished_utc"])


def test_finish_refuses_a_summary_of_other_episodes_than_the_file_holds(tmp_path):
    """A summary beside episodes it does not describe is a number nobody can check."""
    logger = logged(tmp_path, "ep1", "ep2")
    with pytest.raises(ResultsError, match="exactly the episodes beside it"):
        logger.finish(summarise([record("ep1")], resamples=50))
    assert not (logger.run_dir / "summary.json").exists()


def test_a_resume_reopens_a_finished_run_and_records_who_resumed_it(tmp_path):
    """A run resumed under changed code mixes two revisions; run.json must show it once the resume writes."""
    logger = logged(tmp_path, "ep1")
    logger.finish(summarise(logger.records, resamples=50))
    resumed = MetricsLogger(logger.run_dir, CONFIG, resume=True,
                            argv=["objnav", "--resume"])
    resumed.log(record("ep2"))
    info = run_info(logger)
    assert info["finished_utc"] is None
    assert [(r["argv"], r["episodes_done"]) for r in info["resumes"]] == [
        (["objnav", "--resume"], 1)]
    assert info["resumes"][0]["git"]
