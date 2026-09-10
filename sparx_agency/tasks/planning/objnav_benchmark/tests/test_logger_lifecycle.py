"""Who may write a results directory, and when: one logger at a time, and a resume touches run.json only once it acts.

A detached sweep still running and a resume of it would both append the same
episodes and both finish, each with a summary that matches its own records
but not the file. A resume refused before its first episode must leave a
finished run looking finished.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

from sparx_agency.tasks.planning.objnav_benchmark.aggregate import summarise
from sparx_agency.tasks.planning.objnav_benchmark.errors import (
    HarnessError,
    ResultsError,
)
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark.runner import (
    run_benchmark,
    run_episode,
)
from sparx_agency.tasks.planning.objnav_benchmark.tests.corridor import (
    CONFIG,
    GOALS,
    CorridorEnv,
    ScriptedAgent,
)

#: The directory that holds ``sparx_agency/``: a child interpreter's cwd.
REPO_ROOT = pathlib.Path(__import__("sparx_agency").__file__).resolve().parents[1]


def episode(episode_id):
    """One corridor episode's record."""
    return run_episode(CorridorEnv(GOALS), ScriptedAgent(), episode_id)


def files(run_dir):
    """What a reader of the directory would see: run.json and episodes.jsonl, verbatim."""
    return {name: (run_dir / name).read_text(encoding="utf-8")
            for name in ("run.json", "episodes.jsonl")}


# -- one writer ---------------------------------------------------------------------

def test_a_resume_of_a_run_still_being_written_is_refused_and_writes_nothing(tmp_path):
    """The live sweep and the mistaken resume would both append e2 and e3; the second must be turned away."""
    live = MetricsLogger(tmp_path / "run", CONFIG, argv=["live"])
    live.log(episode("e1"))
    before = files(live.run_dir)
    with pytest.raises(ResultsError, match="another process is writing this run"):
        MetricsLogger(live.run_dir, CONFIG, resume=True, argv=["resume"])
    assert files(live.run_dir) == before
    live.log(episode("e2"))
    assert live.completed_episode_ids() == {"e1", "e2"}


def test_another_process_holding_the_directory_is_refused_until_it_exits(tmp_path):
    """The lock must bind across processes, and die with its holder, or a crash would block its own resume."""
    run_dir = tmp_path / "run"
    code = ("import sys\n"
            "from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger\n"
            "logger = MetricsLogger(%r, %r, argv=[])\n"
            "print('ready', flush=True)\n"
            "sys.stdin.readline()\n" % (str(run_dir), CONFIG))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT), os.environ.get("PYTHONPATH", "")) if p)
    child = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, cwd=str(REPO_ROOT), env=env)
    try:
        ready = child.stdout.readline().strip()
        assert ready == "ready", child.stderr.read()
        with pytest.raises(ResultsError, match="another process is writing this run"):
            MetricsLogger(run_dir, CONFIG, resume=True)
    finally:
        child.stdin.close()  # the child reads EOF and exits without close or finish
        try:
            child.wait(timeout=60)
        finally:
            child.kill()
            child.stdout.close()
            child.stderr.close()
    assert child.returncode == 0
    MetricsLogger(run_dir, CONFIG, resume=True).close()


def test_finish_releases_the_directory_to_a_later_resume(tmp_path):
    """A finished run must be resumable, to add episodes or to re-finish it."""
    run_benchmark(CorridorEnv(GOALS), ScriptedAgent(),
                  logger=MetricsLogger(tmp_path / "run", CONFIG, argv=[]))
    with MetricsLogger(tmp_path / "run", CONFIG, resume=True) as resumed:
        assert resumed.completed_episode_ids() == frozenset(GOALS)


def test_a_run_that_raises_inside_the_context_manager_releases_the_directory(tmp_path):
    """The with-block is how a crashing run lets its own resume in."""
    crashing = ScriptedAgent(misbehave_at=0, in_episode="e3")
    with pytest.raises(RuntimeError, match="boom"):
        with MetricsLogger(tmp_path / "run", CONFIG) as logger:
            run_benchmark(CorridorEnv(GOALS), crashing, logger=logger)
    with MetricsLogger(tmp_path / "run", CONFIG, resume=True) as resumed:
        assert resumed.completed_episode_ids() == {"e1", "e2"}


def test_close_is_idempotent_and_a_closed_logger_refuses_to_write(tmp_path):
    """Without the lock, a closed logger's rows could interleave with the next writer's."""
    logger = MetricsLogger(tmp_path / "run", CONFIG)
    logger.log(episode("e1"))
    logger.close()
    logger.close()
    with pytest.raises(ResultsError, match="closed"):
        logger.log(episode("e2"))
    with pytest.raises(ResultsError, match="closed"):
        logger.finish(summarise(logger.records, resamples=50))
    assert len(logger.episodes_path.read_text().splitlines()) == 1
    MetricsLogger(tmp_path / "run", CONFIG, resume=True).close()


def test_finish_refuses_a_file_another_writer_appended_to(tmp_path):
    """Each writer's summary matches its own records; only the file, read back, shows the rows it does not describe."""
    logger = MetricsLogger(tmp_path / "run", CONFIG)
    for episode_id in ("e1", "e2"):
        logger.log(episode(episode_id))
    with logger.episodes_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(episode("e3").to_row(), sort_keys=True) + "\n")
    with pytest.raises(ResultsError, match="holds 3 rows but this logger wrote or read 2"):
        logger.finish(summarise(logger.records, resamples=50))
    assert not (logger.run_dir / "summary.json").exists()


# -- run.json and a resume ------------------------------------------------------------

@pytest.mark.parametrize("options", [{"episode_ids": ["e1", "e2"]},
                                     {"on_agent_error": "recrod"}],
                         ids=["other_episode_list", "typo_in_option"])
def test_a_refused_resume_leaves_a_finished_run_json_untouched(tmp_path, options):
    """A resume refused before its first episode must not make a finished run read as killed."""
    run_dir = tmp_path / "run"
    run_benchmark(CorridorEnv(GOALS), ScriptedAgent(),
                  logger=MetricsLogger(run_dir, CONFIG, argv=[]))
    before = files(run_dir)
    assert json.loads(before["run.json"])["finished_utc"] is not None
    resumed = MetricsLogger(run_dir, CONFIG, resume=True, argv=["oops"])
    with pytest.raises(HarnessError):
        run_benchmark(CorridorEnv(GOALS), ScriptedAgent(), logger=resumed, **options)
    assert files(run_dir) == before


def test_a_resume_enters_run_json_on_its_first_log(tmp_path):
    """Once a resume writes an episode, run.json must say so: two revisions may now share the file."""
    with MetricsLogger(tmp_path / "run", CONFIG, argv=["first"]) as first:
        first.log(episode("e1"))
    before = files(first.run_dir)
    resumed = MetricsLogger(first.run_dir, CONFIG, resume=True, argv=["again"])
    assert files(first.run_dir) == before
    resumed.log(episode("e2"))
    info = json.loads((first.run_dir / "run.json").read_text(encoding="utf-8"))
    assert info["finished_utc"] is None
    assert [(r["argv"], r["episodes_done"]) for r in info["resumes"]] == [(["again"], 1)]


def test_a_resume_that_only_finishes_is_still_entered_in_run_json(tmp_path):
    """Re-finishing a run with nothing left to run is still a resume under this revision."""
    run_benchmark(CorridorEnv(GOALS), ScriptedAgent(),
                  logger=MetricsLogger(tmp_path / "run", CONFIG, argv=[]))
    run_benchmark(CorridorEnv(GOALS), ScriptedAgent(),
                  logger=MetricsLogger(tmp_path / "run", CONFIG, resume=True,
                                       argv=["again"]))
    info = json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))
    assert info["finished_utc"] is not None and info["n_episodes"] == 3
    assert [r["argv"] for r in info["resumes"]] == [["again"]]
