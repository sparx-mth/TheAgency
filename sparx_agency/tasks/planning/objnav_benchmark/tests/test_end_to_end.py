"""The whole headless loop on the demo building: the oracle scores SR 1.0, the same way every time.

This is the first place every package runs together -- environment, label
mapper, headless agent, action converter, runner, scoring, statistics, logger
and comparison table -- so a unit that is right alone and wrong in
combination fails here. The oracle is privileged, so anything short of SR 1.0
is a pipeline bug, not a method's weakness -- and the smoke command exits 1
on it; a random walk through the same loop must score lower, or the success
check is not checking anything.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import random
import subprocess
import sys
import zlib

import pytest

from sparx_agency.core.planning.objnav.agent.headless_agent import (
    HeadlessObjNavAgent,
)
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.tasks.planning.objnav_benchmark import smoke
from sparx_agency.tasks.planning.objnav_benchmark.agent_contract import (
    ON_AGENT_ERROR_RECORD,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.labels import (
    fake_label_mapper,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.oracle_policy import (
    OracleSearchPolicy,
)
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark.runner import run_benchmark
from sparx_agency.tasks.planning.objnav_benchmark.smoke import (
    DEMO_SEED,
    MIN_START_DISTANCE_M,
    MODULE,
    build_demo_env,
    build_demo_episodes,
    build_demo_world,
    gate_failures,
    main,
)
from sparx_agency.tasks.planning.objnav_benchmark.tests.corridor import (
    CorridorEnv,
    ScriptedAgent,
)

#: Episodes of the end-to-end run: two of each category.
N_EPISODES = 6
#: The SPL the oracle must reach. The fake's grid geodesic and its
#: clearance-weighted route cost a few percent; far below this is a bug.
MIN_ORACLE_SPL = 0.8
#: The directory that holds ``sparx_agency/``: the fresh interpreter's cwd.
REPO_ROOT = pathlib.Path(__import__("sparx_agency").__file__).resolve().parents[1]
#: What importing the fake or the smoke run must not pull in.
HEAVY = ("torch", "tensorrt", "pycuda", "cv2", "requests", "PIL", "scipy",
         "yaml", "rclpy", "rospy", "ompl", "networkx", "skimage")


class RandomWalkPolicy:
    """Heads for a random point a metre away each step and stops at random. Sees nothing privileged."""

    name = "random_walk"

    def __init__(self, stop_probability=0.05):
        self.stop_probability = stop_probability

    def reset(self, episode, target):
        # zlib, not hash(): hash() of a string changes between interpreter runs.
        self.rng = random.Random(zlib.crc32(episode.episode_id.encode("utf-8")))

    def plan(self, observation):
        if self.rng.random() < self.stop_probability:
            return NavigationCommand.stop_here()
        pose = observation.pose
        heading = pose.yaw + self.rng.uniform(-math.pi / 2, math.pi / 2)
        return NavigationCommand.follow(
            [(pose.x + math.cos(heading), pose.y + math.sin(heading))])


class StopAtOncePolicy:
    """Stops where it starts; every demo start is 2 m or more from its goal, so every episode fails."""

    name = "stop_at_once"

    def reset(self, episode, target):
        pass

    def plan(self, observation):
        return NavigationCommand.stop_here()


def run(policy_for, directory):
    """Run a fresh demo env with ``policy_for(env)`` behind the headless agent; summary and records."""
    env = build_demo_env(N_EPISODES)
    agent = HeadlessObjNavAgent(policy_for(env), fake_label_mapper())
    logger = MetricsLogger(directory, {"policy": agent.name}, argv=[])
    return run_benchmark(env, agent, logger=logger), logger.records


def without_wall_time(records):
    rows = [record.to_row() for record in records]
    for row in rows:
        row.pop("wall_s")
    return rows


@pytest.fixture(scope="module")
def oracle_run(tmp_path_factory):
    """One oracle run on the demo building, shared by the tests that only read it."""
    return run(OracleSearchPolicy, tmp_path_factory.mktemp("oracle") / "run")


# -- the oracle ------------------------------------------------------------------

def test_the_oracle_finds_every_target_on_the_demo_building(oracle_run):
    """Privileged and deterministic: a single miss is a bug somewhere in the pipeline."""
    summary, records = oracle_run
    assert summary.agent == "headless/oracle"
    assert summary.overall.n_episodes == N_EPISODES
    assert summary.overall.success_rate == 1.0
    assert summary.terminations == {"stop": N_EPISODES, "step_limit": 0, "agent_error": 0}
    assert summary.overall.spl >= MIN_ORACLE_SPL
    assert summary.overall.soft_spl == pytest.approx(summary.overall.spl)
    assert {group.key for group in summary.by_category} == {"bed", "chair", "toilet"}
    assert all(record.agent_info["policy"]["holds"] == 0 for record in records)


def test_the_oracle_is_never_blocked_on_the_demo_building(oracle_run):
    """Its clearance-weighted route keeps a cell off the walls; a blocked step means that stopped working."""
    _, records = oracle_run
    for record in records:
        info = record.agent_info
        assert info["blocked_forward"] == 0, record.episode_id
        assert info["blocked_notifications"] == info["policy"]["blocked"] == 0, (
            record.episode_id)


def test_every_episodes_path_is_what_its_observed_poses_trace(oracle_run):
    """The environment's p and the runner's must agree to rounding, or SPL rests on a frame bug."""
    _, records = oracle_run
    for record in records:
        assert record.path_length_m >= MIN_START_DISTANCE_M - 1.0
        assert abs(record.observed_path_length_m - record.path_length_m) <= 1e-9
        assert record.distance_to_goal_m == 0.0
        assert record.native_metrics == {"success": 1.0, "distance_to_goal": 0.0}
        assert record.spl == pytest.approx(
            record.shortest_path_m / max(record.shortest_path_m, record.path_length_m))


def test_a_random_walk_through_the_same_loop_scores_lower(oracle_run, tmp_path):
    """If a policy that ignores the target scored as well, success would not measure finding it."""
    oracle_summary, _ = oracle_run
    summary, records = run(lambda env: RandomWalkPolicy(), tmp_path / "run")
    assert summary.agent == "headless/random_walk"
    assert summary.overall.success_rate < oracle_summary.overall.success_rate
    assert summary.overall.spl < oracle_summary.overall.spl
    assert all(record.termination in ("stop", "step_limit") for record in records)


def test_the_same_run_twice_gives_identical_records_and_summary(oracle_run, tmp_path):
    """Every seed is fixed, so two runs differ only in wall time; anything else is hidden state."""
    summary, records = oracle_run
    again_summary, again_records = run(OracleSearchPolicy, tmp_path / "run")
    assert again_summary == summary
    assert without_wall_time(again_records) == without_wall_time(records)


# -- the smoke run -----------------------------------------------------------------

def test_the_smoke_run_writes_its_results_and_returns_zero(tmp_path, capsys):
    """The first command anyone runs; it must leave a complete, resumable results directory."""
    out = tmp_path / "run"
    assert main(["--quick", "--out", str(out)]) == 0
    captured = capsys.readouterr()
    printed = captured.out
    assert "| Method | SR (%)" in printed and "headless/oracle (ours)" in printed
    assert str(out) in printed
    assert "FAILED" not in captured.err
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["overall"]["n_episodes"] == 2
    assert summary["overall"]["success_rate"] == 1.0
    lines = (out / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    run_info = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run_info["n_episodes"] == 2 and run_info["finished_utc"] is not None
    assert run_info["argv"] == [MODULE, "--quick", "--out", str(out)]
    assert run_info["config"]["episodes"] == 2
    assert run_info["config"]["agent"] == "headless/oracle"


def test_the_smoke_run_returns_one_and_says_why_when_the_oracle_misses(
        tmp_path, capsys, monkeypatch):
    """A scripted pre-flight (smoke --quick && run the sweep) must stop at a pipeline that lost a success."""
    monkeypatch.setattr(smoke, "OracleSearchPolicy", lambda env: StopAtOncePolicy())
    assert main(["--quick", "--out", str(tmp_path / "run")]) == 1
    err = capsys.readouterr().err
    assert "smoke run FAILED: the oracle's SR is 0.0%, not 100%" in err


def test_the_smoke_gate_passes_the_oracle_and_fails_an_agent_error(oracle_run):
    """An agent error is a crash inside the pipeline, even where the SR alone would not show it."""
    summary, _ = oracle_run
    assert gate_failures(summary) == ()
    crashed = run_benchmark(CorridorEnv({"e1": 1.0, "e2": 1.0}),
                            ScriptedAgent(misbehave_at=1, in_episode="e2"),
                            on_agent_error=ON_AGENT_ERROR_RECORD)
    assert gate_failures(crashed) == ("the oracle's SR is 50.0%, not 100%",
                                      "1 episode ended on an agent error")


@pytest.mark.parametrize("argv", [["--episodes", "0"], ["--quick", "--episodes", "3"]],
                         ids=["zero_episodes", "quick_and_episodes"])
def test_the_smoke_command_line_refuses_a_run_it_cannot_mean(argv, tmp_path, capsys):
    """Zero episodes, or two episode counts at once, is a typo to report, not a run to start."""
    out = tmp_path / "run"
    with pytest.raises(SystemExit) as exit_info:
        main(argv + ["--out", str(out)])
    assert exit_info.value.code == 2
    assert not out.exists()
    capsys.readouterr()


def test_the_demo_episodes_are_deterministic_cycle_the_categories_and_start_far_away():
    """The smoke run is only comparable with itself if its episodes never change."""
    world = build_demo_world()
    episodes = build_demo_episodes(world, N_EPISODES, DEMO_SEED)
    assert episodes == build_demo_episodes(build_demo_world(), N_EPISODES, DEMO_SEED)
    assert episodes != build_demo_episodes(world, N_EPISODES, DEMO_SEED + 1)
    assert [e.target_category for e in episodes] == ["bed", "chair", "toilet"] * 2
    assert len({e.episode_id for e in episodes}) == N_EPISODES
    env = build_demo_env(N_EPISODES)
    for episode in episodes:
        env.reset(episode.episode_id)
        env.step(DiscreteAction.STOP)  # at once: the measurement's l is d0
        assert env.measure().shortest_path_m >= MIN_START_DISTANCE_M


def test_the_fake_label_mapper_covers_every_category_of_the_demo_building():
    """A category without a row would fail at reset, before the first step of a real run."""
    mapper = fake_label_mapper()
    assert set(build_demo_world().categories) <= set(mapper.categories())
    assert mapper.target_labels("chair").accepts("Armchair")


@pytest.mark.parametrize("module", [
    "sparx_agency.tasks.planning.objnav_benchmark.fake_env",
    "sparx_agency.tasks.planning.objnav_benchmark.smoke",
])
def test_importing_the_fake_and_the_smoke_run_pulls_nothing_heavy(module):
    """The thing to run first must run anywhere numpy does; a fresh interpreter is the only honest check."""
    code = ("import importlib, sys\n"
            "importlib.import_module(%r)\n"
            "print(','.join(sorted({n.split('.')[0] for n in sys.modules} & set(%r))))\n"
            % (module, HEAVY))
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT), os.environ.get("PYTHONPATH", "")) if p)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, cwd=str(REPO_ROOT), env=env, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", "%s imported %s" % (module, result.stdout)
