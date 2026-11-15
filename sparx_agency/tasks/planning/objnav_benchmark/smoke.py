"""Run the whole headless loop on a fake building; the thing to run first.

``python -m sparx_agency.tasks.planning.objnav_benchmark.smoke [--quick]``
builds a small ASCII building -- a bedroom, a bathroom and a hall, with a bed,
a toilet and a chair -- puts the fake environment on it, and drives the
privileged oracle through every episode with the real headless agent, action
converter, runner, scoring, logger and comparison table: the whole pipeline a
benchmark branch uses, minus the simulator. If the oracle scores an SR below
100, or any episode ends on an agent error, the pipeline has lost a success
that no method could be blamed for: the command then says why and exits 1,
so a scripted pre-flight (``smoke --quick && ./run_sweep.sh``) stops there.

A **test rig, not a benchmark**: the oracle reads the ground truth, so its
numbers bound the pipeline and are never a result to report. Results go under
``~/objnav_benchmark/fake/test/<UTC stamp>`` unless ``--out`` says otherwise --
outside the repository, like every run.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import random
import sys
import types
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sparx_agency.core.planning.objnav.agent.headless_agent import (
    HeadlessObjNavAgent,
)
from sparx_agency.core.planning.objnav.camera_geometry import intrinsics_from_hfov
from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark.comparison import (
    comparison_table,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.env import (
    DEFAULT_AGENT_RADIUS_M,
    DEFAULT_SUCCESS_DISTANCE_M,
    FakeEpisodeSpec,
    FakeObjNavEnv,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.geodesics import (
    geodesic_field,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.labels import (
    fake_label_mapper,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.oracle_policy import (
    OracleSearchPolicy,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.raster import (
    goal_mask,
    navigable_mask,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.world import GridWorld
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark.records import EpisodeRecord
from sparx_agency.tasks.planning.objnav_benchmark.results_io import (
    default_run_dir,
)
from sparx_agency.tasks.planning.objnav_benchmark.runner import run_benchmark
from sparx_agency.tasks.planning.objnav_benchmark.summaries import (
    BenchmarkSummary,
)

#: The demo building, north up, 0.25 m cells (6 m x 4 m): a bedroom (bed in
#: its north-west corner) and a bathroom (toilet) over a hall (chair against
#: its south wall), joined by 1.25 m doors -- wide enough to leave three
#: navigable cells for a 0.18 m agent.
DEMO_MAP = (
    "########################",
    "#bbbb......#...........#",
    "#bbbb......#...........#",
    "#..........#........t..#",
    "#..........#...........#",
    "#..........#...........#",
    "#..........#...........#",
    "###.....######.....#####",
    "#......................#",
    "#......................#",
    "#......................#",
    "#......................#",
    "#......................#",
    "#......................#",
    "#..........c...........#",
    "########################",
)

#: The demo map's object characters. Read-only: the map and its categories
#: must not drift apart between runs of one process.
DEMO_LEGEND = types.MappingProxyType({"b": "bed", "c": "chair", "t": "toilet"})

#: The scene every demo episode reports.
DEMO_SCENE_ID = "demo_building"
#: The seed of the demo episodes' starts.
DEMO_SEED = 0
#: Episodes in a default run, and in ``--quick``.
DEFAULT_EPISODES = 6
QUICK_EPISODES = 2
#: No demo start is nearer its goal than this, metres: an episode that starts
#: at the goal tests nothing.
MIN_START_DISTANCE_M = 2.0

#: The demo camera: Habitat's ObjectNav mount height and a HM3D-like field of
#: view, at a resolution small enough to ray cast every pixel in milliseconds.
CAMERA_WIDTH = 64
CAMERA_HEIGHT = 48
CAMERA_HFOV_DEG = 79.0
CAMERA_MOUNT_M = 0.88
CAMERA_MIN_DEPTH_M = 0.1
CAMERA_MAX_DEPTH_M = 5.0

#: How the command line is recorded in ``run.json``.
MODULE = "sparx_agency.tasks.planning.objnav_benchmark.smoke"


def build_demo_world() -> GridWorld:
    """The demo building of :data:`DEMO_MAP`."""
    return GridWorld.from_ascii(DEMO_MAP, DEMO_LEGEND)


def demo_camera() -> CameraSpec:
    """The demo camera: 64 x 48, 79 degrees across, 0.88 m up, 0.1 to 5 m of depth."""
    return CameraSpec(intrinsics_from_hfov(CAMERA_WIDTH, CAMERA_HEIGHT,
                                           CAMERA_HFOV_DEG),
                      height_m=CAMERA_MOUNT_M, min_depth_m=CAMERA_MIN_DEPTH_M,
                      max_depth_m=CAMERA_MAX_DEPTH_M)


def build_demo_episodes(world: GridWorld, n: int, seed: int = 0, *,
                        success_distance_m: float = DEFAULT_SUCCESS_DISTANCE_M,
                        agent_radius_m: float = DEFAULT_AGENT_RADIUS_M
                        ) -> List[FakeEpisodeSpec]:
    """``n`` deterministic episodes: targets in turn, starts drawn from the navigable floor.

    Episode ``i`` looks for ``world.categories[i % len(categories)]`` and
    starts at the centre of a navigable cell at least
    :data:`MIN_START_DISTANCE_M` of walking from that category's goal region,
    facing a random heading. ``random.Random(seed)`` over sorted candidate
    cells, so the same arguments give the same episodes on every platform and
    numpy version.

    Args:
        world: The building.
        n: How many episodes, at least 1.
        seed: The draw's seed.
        success_distance_m: The environment's goal-region radius.
        agent_radius_m: The environment's agent radius.

    Returns:
        The episodes, ids ``"demo-000"``, ``"demo-001"``, ...

    Raises:
        ObjNavError: If ``n`` or ``seed`` is not an integer, ``n < 1``, the
            world has no object, or a category has no start far enough away.
    """
    for name, value in (("n", n), ("seed", seed)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ObjNavError("%s must be an integer, got %r" % (name, value))
    if n < 1:
        raise ObjNavError("n must be at least 1, got %d" % n)
    categories = world.categories
    if not categories:
        raise ObjNavError("the world has no object to look for")
    navigable = navigable_mask(world, agent_radius_m)
    candidates = {}
    for category in categories:
        goal = goal_mask(world, category, success_distance_m, agent_radius_m)
        field = geodesic_field(world, goal, navigable)
        cells = [(int(r), int(c)) for r, c in zip(*(field >= MIN_START_DISTANCE_M).nonzero())
                 if math.isfinite(field[r, c])]
        if not cells:
            raise ObjNavError("no navigable cell is %.1f m or more from any %r"
                              % (MIN_START_DISTANCE_M, category))
        candidates[category] = sorted(cells)
    rng = random.Random(seed)
    episodes = []
    for index in range(n):
        category = categories[index % len(categories)]
        row, col = rng.choice(candidates[category])
        x, y = world.cell_center(row, col)
        episodes.append(FakeEpisodeSpec(
            episode_id="demo-%03d" % index, scene_id=DEMO_SCENE_ID,
            target_category=category,
            start=AgentPose(x, y, 0.0, rng.uniform(-math.pi, math.pi))))
    return episodes


def build_demo_env(n_episodes: int = DEFAULT_EPISODES, *, seed: int = DEMO_SEED,
                   camera: Optional[CameraSpec] = None) -> FakeObjNavEnv:
    """The fake environment on the demo building, with :func:`build_demo_episodes`.

    Args:
        n_episodes: How many episodes.
        seed: The episodes' seed.
        camera: The agent's camera; :func:`demo_camera` when None.
    """
    world = build_demo_world()
    return FakeObjNavEnv(world, build_demo_episodes(world, n_episodes, seed),
                         camera=demo_camera() if camera is None else camera)


def _config(env: FakeObjNavEnv, agent: HeadlessObjNavAgent, n: int,
            seed: int) -> Dict[str, Any]:
    """Everything that decides this run's numbers, as strict JSON for ``run.json``."""
    camera = env.camera
    return {
        "env": env.name, "world": "demo", "world_map": list(DEMO_MAP),
        "episodes": n, "seed": seed, "agent": agent.name,
        "max_steps": env.max_steps,
        "success_distance_m": env.success_distance_m,
        "agent_radius_m": env.agent_radius_m,
        "camera": {"width": camera.intrinsics.width,
                   "height": camera.intrinsics.height,
                   "hfov_deg": CAMERA_HFOV_DEG, "height_m": camera.height_m,
                   "min_depth_m": camera.min_depth_m,
                   "max_depth_m": camera.max_depth_m},
    }


def gate_failures(summary: BenchmarkSummary) -> Tuple[str, ...]:
    """Why a smoke run falls short of the privileged oracle's perfect score: one sentence per failure, none when it does not.

    An SR below 1.0 is a success the pipeline lost; an agent error, a crash
    somewhere inside it.
    """
    failures = []
    if summary.overall.success_rate < 1.0:
        failures.append("the oracle's SR is %.1f%%, not 100%%"
                        % (100.0 * summary.overall.success_rate))
    if summary.agent_errors:
        failures.append("%d episode%s ended on an agent error"
                        % (summary.agent_errors,
                           "" if summary.agent_errors == 1 else "s"))
    return tuple(failures)


def _report(position: int, total: int, record: EpisodeRecord) -> None:
    print("[%d/%d] %s %-6s success=%-5s spl=%.3f steps=%d"
          % (position, total, record.episode_id, record.target_category,
             record.success, record.spl, record.steps))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    count = parser.add_mutually_exclusive_group()
    count.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                       help="episodes to run (default %(default)s)")
    count.add_argument("--quick", action="store_true",
                       help="run %d episodes" % QUICK_EPISODES)
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="results directory (default: "
                             "~/objnav_benchmark/fake/test/<UTC stamp>)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the oracle on the demo building, log it, print its table and where it went.

    Args:
        argv: The arguments, without the program name; ``sys.argv[1:]`` when
            None.

    Returns:
        0 when the oracle found every target without an agent error; 1, after
        saying why (:func:`gate_failures`), otherwise.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    n = QUICK_EPISODES if args.quick else args.episodes
    if n < 1:
        parser.error("--episodes must be at least 1, got %d" % n)
    env = build_demo_env(n)
    agent = HeadlessObjNavAgent(OracleSearchPolicy(env), fake_label_mapper())
    run_dir = (args.out if args.out is not None
               else default_run_dir(env.benchmark, env.split))
    recorded = [MODULE] + list(sys.argv[1:] if argv is None else argv)
    logger = MetricsLogger(run_dir, _config(env, agent, n, DEMO_SEED),
                           argv=recorded)
    try:
        summary = run_benchmark(env, agent, logger=logger, progress=_report)
    finally:
        env.close()
    print(comparison_table(summary, []))
    print("results: %s" % logger.run_dir)
    failures = gate_failures(summary)
    if failures:
        print("smoke run FAILED: %s -- the oracle is privileged, so this is a "
              "pipeline bug, not a method's weakness" % "; ".join(failures),
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
