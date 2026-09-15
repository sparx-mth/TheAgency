"""HM3D's success rule, its path accounting and its four metrics, on geometry a test chose.

Every mistake this adapter can make produces a plausible number instead of a
crash: a success radius compared with ``<=`` instead of ``<``, an ``l`` taken
from the publisher's row rather than from the navmesh, a ``p`` that counts a
turn, a SoftSPL invented where habitat-lab has none. So the two seams
``HM3DEnv`` offers -- ``simulator=`` and ``distance_factory=`` -- are used to
put known quantities on both sides of the measurement and check the corners.

habitat-sim, its meshes and habitat-lab are not installed here, and none of
them is needed to pin arithmetic. The one place the real pathfinder is
unavoidable is :class:`ViewPointDistance`, so that class is exercised against a
stub module standing in for ``habitat_sim``: the calls it makes into the
pathfinder *are* the definition of the measure, and getting them wrong (a fresh
path per query, view points re-requested, an unreachable goal read as a number)
is invisible in any single episode's score.

The last test drives the adapter through the harness that will really run the
benchmark, because satisfying ``ObjNavEnv`` method by method is not the same as
satisfying its contract step by step.
"""
from __future__ import annotations

from dataclasses import replace
import gzip
import json
import math
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.core.planning.objnav.interfaces.agent import ObjNavAgent
from sparx_agency.core.planning.objnav.labels.datasets.hm3d import CATEGORIES
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.decision import AgentDecision
from sparx_agency.core.planning.objnav.types.measurement import (
    NATIVE_KEYS, TERMINATION_STEP_LIMIT, TERMINATION_STOP)
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark.env_contract import checked_reset
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark.runner import run_benchmark
from sparx_agency.tasks.planning.objnav_benchmark.scoring import path_length_3d, score_episode
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.dataset import HM3DDataset
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.env import HM3DEnv
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.geodesic import (
    ViewPointDistance, view_point_positions)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import HM3D_V1

A = DiscreteAction

#: Two scene folders whose content shards sort in the order written here, so
#: the published episode order a run sees is the order of this file.
SCENE, OTHER_SCENE = "00800-TEEsavR23oF", "00802-ziup5kvtCCR"
EPISODE, OTHER_EPISODE = SCENE + "/000000", OTHER_SCENE + "/000000"

#: The published protocol at an 8x6 frame and a four-action budget. Nothing
#: here reads a pixel, and a budget an agent can actually spend is what makes
#: the step-limit corner of the success rule reachable in a test.
PROTOCOL = replace(HM3D_V1, width=8, height=6, max_steps=4)

#: Habitat's published start for every row below (+Y up), and the ENU pose the
#: bridge's convention turns it into: ``x = -z``, ``y = -x``, ``z = y``.
START = (1.0, 0.5, 2.0)
START_ENU = (-2.0, -1.0, 0.5)


def goal():
    """One evaluator-only goal row, shaped like the publisher's.

    Two view points, because a distance measured to one of them is not the
    measure habitat-lab defines.
    """
    return {"object_id": 7, "object_category": "chair", "position": [1.0, 0.5, 3.0],
            "view_points": [{"agent_state": {"position": [1.0, 0.5, 2.5]}, "iou": 0.4},
                            {"agent_state": {"position": [1.5, 0.5, 3.0]}, "iou": 0.2}]}


def row(category="chair", position=START, rotation=(0.0, 0.0, 0.0, 1.0), geodesic=4.25):
    """One published episode row, minus the scene identity the writer fills in."""
    return {"start_position": list(position), "start_rotation": list(rotation),
            "object_category": category, "goals": [],
            "info": {"geodesic_distance": geodesic}}


def _write_gz(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)


def write_release(root, protocol, scenes):
    """A miniature published release on disk, read back through the real loader.

    ``scenes`` maps a scene folder to its rows in publisher order. The rows are
    written the way habitat-lab writes them -- XYZW start rotations, emptied
    per-episode goals, a de-duplicated ``goals_by_category`` -- so the episodes
    the environment serves come out of the same parsing a real run uses.
    """
    root = Path(root)
    episodes_dir, scenes_dir = root / "episodes" / protocol.split, root / "scenes"
    table = {name: index for index, name in enumerate(CATEGORIES)}
    _write_gz(episodes_dir / ("%s.json.gz" % protocol.split),
              {"episodes": [], "category_to_task_category_id": table})
    for folder, rows in scenes.items():
        stem = folder.split("-", 1)[1]
        relative = "%s/%s/%s/%s.basis.glb" % (protocol.scene_root_name, protocol.split,
                                              folder, stem)
        mesh = scenes_dir / relative
        mesh.parent.mkdir(parents=True, exist_ok=True)
        for path in (mesh, mesh.with_name("%s.basis.navmesh" % stem)):
            path.write_bytes(b"an asset the loader only proves is present")
        written = [dict(entry, scene_id="data/scene_datasets/" + relative,
                        episode_id=str(index)) for index, entry in enumerate(rows)]
        _write_gz(episodes_dir / "content" / ("%s.json.gz" % stem),
                  {"episodes": written, "category_to_task_category_id": table,
                   "goals_by_category": {"%s.basis.glb_%s" % (stem, entry["object_category"]):
                                         [goal()] for entry in written}})
    return HM3DDataset(episodes_dir, scenes_dir, protocol, full=False)


class FakeSimulator:
    """habitat-sim replaced by exact kinematics on the protocol's own action spec.

    The pose it reports is the published start put through the bridge's frame
    convention, so the environment's check that the start survived the reset
    means here what it means in a real run. ``snap_m`` moves that start, as a
    navmesh disagreeing with the episodes would; ``climb_m`` lifts every
    forward step, as a stair does.
    """

    def __init__(self, protocol, *, snap_m=0.0, climb_m=0.0):
        self.actions = protocol.actions()
        self.camera = protocol.camera()
        self.snap_m, self.climb_m = snap_m, climb_m
        self.resets, self.actions_taken, self.closes = [], [], 0
        self._scene = None
        self._pose = None

    @property
    def pathfinder(self):
        """A token standing in for the loaded scene's navmesh."""
        if self._scene is None:
            raise EnvContractError("Simulator not reset; there is no navmesh yet")
        return "navmesh of %s" % self._scene

    def reset(self, scene_path, navmesh_path, position, rotation_wxyz, seed):
        self._scene = str(scene_path)
        self.resets.append(SimpleNamespace(
            scene=str(scene_path), navmesh=str(navmesh_path),
            position=tuple(float(v) for v in position),
            rotation=tuple(float(v) for v in rotation_wxyz), seed=seed))
        x, y, z = (float(v) for v in position)
        self._pose = AgentPose(
            x=-z + self.snap_m, y=-x, z=y,
            yaw=2.0 * math.atan2(rotation_wxyz[2], rotation_wxyz[0]))
        return self._frame()

    def step(self, action):
        if self._scene is None or not self.actions.allows(action):
            raise EnvContractError("Simulator not reset, or unsupported action")
        self.actions_taken.append(action)
        pose, spec = self._pose, self.actions
        if action == A.MOVE_FORWARD:
            pose = replace(pose, x=pose.x + spec.forward_step_m * math.cos(pose.yaw),
                           y=pose.y + spec.forward_step_m * math.sin(pose.yaw),
                           z=pose.z + self.climb_m)
        elif action in (A.TURN_LEFT, A.TURN_RIGHT):
            sign = 1.0 if action == A.TURN_LEFT else -1.0
            pose = replace(pose, yaw=pose.yaw + sign * spec.turn_angle_rad)
        elif action in (A.LOOK_UP, A.LOOK_DOWN):
            sign = 1.0 if action == A.LOOK_DOWN else -1.0
            pose = replace(pose, camera_pitch=pose.camera_pitch + sign * spec.tilt_angle_rad)
        self._pose = pose
        return self._frame()

    def _frame(self):
        size = (self.camera.intrinsics.height, self.camera.intrinsics.width)
        return (np.full(size + (3,), len(self.actions_taken) % 256, np.uint8),
                np.full(size, 1.5, np.float32), self._pose)

    def close(self):
        self.closes += 1
        self._scene = None


class ProvenancedSimulator(FakeSimulator):
    """A bridge that can say which navmesh it loaded, as the real one can."""

    def navmesh_provenance(self):
        return {"mode": "agent", "navigable_area_m2": 12.5, "islands": 1}


class FakeDistance:
    """The geodesic replaced by a schedule: the first query is ``l``, the rest ``dT``.

    The environment measures once at reset and once at the end, so two numbers
    say exactly where an episode started and where it finished -- including the
    corners (a start on the goal, an unreachable start) that no navmesh can be
    talked into producing on demand. The last value repeats, so an evaluator's
    diagnostic query in between changes nothing.
    """

    def __init__(self, pathfinder, view_points, schedule):
        self.pathfinder = pathfinder
        self.view_point_count = len(np.asarray(view_points))
        self.schedule = tuple(schedule)
        self.queries = []

    def distance(self, position):
        self.queries.append(tuple(float(v) for v in position))
        return self.schedule[min(len(self.queries), len(self.schedule)) - 1]


def scripted_distances(*schedules):
    """A ``distance_factory`` giving each goal set, in first-measured order, the next schedule."""
    made = []

    def factory(pathfinder, view_points):
        made.append(FakeDistance(pathfinder, view_points,
                                 schedules[min(len(made), len(schedules) - 1)]))
        return made[-1]

    factory.made = made
    return factory


def rig(root, *, scenes=None, schedules=((4.0, 4.0),), simulator=None, **options):
    """A written release, a fake simulator and a scripted geodesic behind one env."""
    dataset = write_release(root, PROTOCOL, scenes or {SCENE: [row()]})
    distances = scripted_distances(*schedules)
    simulator = FakeSimulator(PROTOCOL) if simulator is None else simulator
    env = HM3DEnv(dataset, PROTOCOL, simulator=simulator,
                  distance_factory=distances, **options)
    return SimpleNamespace(env=env, dataset=dataset, simulator=simulator,
                           distances=distances)


def play(env, episode_id, actions):
    """Run one episode, accounting ``p`` from the observed poses as the harness does."""
    _, observation = env.reset(episode_id)
    positions = [observation.pose.position()]
    for action in actions:
        observation = env.step(action)
        positions.append(observation.pose.position())
    return env.measure(), path_length_3d(positions)


@pytest.mark.parametrize("final_m,succeeds", [(0.09, True), (0.1, False), (0.11, False)],
                         ids=["inside", "on_the_radius", "outside"])
def test_a_stop_succeeds_only_strictly_inside_the_success_radius(tmp_path, final_m, succeeds):
    """habitat-lab's ObjectNav asks ``distance < SUCCESS_DISTANCE``. An adapter
    written with ``<=`` grants the radius itself, which lifts the success rate
    on precisely the marginal episodes a method is judged by."""
    fixture = rig(tmp_path, schedules=((1.0, final_m),))
    measurement, observed = play(fixture.env, EPISODE, [A.MOVE_FORWARD, A.STOP])
    assert measurement.success is succeeds
    assert measurement.stop_called and measurement.termination == TERMINATION_STOP
    assert measurement.final_distance_to_goal_m == final_m
    assert score_episode(measurement, observed).success is succeeds


def test_reaching_the_goal_without_stop_is_not_a_success(tmp_path):
    """Habitat grants nothing for arriving: the agent has to say it has arrived.
    The episode below ends standing on a view point with a perfect SoftSPL and
    no success at all, which is why success must never be inferred from ``dT``."""
    fixture = rig(tmp_path, schedules=((1.0, 0.0),))
    measurement, observed = play(fixture.env, EPISODE, [A.MOVE_FORWARD] * PROTOCOL.max_steps)
    assert measurement.success is False and measurement.stop_called is False
    assert measurement.termination == TERMINATION_STEP_LIMIT
    assert measurement.final_distance_to_goal_m == 0.0
    assert measurement.native_metrics["soft_spl"] == pytest.approx(1.0)
    assert score_episode(measurement, observed).spl == 0.0


def test_the_shortest_path_is_measured_at_reset_not_read_off_the_published_row(tmp_path):
    """habitat-lab's SPL reference is ``DistanceToGoal`` at reset, measured to
    the goal view points on the navmesh the episode runs on. The row's
    ``info.geodesic_distance`` was measured to the object instead, so scoring
    against it would rescale every SPL in the split by a few per cent and look
    entirely normal."""
    fixture = rig(tmp_path, scenes={SCENE: [row(geodesic=7.5)]}, schedules=((3.0, 0.05),))
    measurement, _ = play(fixture.env, EPISODE, [A.STOP])
    assert measurement.shortest_path_m == 3.0 == measurement.start_distance_to_goal_m
    assert measurement.info["published_geodesic_m"] == 7.5
    assert measurement.info["view_points"] == 2
    assert fixture.distances.made[0].queries[0] == START


def test_the_path_length_counts_travel_in_three_dimensions_and_nothing_else(tmp_path):
    """``p`` is habitat-lab's ``_agent_episode_distance``: 3-D chords between
    successive base positions. Counting a turn or a tilt would deflate SPL on
    exactly the episodes that looked around most, accounting it in the plane
    would lose every stair, and carrying it across a reset would charge each
    episode for the one before it."""
    fixture = rig(tmp_path, schedules=((2.0, 2.0),),
                  simulator=FakeSimulator(PROTOCOL, climb_m=0.1))
    climbed, observed = play(fixture.env, EPISODE,
                             [A.TURN_LEFT, A.LOOK_DOWN, A.MOVE_FORWARD, A.STOP])
    assert climbed.path_length_m == pytest.approx(math.hypot(0.25, 0.1))
    assert climbed.path_length_m == pytest.approx(observed)
    assert climbed.steps == 4

    still, _ = play(fixture.env, EPISODE, [A.TURN_RIGHT, A.LOOK_UP, A.STOP])
    assert still.path_length_m == 0.0


def test_the_native_metrics_reproduce_the_shared_scorer_on_the_same_episode(tmp_path):
    """The native block exists so that a drift between habitat-lab's arithmetic
    and the harness's raises instead of being filed as a result. It is only
    worth carrying while all four values survive ``score_episode``."""
    fixture = rig(tmp_path, schedules=((0.4, 0.05),))
    measurement, observed = play(fixture.env, EPISODE,
                                 [A.MOVE_FORWARD, A.MOVE_FORWARD, A.STOP])
    score = score_episode(measurement, observed)
    assert score.native_checked == NATIVE_KEYS
    assert measurement.native_metrics["spl"] == pytest.approx(score.spl) == pytest.approx(0.8)
    assert measurement.native_metrics["soft_spl"] == pytest.approx(score.soft_spl)
    assert measurement.native_metrics["soft_spl"] == pytest.approx(0.875 * 0.8)


def test_the_native_soft_spl_is_omitted_rather_than_invented_when_the_start_is_on_the_goal(tmp_path):
    """habitat-lab divides by ``d0`` unguarded, so on a start already at a view
    point it has no SoftSPL to report. Filling the key with this harness's own
    convention would leave the cross-check comparing the harness with itself,
    which is the one thing it must not do."""
    fixture = rig(tmp_path, schedules=((0.0, 0.0),))
    measurement, observed = play(fixture.env, EPISODE, [A.MOVE_FORWARD, A.STOP])
    assert measurement.success is True and measurement.shortest_path_m == 0.0
    assert "soft_spl" not in measurement.native_metrics
    assert measurement.native_metrics["spl"] == 0.0
    assert score_episode(measurement, observed).native_checked == (
        "success", "spl", "distance_to_goal")


def test_the_public_episode_carries_no_privileged_metadata(tmp_path):
    """The agent receives the episode at reset, and a goal position or a
    geodesic in its metadata would make SR and SPL meaningless while failing
    nothing. The harness's tripwire only helps while the adapter keeps the
    mapping empty and leaves the privileged numbers on the evaluator's side."""
    fixture = rig(tmp_path)
    episode, observation = checked_reset(fixture.env, EPISODE)
    assert episode.metadata == {}
    assert (episode.benchmark, episode.split) == ("hm3d_v1", "val")
    assert (episode.scene_id, episode.target_category) == (SCENE, "chair")
    assert episode.max_steps == PROTOCOL.max_steps
    assert observation.step == 0 and observation.target_category == "chair"
    assert set(fixture.env.evaluation_diagnostics()) == {
        "distance_to_goal_m", "path_length_m", "shortest_path_m",
        "success_distance_m", "published_geodesic_m", "view_points"}


def test_reset_starts_at_the_published_pose_and_refuses_a_navmesh_that_moves_it(tmp_path):
    """A start silently snapped onto a navmesh is a different episode from the
    published one, and its ``l`` is a different number. The rotation matters as
    much: forwarded in the publisher's XYZW order it would face the agent
    somewhere else and nothing downstream could tell."""
    turned = (0.0, 0.70710678118654752, 0.0, 0.70710678118654752)
    fixture = rig(tmp_path, scenes={SCENE: [row(rotation=turned), row(category="bed")]})
    _, observation = fixture.env.reset(EPISODE)
    started = fixture.simulator.resets[0]
    assert started.position == START
    assert started.rotation == (0.70710678118654752, 0.0, 0.70710678118654752, 0.0)
    assert observation.pose.position() == START_ENU
    assert observation.pose.yaw == pytest.approx(math.pi / 2)

    fixture.env.reset(EPISODE)
    fixture.env.reset(SCENE + "/000001")
    seeds = [reset.seed for reset in fixture.simulator.resets]
    assert seeds[0] == seeds[1] != seeds[2]

    snapped = rig(tmp_path / "snapped", simulator=FakeSimulator(PROTOCOL, snap_m=0.01))
    with pytest.raises(EnvContractError, match="published start"):
        snapped.env.reset(EPISODE)


def test_a_start_with_no_reachable_goal_is_refused_at_reset(tmp_path):
    """The harness refuses a non-finite ``l``, and a run would stop at that
    episode again on every resume, so the refusal has to name the preflight
    that exists to find it instead of reaching the scorer as a NaN."""
    fixture = rig(tmp_path, schedules=((math.inf,),))
    with pytest.raises(EnvContractError, match="reachable"):
        fixture.env.reset(EPISODE)


def test_an_excluded_episode_is_neither_served_nor_resettable(tmp_path):
    """Excluding an episode is a recorded decision about which split was scored;
    serving it anyway would put an unscoreable episode back into the run and
    make the results directory disagree with the manifest beside it."""
    fixture = rig(tmp_path, scenes={SCENE: [row(), row(category="bed")]},
                  excluded_episode_ids=(EPISODE,))
    assert fixture.env.episode_ids() == (SCENE + "/000001",)
    with pytest.raises(EnvContractError, match="excluded"):
        fixture.env.reset(EPISODE)


def test_validate_starts_finds_the_unreachable_episodes_before_the_run_does(tmp_path):
    """Preflight measures ``l`` for the whole selection one scene at a time,
    because the alternative is discovering an unreachable start at episode 900
    and again on every resume. The navmesh provenance and the agreement with
    the publisher's own geodesic are what say afterwards which mesh produced
    the split's SPL -- reported, never asserted, since a release generated
    under another definition must still be runnable."""
    scenes = {SCENE: [row(geodesic=3.0005), row(category="bed", geodesic=9.0)]}
    fixture = rig(tmp_path, scenes=scenes, schedules=((3.0,), (math.inf,)),
                  simulator=ProvenancedSimulator(PROTOCOL))
    assert fixture.env.geodesic_agreement() is None

    seen = []
    distances = fixture.env.validate_starts(progress=lambda *args: seen.append(args))
    assert distances == {EPISODE: 3.0, SCENE + "/000001": math.inf}
    assert fixture.env.unreachable_episode_ids == (SCENE + "/000001",)
    assert seen == [(SCENE, 1, 1)]
    assert fixture.env.navmesh_by_scene[SCENE]["mode"] == "agent"

    agreement = fixture.env.geodesic_agreement()
    assert agreement["episodes"] == 1 and agreement["within_1mm"] == 1
    assert agreement["max_abs_m"] == pytest.approx(0.0005)
    assert agreement["median_abs_m"] == pytest.approx(0.0005)

    plain = rig(tmp_path / "plain", schedules=((3.0,),))
    plain.env.validate_starts()
    assert plain.env.navmesh_by_scene == {}


def test_an_episode_ends_on_stop_or_at_the_budget_and_accepts_nothing_after(tmp_path):
    """An adapter that kept stepping after STOP would let an agent spend its
    budget past the end of the episode; one that did not end at the budget
    would let it run forever; one that forgot to clear its STOP flag would end
    every later episode at step 0. Each of the three is a scored number."""
    fixture = rig(tmp_path)
    fixture.env.reset(EPISODE)
    with pytest.raises(EnvContractError, match="after episode end"):
        fixture.env.measure()
    assert fixture.env.episode_over is False
    fixture.env.step(A.STOP)
    assert fixture.env.episode_over is True
    with pytest.raises(EnvContractError, match="not running"):
        fixture.env.step(A.MOVE_FORWARD)

    fixture.env.reset(EPISODE)
    assert fixture.env.episode_over is False
    for _ in range(PROTOCOL.max_steps):
        fixture.env.step(A.TURN_LEFT)
    assert fixture.env.episode_over is True
    with pytest.raises(EnvContractError, match="not running"):
        fixture.env.step(A.STOP)


def test_the_distance_is_measured_to_every_view_point_of_every_goal():
    """habitat-lab's ``DistanceToGoal`` with ``DISTANCE_TO: VIEW_POINTS`` takes
    one multi-goal path over the view points of all of an episode's goals.
    Measuring to one goal, or to the object centres, silently asks a different
    question than the benchmark's."""
    goals = [{"view_points": [{"agent_state": {"position": [0.0, 0.1, 0.2]}},
                              {"agent_state": {"position": [1.0, 1.1, 1.2]}}]},
             {"view_points": [{"agent_state": {"position": [2.0, 2.1, 2.2]}}]}]
    points = view_point_positions(goals)
    assert points.dtype == np.float32 and points.shape == (3, 3)
    np.testing.assert_allclose(points, [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2],
                                        [2.0, 2.1, 2.2]], rtol=1e-6)


@pytest.mark.parametrize("goals,message", [
    ([], "No goal view points"),
    ([{"view_points": []}], "No goal view points"),
    ([{"position": [0.0, 0.0, 0.0]}], "No goal view points"),
    ([{"view_points": [{"agent_state": {"position": [0.0, 1.0]}}]}], "three finite numbers"),
    ([{"view_points": [{"agent_state": {"position": [0.0, 1.0, float("nan")]}}]}],
     "three finite numbers"),
], ids=["no_goals", "no_view_points", "only_a_centre", "two_numbers", "not_finite"])
def test_a_goal_set_without_usable_view_points_cannot_be_scored(goals, message):
    """Falling back to the object centre would move the goal by the object's own
    radius and change every success on the split, so an unusable goal set has to
    stop the episode rather than be worked around."""
    with pytest.raises(ValueError, match=message):
        view_point_positions(goals)


@pytest.mark.parametrize("points", [
    np.zeros((0, 3), np.float32), np.zeros((2, 2), np.float32),
    np.array([[0.0, 0.0, np.inf]], np.float32), np.zeros(3, np.float32),
], ids=["empty", "two_columns", "not_finite", "one_dimensional"])
def test_view_point_distance_refuses_a_goal_set_it_cannot_measure(points):
    """A malformed ends array does not raise inside habitat-sim; it measures to
    whatever it was given, so it is refused where it is built."""
    with pytest.raises(ValueError):
        ViewPointDistance(object(), points)


class StubPath:
    """``habitat_sim.MultiGoalShortestPath``: ends set once, start overwritten per query."""

    geodesic_distance = 0.0


class StubPathfinder:
    """A pathfinder returning scripted geodesics and recording what it was asked."""

    def __init__(self, values):
        self.values = list(values)
        self.queries = []

    def find_path(self, path):
        self.queries.append((path, tuple(float(v) for v in path.requested_start)))
        path.geodesic_distance = self.values.pop(0)
        return True


def test_one_multi_goal_path_is_reused_and_an_unreachable_goal_reads_as_infinity(monkeypatch):
    """habitat-lab caches one ``MultiGoalShortestPath`` per episode and moves
    only its start; rebuilding it per query, or re-requesting the ends, is the
    kind of change that costs an hour a run and shows up nowhere in the score.
    An unreachable start comes back as NaN from habitat-sim, and a NaN that
    reached SPL would poison the mean of the whole split instead of the episode.

    ``habitat_sim`` is not installed here, so a stub module stands in for it:
    the calls this class makes into the pathfinder are the definition of the
    measure, and they are worth pinning even without the real library.
    """
    module = types.ModuleType("habitat_sim")
    module.MultiGoalShortestPath = StubPath
    monkeypatch.setitem(sys.modules, "habitat_sim", module)

    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], np.float32)
    finder = StubPathfinder([2.5, float("nan")])
    distance = ViewPointDistance(finder, points)
    assert distance.view_point_count == 2
    assert distance.distance((3.0, 0.0, 4.0)) == 2.5
    assert distance.distance((3.0, 0.0, 5.0)) == math.inf

    first, second = finder.queries
    assert first[0] is second[0]
    np.testing.assert_array_equal(first[0].requested_ends, points)
    assert (first[1], second[1]) == ((3.0, 0.0, 4.0), (3.0, 0.0, 5.0))


class Walker(ObjNavAgent):
    """Plays one script in every episode, repeating its last action."""

    name = "walker"

    def __init__(self, script):
        self.script = tuple(script)
        self._index = 0

    def reset(self, episode):
        self._index = 0

    def act(self, observation):
        action = self.script[min(self._index, len(self.script) - 1)]
        self._index += 1
        return AgentDecision(action)


def test_the_adapter_satisfies_the_whole_environment_contract_under_the_runner(tmp_path):
    """Satisfying ``ObjNavEnv`` method by method is not the same as satisfying
    its contract step by step. This drives the adapter through the harness that
    will really run the benchmark, under the protocol's own evaluation
    settings, so the reset, step and measurement clauses, the kinematic check
    against the published action geometry and the cross-check between the
    environment's ``p`` and the poses the runner observed all have to hold at
    once -- across two scenes, so the scene reload and the distance cache it
    clears are part of what is proven."""
    fixture = rig(tmp_path, scenes={SCENE: [row()], OTHER_SCENE: [row(category="bed")]},
                  schedules=((0.2, 0.05), (2.0, 5.0)))
    assert fixture.env.episode_ids() == (EPISODE, OTHER_EPISODE)
    settings = PROTOCOL.evaluation_settings()
    with MetricsLogger(tmp_path / "run", {"protocol": PROTOCOL.protocol_id}) as logger:
        summary = run_benchmark(
            fixture.env, Walker([A.TURN_LEFT, A.LOOK_DOWN, A.MOVE_FORWARD, A.STOP]),
            logger=logger, require_stop_for_success=settings.require_stop_for_success,
            path_length_dimension=settings.path_length_dimension,
            path_length_epsilon_m=settings.path_length_epsilon_m,
            path_tolerance_m=settings.path_tolerance_m, kinematics=settings.kinematics,
            on_agent_error=settings.on_agent_error)

    assert (summary.benchmark, summary.split, summary.agent) == ("hm3d_v1", "val", "walker")
    assert summary.overall.n_episodes == 2 and summary.overall.success_rate == 0.5
    assert summary.overall.spl == pytest.approx(0.4)
    rows = [json.loads(line) for line
            in (tmp_path / "run" / "episodes.jsonl").read_text().splitlines()]
    assert [entry["episode_id"] for entry in rows] == [EPISODE, OTHER_EPISODE]
    assert [entry["termination"] for entry in rows] == ["stop", "stop"]
    assert rows[0]["spl"] == pytest.approx(0.8) and rows[1]["spl"] == 0.0
    assert rows[0]["path_length_m"] == pytest.approx(rows[0]["observed_path_length_m"])

    made = fixture.distances.made
    assert len(made) == 2 and made[0].pathfinder != made[1].pathfinder
    fixture.env.close()
    assert fixture.simulator.closes == 1


# -- the unmoved-agent short circuit ---------------------------------------

class CountingPathfinder:
    """Records how many multi-goal queries actually reach the navmesh."""

    def __init__(self, distances):
        self.distances = list(distances)
        self.calls = 0

    def find_path(self, path):
        path.geodesic_distance = self.distances[min(self.calls, len(self.distances) - 1)]
        self.calls += 1
        return True


class MutablePath:
    """A stand-in for habitat_sim.MultiGoalShortestPath."""

    requested_start = None
    requested_ends = None
    geodesic_distance = float("inf")


def test_a_turn_reuses_the_previous_distance_instead_of_re_querying(monkeypatch):
    """habitat-lab recomputes only when the agent moved, and a goal set here
    holds over a thousand view points: querying it on every one of 500 actions
    would dominate the run, and would also not be what the reference measures.
    """
    import sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.geodesic as module

    fake = types.SimpleNamespace(MultiGoalShortestPath=MutablePath)
    monkeypatch.setitem(sys.modules, "habitat_sim", fake)
    pathfinder = CountingPathfinder([4.0, 2.5])
    distance = module.ViewPointDistance(pathfinder, [[1.0, 0.0, 1.0]])

    here = (0.0, 0.0, 0.0)
    assert distance.distance(here) == 4.0
    assert distance.distance(here) == 4.0           # a turn: nothing moved
    assert distance.distance((0.0, 0.0, 5e-5)) == 4.0   # inside habitat's atol
    assert pathfinder.calls == distance.queries == 1
    assert distance.distance((0.25, 0.0, 0.0)) == 2.5   # a forward step
    assert pathfinder.calls == distance.queries == 2
