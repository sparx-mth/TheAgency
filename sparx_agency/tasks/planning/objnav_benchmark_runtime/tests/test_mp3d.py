"""The MP3D adapter without Matterport3D, habitat-sim or a GPU: its table, its protocol, its loader and its arithmetic.

Every mistake this adapter can make produces a plausible number instead of a
crash: a success radius compared with ``<=`` instead of ``<``, an ``l`` read
off the publisher's row rather than measured, a ``p`` that counts a turn, a
start quietly moved by a navmesh that does not match the episodes. The two
seams ``MP3DEnv`` offers -- ``simulator=`` and ``distance_factory=`` -- put
known quantities on both sides of the measurement so the corners can be
pinned on geometry a test chose.

The release fixture writes the published layout (``val.json.gz`` with the
category tables, ``content/<scene>.json.gz`` with XYZW rotations and either
``goals_by_category`` or inline ``goals``), so the episodes the environment
serves come out of the same parsing a real run uses.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import gzip
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.core.planning.objnav.labels.datasets.mp3d import (
    CATEGORIES, mp3d_label_mapper)
from sparx_agency.core.planning.objnav.labels.registry import default_label_mapper_registry
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.measurement import (
    NATIVE_KEYS, TERMINATION_STEP_LIMIT, TERMINATION_STOP)
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.core.planning.object_search.perception.doors import DOOR_LABELS
from sparx_agency.core.planning.object_search.perception.projection import STAIR_LABELS
from sparx_agency.tasks.planning.objnav_benchmark.scoring import path_length_3d, score_episode
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import habitat_position
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.dataset import (
    MP3DDataset, published_counts, xyzw_to_wxyz)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.env import MP3DEnv
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.protocol import (
    APEXNAV_SUCCESS_DISTANCE_M, MP3D_V1, PUBLISHED_VAL, MP3DProtocol, protocol_for)

A = DiscreteAction

#: Two validation scenes whose shards sort in the order written here.
SCENE, OTHER_SCENE = "2azQ1b91cZZ", "8194nk5LbLH"
EPISODE, OTHER_EPISODE = SCENE + "/000000", OTHER_SCENE + "/000000"

#: The published protocol at an 8x6 frame and a four-action budget: nothing
#: here reads a pixel, and a budget an agent can spend makes the step-limit
#: corner of the success rule reachable.
PROTOCOL = replace(MP3D_V1, width=8, height=6, max_steps=4)

#: Habitat's published start for every row below (+Y up), and the ENU pose the
#: bridge's convention turns it into: ``x = -z``, ``y = -x``, ``z = y``.
START = (1.0, 0.5, 2.0)
START_ENU = (-2.0, -1.0, 0.5)


def goal(category="chair"):
    """One evaluator-only goal row, shaped like the publisher's: two view points."""
    return {"object_id": 7, "object_category": category, "position": [1.0, 0.5, 3.0],
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


def write_release(root, protocol=PROTOCOL, scenes=None, *, inline_goals=False,
                  categories=CATEGORIES, scene_prefix="", **loader):
    """A miniature published release on disk, read back through the real loader.

    ``scenes`` maps a scene hash to its rows in publisher order. ``inline_goals``
    writes the pre-de-duplication shape (every row carrying its ``goals``,
    no ``goals_by_category``), which the loader must read identically.
    """
    root = Path(root)
    scenes = scenes if scenes is not None else {SCENE: [row()]}
    episodes_dir, scenes_dir = root / "episodes" / protocol.split, root / "scenes"
    table = {name: index for index, name in enumerate(categories)}
    _write_gz(episodes_dir / ("%s.json.gz" % protocol.split),
              {"episodes": [], "category_to_task_category_id": table,
               "category_to_mp3d_category_id": {name: index + 3 for name, index in table.items()}})
    for scene, rows in scenes.items():
        relative = "%s/%s/%s.glb" % (protocol.scene_root_name, scene, scene)
        mesh = scenes_dir / relative
        mesh.parent.mkdir(parents=True, exist_ok=True)
        for path in (mesh, mesh.with_name(scene + ".navmesh")):
            path.write_bytes(b"an asset the loader only proves is present")
        goals = {"%s.glb_%s" % (scene, entry["object_category"]): [goal(entry["object_category"])]
                 for entry in rows}
        written = []
        for index, entry in enumerate(rows):
            item = dict(entry, scene_id=scene_prefix + relative, episode_id=str(index))
            if inline_goals:
                item["goals"] = goals["%s.glb_%s" % (scene, entry["object_category"])]
            written.append(item)
        payload = {"episodes": written, "category_to_task_category_id": table}
        if not inline_goals:
            payload["goals_by_category"] = goals
        _write_gz(episodes_dir / "content" / ("%s.json.gz" % scene), payload)
    loader.setdefault("full", False)
    return MP3DDataset(episodes_dir, scenes_dir, protocol, **loader)


class FakeSimulator:
    """habitat-sim replaced by exact kinematics on the protocol's own action spec.

    ``snap_m`` moves the start as a navmesh disagreeing with the episodes
    would; ``climb_m`` lifts every forward step, as a stair does.
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
        self._pose = AgentPose(x=-z + self.snap_m, y=-x, z=y,
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
        return {"mode": "agent", "navigable_area_m2": 30.6, "islands": 1}


class FakeDistance:
    """The geodesic replaced by a schedule: the first query is ``l``, the rest ``dT``."""

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
    dataset = write_release(root, PROTOCOL, scenes)
    distances = scripted_distances(*schedules)
    simulator = FakeSimulator(PROTOCOL) if simulator is None else simulator
    env = MP3DEnv(dataset, PROTOCOL, simulator=simulator,
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


# --- the published vocabulary -------------------------------------------------

def test_the_table_is_the_published_21_categories_and_is_registered():
    mapper = mp3d_label_mapper()
    assert len(CATEGORIES) == 21
    for category in CATEGORIES:
        target = mapper.target_labels(category)
        assert target.category == category and target.detector_prompts
    registry = default_label_mapper_registry()
    assert "mp3d" in registry
    assert len(registry.create("mp3d").vocabulary()) == len(mapper.vocabulary())


def test_the_vocabulary_carries_the_door_and_stair_prompts_the_method_needs():
    vocabulary = set(mp3d_label_mapper().vocabulary())
    assert DOOR_LABELS.issubset(vocabulary)
    assert STAIR_LABELS.issubset(vocabulary)


def test_a_desk_counts_as_the_table_goal_rather_than_as_room_context():
    """mpcat40 has no desk: MP3D annotates desks under ``table``. A desk prompt
    outside the table row would make every desk a non-goal the detector reports."""
    mapper = mp3d_label_mapper()
    assert mapper.target_labels("table").accepts("desk")
    assert "desk" in mapper.vocabulary()


def test_mp3d_goal_words_are_not_context_words():
    """A context prompt equal to a goal prompt would be scored as the goal."""
    mapper = mp3d_label_mapper()
    goals = set()
    for category in CATEGORIES:
        goals |= set(mapper.target_labels(category).accept_labels)
    from sparx_agency.core.planning.objnav.labels.datasets.mp3d import CONTEXT_VOCABULARY
    from sparx_agency.core.common.label_match import normalize_label
    assert not goals & {normalize_label(word) for word in CONTEXT_VOCABULARY}


# --- the published protocol ---------------------------------------------------

def test_the_protocol_is_habitat_labs_objectnav_mp3d_configuration():
    p = MP3D_V1
    assert (p.max_steps, p.width, p.height, p.hfov_deg) == (500, 640, 480, 79.0)
    assert (p.camera_height_m, p.agent_height_m, p.agent_radius_m) == (0.88, 0.88, 0.18)
    assert (p.min_depth_m, p.max_depth_m) == (0.5, 5.0)
    assert (p.forward_step_m, p.turn_angle_deg, p.tilt_angle_deg) == (0.25, 30.0, 30.0)
    assert p.success_distance_m == 0.1 and p.distance_to == "view_points"
    assert p.allow_sliding is False and p.navmesh == "agent"
    assert p.benchmark == "mp3d_v1" and p.protocol_id == "mp3d-objectnav-v1-val/2"
    assert PUBLISHED_VAL["v1"] == {"scenes": 11, "episodes": 2195}
    assert protocol_for("v1") is MP3D_V1
    with pytest.raises(KeyError):
        protocol_for("v2")


def test_the_harness_options_are_the_habitat_lab_rules_not_a_relaxation():
    settings = MP3D_V1.evaluation_settings()
    assert settings.require_stop_for_success is True
    assert settings.path_length_dimension == "3d"
    assert settings.path_length_epsilon_m == 0.0
    # The two bounds the recomputed navmesh needs are HM3D's, derived from habitat-sim, not from a dataset.
    assert settings.kinematics.climb_m == 0.85 and settings.kinematics.reproject_m == 0.15


def test_the_protocol_hands_the_method_the_same_camera_and_actions_as_the_other_benchmarks():
    """One algorithm on every benchmark: the pitch bounds and the depth-hole
    rule the Gibson and HM3D protocols carry are here too, read from the spec."""
    spec = MP3D_V1.actions()
    assert (spec.min_pitch_deg, spec.max_pitch_deg) == (-30.0, 60.0)
    assert len(spec.actions) == 6
    camera = MP3D_V1.camera()
    assert camera.intrinsics.width == 640 and camera.height_m == 0.88
    holes = MP3D_V1.depth_holes()
    assert holes.enabled and holes.max_fraction == 0.5


def test_a_protocol_that_changes_a_number_changes_its_identity():
    with pytest.raises(ValueError):
        MP3DProtocol(dataset_version="v3", protocol_id="x", benchmark="y")
    with pytest.raises(ValueError):
        replace(MP3D_V1, distance_to="object")
    with pytest.raises(ValueError):
        replace(MP3D_V1, navmesh="whatever-is-loaded")
    loose = MP3D_V1.with_success_distance(APEXNAV_SUCCESS_DISTANCE_M)
    assert loose.success_distance_m == 0.2 and loose.protocol_id.endswith("+success0.2m")
    train = MP3D_V1.with_split("train")
    assert train.protocol_id == "mp3d-objectnav-v1-train/2" and train.split == "train"
    assert published_counts(train) == {} and published_counts(MP3D_V1) == PUBLISHED_VAL["v1"]


# --- the published release ----------------------------------------------------

def test_episodes_are_scene_qualified_and_keep_the_publishers_own_id(tmp_path):
    dataset = write_release(tmp_path, scenes={SCENE: [row(), row("bed")],
                                              OTHER_SCENE: [row("sofa")]})
    assert dataset.episode_ids() == (EPISODE, SCENE + "/000001", OTHER_EPISODE)
    first = dataset.episodes[EPISODE]
    assert first.scene_key == SCENE and first.source_episode_id == "0"
    assert first.scene_relative_path == "mp3d/%s/%s.glb" % (SCENE, SCENE)
    assert first.goals_key == "%s.glb_chair" % SCENE
    assert dataset.scene_counts == {SCENE: 2, OTHER_SCENE: 1}


def test_the_published_xyzw_rotation_is_converted_for_habitat_sim(tmp_path):
    quarter = (0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))  # XYZW
    dataset = write_release(tmp_path, scenes={SCENE: [row(rotation=quarter)]})
    assert dataset.episodes[EPISODE].start_rotation_wxyz == pytest.approx(
        (quarter[3], quarter[0], quarter[1], quarter[2]))
    assert xyzw_to_wxyz((0.0, 0.0, 0.0, 1.0)) == (1.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        xyzw_to_wxyz((0.0, 0.0, 0.0, 2.0))


def test_inline_goals_and_the_habitat_lab_prefix_are_read_as_the_published_shape(tmp_path):
    """habitat-lab de-duplicates inline goals at load and strips
    ``data/scene_datasets/``; a release in either shape is the same release."""
    dataset = write_release(tmp_path, scenes={SCENE: [row(), row()]}, inline_goals=True,
                            scene_prefix="data/scene_datasets/")
    episode = dataset.episodes[EPISODE]
    assert episode.scene_relative_path == "mp3d/%s/%s.glb" % (SCENE, SCENE)
    assert len(dataset.goals_for(episode)) == 1
    assert dataset.goals_for(episode)[0]["view_points"][0]["agent_state"]["position"] == [1.0, 0.5, 2.5]


def test_a_release_with_another_vocabulary_is_refused(tmp_path):
    with pytest.raises(ValueError, match="21 MP3D ObjectNav v1 goals"):
        write_release(tmp_path, categories=CATEGORIES[:-1] + ("desk",))


def test_a_shard_holding_another_scenes_episodes_is_refused(tmp_path):
    dataset_root = tmp_path / "a"
    write_release(dataset_root, scenes={SCENE: [row()]})
    shard = dataset_root / "episodes" / "val" / "content" / ("%s.json.gz" % SCENE)
    payload = json.load(gzip.open(shard, "rt"))
    payload["episodes"][0]["scene_id"] = "mp3d/%s/%s.glb" % (OTHER_SCENE, OTHER_SCENE)
    _write_gz(shard, payload)
    with pytest.raises(ValueError, match="holds an episode of scene"):
        MP3DDataset(shard.parents[1], dataset_root / "scenes", PROTOCOL, full=False)


def test_a_partial_release_cannot_pass_as_the_full_split(tmp_path):
    with pytest.raises(ValueError, match="documented as 11 scenes"):
        write_release(tmp_path, full=True)
    dataset = write_release(tmp_path / "b", full=False)
    assert len(dataset.episodes) == 1


def test_a_missing_navmesh_is_not_regenerated(tmp_path):
    dataset = write_release(tmp_path, scenes={SCENE: [row()]})
    glb, navmesh = dataset.scene_assets(dataset.episodes[EPISODE])
    assert glb.name == SCENE + ".glb" and navmesh.name == SCENE + ".navmesh"
    navmesh.unlink()
    with pytest.raises(FileNotFoundError, match="never generates one"):
        MP3DDataset(dataset.episodes_dir, dataset.scenes_dir, PROTOCOL, full=False)


def test_scene_selection_and_per_scene_caps_label_the_run_a_subset(tmp_path):
    written = write_release(tmp_path, scenes={SCENE: [row(), row(), row()], OTHER_SCENE: [row()]})
    dataset = MP3DDataset(written.episodes_dir, written.scenes_dir, PROTOCOL, full=False,
                          scenes=[OTHER_SCENE])
    assert dataset.episode_ids() == (OTHER_EPISODE,)
    capped = MP3DDataset(dataset.episodes_dir, dataset.scenes_dir, PROTOCOL, full=False,
                         episodes_per_scene=2)
    assert len(capped.episodes) == 3 and capped.scene_counts[SCENE] == 2
    with pytest.raises(ValueError):
        MP3DDataset(dataset.episodes_dir, dataset.scenes_dir, PROTOCOL, full=True, scenes=[SCENE])
    with pytest.raises(FileNotFoundError, match="No content shard"):
        MP3DDataset(dataset.episodes_dir, dataset.scenes_dir, PROTOCOL, full=False, scenes=["nope"])


def test_the_manifest_hashes_the_episode_files_and_names_the_release(tmp_path):
    dataset = write_release(tmp_path)
    manifest = dataset.manifest(hash_scenes=False)
    assert manifest["release"].startswith("MP3D ObjectNav v1 val on mp3d-habitat")
    assert manifest["episode_count"] == 1 and manifest["scene_count"] == 1
    hashed = [entry for entry in manifest["files"] if "sha256" in entry]
    assert all(entry["path"].endswith(".json.gz") for entry in hashed) and len(hashed) == 2
    assert manifest["scene_assets_hashed"] is False
    assert len(dataset.manifest()["files"]) == 4  # the two episode files, the mesh and the navmesh


# --- the success rule and the arithmetic --------------------------------------

@pytest.mark.parametrize("final_m,succeeds", [(0.09, True), (0.1, False), (0.11, False)],
                         ids=["inside", "on_the_radius", "outside"])
def test_a_stop_succeeds_only_strictly_inside_the_success_radius(tmp_path, final_m, succeeds):
    fixture = rig(tmp_path, schedules=((1.0, final_m),))
    measurement, observed = play(fixture.env, EPISODE, [A.MOVE_FORWARD, A.STOP])
    assert measurement.success is succeeds
    assert measurement.stop_called and measurement.termination == TERMINATION_STOP
    assert measurement.final_distance_to_goal_m == final_m
    assert score_episode(measurement, observed).success is succeeds


def test_reaching_the_goal_without_stop_is_not_a_success(tmp_path):
    fixture = rig(tmp_path, schedules=((1.0, 0.0),))
    measurement, observed = play(fixture.env, EPISODE, [A.MOVE_FORWARD] * PROTOCOL.max_steps)
    assert measurement.success is False and measurement.stop_called is False
    assert measurement.termination == TERMINATION_STEP_LIMIT
    assert measurement.native_metrics["soft_spl"] == pytest.approx(1.0)
    assert score_episode(measurement, observed).spl == 0.0
    assert fixture.env.episode_over


def test_the_shortest_path_is_measured_at_reset_not_read_off_the_published_row(tmp_path):
    fixture = rig(tmp_path, scenes={SCENE: [row(geodesic=7.5)]}, schedules=((3.0, 0.05),))
    measurement, _ = play(fixture.env, EPISODE, [A.STOP])
    assert measurement.shortest_path_m == 3.0 == measurement.start_distance_to_goal_m
    assert measurement.info["published_geodesic_m"] == 7.5
    assert measurement.info["view_points"] == 2
    assert measurement.info["source_episode_id"] == "0"
    assert fixture.distances.made[0].queries[0] == START


def test_the_path_length_counts_travel_in_three_dimensions_and_nothing_else(tmp_path):
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
    fixture = rig(tmp_path, schedules=((0.4, 0.05),))
    measurement, observed = play(fixture.env, EPISODE, [A.MOVE_FORWARD, A.MOVE_FORWARD, A.STOP])
    score = score_episode(measurement, observed)
    assert score.native_checked == NATIVE_KEYS
    assert measurement.native_metrics["spl"] == pytest.approx(score.spl) == pytest.approx(0.8)
    assert measurement.native_metrics["soft_spl"] == pytest.approx(0.875 * 0.8)


def test_the_native_soft_spl_is_omitted_when_the_start_is_on_the_goal(tmp_path):
    fixture = rig(tmp_path, schedules=((0.0, 0.0),))
    measurement, _ = play(fixture.env, EPISODE, [A.STOP])
    assert measurement.success is True
    # habitat-lab's SPL divides by max(l, p) unguarded: with l == p == 0 there is no ratio to report.
    assert "soft_spl" not in measurement.native_metrics and "spl" not in measurement.native_metrics
    assert measurement.native_metrics["success"] == 1.0


def test_the_episode_the_policy_sees_carries_no_goal_and_the_benchmark_identity(tmp_path):
    fixture = rig(tmp_path)
    episode, observation = fixture.env.reset(EPISODE)
    assert episode.benchmark == "mp3d_v1" and episode.split == "val"
    assert episode.target_category == "chair" and episode.max_steps == PROTOCOL.max_steps
    assert episode.metadata == {}  # a simulator without scene_structure yields none
    assert observation.pose.position() == pytest.approx(START_ENU)
    assert not hasattr(observation, "distance_to_goal_m")
    assert habitat_position(observation.pose) == pytest.approx(START)
    for name in ("goals", "view_points", "geodesic"):
        assert not hasattr(episode, name)


def test_the_simulator_is_handed_the_published_assets_and_a_reproducible_seed(tmp_path):
    fixture = rig(tmp_path, scenes={SCENE: [row()], OTHER_SCENE: [row("bed")]})
    fixture.env.reset(EPISODE)
    fixture.env.reset(OTHER_EPISODE)
    first, second = fixture.simulator.resets
    assert first.scene.endswith("mp3d/%s/%s.glb" % (SCENE, SCENE))
    assert first.navmesh.endswith("mp3d/%s/%s.navmesh" % (SCENE, SCENE))
    assert first.position == START and first.rotation == (1.0, 0.0, 0.0, 0.0)
    assert first.seed != second.seed
    fixture.env.reset(EPISODE)
    assert fixture.simulator.resets[2].seed == first.seed
    # A new scene is a new navmesh: the distance fields are rebuilt on it.
    assert len(fixture.distances.made) == 3


def test_a_simulator_that_moves_the_published_start_is_refused(tmp_path):
    fixture = rig(tmp_path, simulator=FakeSimulator(PROTOCOL, snap_m=0.01))
    with pytest.raises(EnvContractError, match="changed the published start"):
        fixture.env.reset(EPISODE)


def test_an_unreachable_start_is_refused_rather_than_scored(tmp_path):
    fixture = rig(tmp_path, schedules=((float("inf"), float("inf")),))
    with pytest.raises(EnvContractError, match="No goal view point is reachable"):
        fixture.env.reset(EPISODE)


def test_the_start_validation_finds_unreachable_starts_and_the_publishers_agreement(tmp_path):
    fixture = rig(tmp_path,
                  scenes={SCENE: [row(geodesic=4.0), row("bed", geodesic=2.0)],
                          OTHER_SCENE: [row("sofa", geodesic=1.0)]},
                  schedules=((4.0,), (float("inf"),), (1.005,)),
                  simulator=ProvenancedSimulator(PROTOCOL))
    seen = []
    distances = fixture.env.validate_starts(progress=lambda scene, done, total: seen.append((scene, done, total)))
    assert distances[EPISODE] == 4.0 and distances[OTHER_EPISODE] == 1.005
    assert math.isinf(distances[SCENE + "/000001"])
    assert fixture.env.unreachable_episode_ids == (SCENE + "/000001",)
    assert seen == [(SCENE, 1, 2), (OTHER_SCENE, 2, 2)]
    agreement = fixture.env.geodesic_agreement()
    assert agreement["episodes"] == 2 and agreement["within_1mm"] == 1
    assert agreement["max_abs_m"] == pytest.approx(0.005)
    assert set(fixture.env.navmesh_by_scene) == {SCENE, OTHER_SCENE}
    # Excluding them is explicit, and an excluded episode is never served.
    fixture.env.excluded_episode_ids = fixture.env.unreachable_episode_ids
    assert SCENE + "/000001" not in fixture.env.episode_ids()
    with pytest.raises(EnvContractError, match="excluded"):
        fixture.env.reset(SCENE + "/000001")


def test_evaluator_telemetry_reaches_the_recorder_and_not_the_observation(tmp_path):
    fixture = rig(tmp_path, schedules=((2.0, 1.5),))
    with pytest.raises(EnvContractError):
        fixture.env.evaluation_diagnostics()
    _, observation = fixture.env.reset(EPISODE)
    fixture.env.step(A.MOVE_FORWARD)
    diagnostics = fixture.env.evaluation_diagnostics()
    assert diagnostics["distance_to_goal_m"] == 1.5 and diagnostics["shortest_path_m"] == 2.0
    assert diagnostics["path_length_m"] == pytest.approx(0.25)
    assert diagnostics["success_distance_m"] == 0.1 and diagnostics["view_points"] == 2
    assert diagnostics["collisions"] is None  # the fake bridge reports none
    assert not hasattr(observation, "distance_to_goal_m")


def test_the_environment_is_closed_once_and_the_distances_released(tmp_path):
    fixture = rig(tmp_path)
    fixture.env.reset(EPISODE)
    fixture.env.close()
    assert fixture.simulator.closes == 1
    with pytest.raises(EnvContractError):
        fixture.env.measure()


def test_the_default_simulator_is_the_shared_bridge_built_from_the_protocol(tmp_path):
    """The one place MP3D's numbers meet a simulator: the bridge is built with
    the protocol's camera, actions, body, sliding rule, navmesh choice and
    depth-hole rule -- and nothing is rendered until reset()."""
    dataset = write_release(tmp_path)
    env = MP3DEnv(dataset, MP3D_V1)
    bridge = env._simulator
    assert bridge.navmesh == "agent" and bridge.allow_sliding is False
    assert (bridge.radius_m, bridge.height_m) == (0.18, 0.88)
    assert bridge.camera.max_depth_m == 5.0 and bridge.actions.forward_step_m == 0.25
    assert bridge.depth_holes.enabled and bridge.semantic is False
    assert env.target_projector is not None and env.name == "habitat/mp3d-objectnav-v1"
    env.close()


def test_the_recorded_protocol_round_trips_as_plain_data():
    recorded = asdict(MP3D_V1)
    assert recorded["navmesh"] == "agent" and recorded["benchmark"] == "mp3d_v1"
    assert MP3DProtocol(**recorded) == MP3D_V1
