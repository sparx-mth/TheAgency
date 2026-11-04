"""MP3D ground-truth perception: the mpcat40 goal table, the MP3D adapter's semantic path, and HM3D left as it was.

What is proved here, without habitat-sim: the goal classes are mpcat40.tsv's
and a release that disagrees is refused; every table word is in the MP3D
vocabulary; the publisher's goal ids outrank the class table, an unlisted
goal-class instance is dropped, a context word never lands on a goal-class
instance, and instance 0 is never reported; the shared bridge names instances
under a requested scheme with the default (HM3D's) path untouched; the loader
reads val_mini's single-file layout and locates the ``.house`` /
``_semantic.ply`` annotations; the environment publishes every frame with both
names per instance; ``mp3d.run`` no longer refuses ``--perception
ground-truth`` and records the MP3D table; and the HM3D oracle's identity and
method record are byte-for-byte what they were.
"""
from __future__ import annotations

import gzip
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sparx_agency.core.common.label_match import normalize_label
from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.core.planning.objnav.labels.datasets.hm3d import hm3d_label_mapper
from sparx_agency.core.planning.objnav.labels.datasets.mp3d import CATEGORIES, mp3d_label_mapper
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction as A
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_options import PERCEPTION_GROUND_TRUTH
from sparx_agency.tasks.planning.objnav_benchmark_runtime.ground_truth import hm3d_categories
from sparx_agency.tasks.planning.objnav_benchmark_runtime.ground_truth.mp3d_categories import (
    GOAL_LABELS, GOAL_MPCAT40_INDEX, MP3D_TABLE, MPCAT40_TO_CONTEXT, RAW_TO_CONTEXT, MP3DAnnotation,
    check_release, instance_labels, table_sha256)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.ground_truth.perception import (
    GroundTruthDetector, GroundTruthSegmenter)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.ground_truth.source import GroundTruthSource
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import HabitatRGBDSimulator
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d import run as mp3d_run
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.assets import fetch_semantic_config
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.dataset import (
    SEMANTIC_SCENE_DATASET_CONFIG, MP3DDataset)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.env import MP3DEnv
from sparx_agency.tasks.planning.objnav_benchmark_runtime.policy_factory import build_policy
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_ground_truth_perception import arguments
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_mp3d import (
    PROTOCOL, SCENE, FakeSimulator, goal, row, scripted_distances, write_release)

MAPPER = mp3d_label_mapper()
VOCABULARY = MAPPER.vocabulary()


def note(raw, mpcat40):
    return MP3DAnnotation(raw, mpcat40)


def labels_of(objects, goals=None):
    return instance_labels(objects, goals or {}, VOCABULARY)


def mp3d_source(objects, goals=None, scene=SCENE):
    source = GroundTruthSource(MAPPER, MP3D_TABLE)
    source.begin_scene(scene, objects, goals or {})
    return source


def with_official_release_table(dataset):
    """The fixture writer's goal-class indices are placeholders; the oracle needs mpcat40.tsv's."""
    dataset.category_to_mp3d_category_id = dict(GOAL_MPCAT40_INDEX)
    return dataset


#: The publisher's config, as served (dl.fbaipublicfiles.com/habitat/mp3d/config_v1/), in the fields that matter.
OFFICIAL_CONFIG = {"stages": {"paths": {".glb": ["*/*.glb"]}, "default_attributes": {
    "semantic_asset": "%%CONFIG_NAME_AS_ASSET_FILENAME%%_semantic.ply",
    "semantic_descriptor_filename": "%%CONFIG_NAME_AS_ASSET_FILENAME%%.house",
    "up": [0, 0, 1], "front": [0, 1, 0], "semantic_up": [0, 1, 0], "semantic_front": [0, 0, -1]}}}


def install_config(dataset):
    path = dataset.scenes_dir / "mp3d" / SEMANTIC_SCENE_DATASET_CONFIG
    path.write_text(json.dumps(OFFICIAL_CONFIG))
    return path


def annotate(dataset, scene=SCENE, *, config=True):
    glb = dataset.scenes_dir / ("mp3d/%s/%s.glb" % (scene, scene))
    for name in (scene + ".house", scene + "_semantic.ply"):
        glb.with_name(name).write_bytes(b"an annotation the loader only proves is present")
    if config:
        install_config(dataset)


# ── the table ────────────────────────────────────────────────────────────────

def test_the_goal_classes_are_mpcat40s_and_a_release_that_disagrees_is_refused():
    assert tuple(GOAL_MPCAT40_INDEX) == CATEGORIES              # the 21 published goals, in release order
    check_release(GOAL_MPCAT40_INDEX)                           # the objectnav_mp3d_v1 table is exactly this
    with pytest.raises(ValueError, match="mpcat40.tsv"):
        check_release(dict(GOAL_MPCAT40_INDEX, table=4))        # a shifted index is a different class


def test_every_table_word_is_in_the_mp3d_vocabulary_and_each_goal_labels_as_its_primary_prompt():
    for category, label in GOAL_LABELS.items():
        assert label == MAPPER.target_labels(category).detector_prompts[0] and label in VOCABULARY
    for word in list(RAW_TO_CONTEXT.values()) + list(MPCAT40_TO_CONTEXT.values()):
        assert word in VOCABULARY and word not in GOAL_LABELS.values()
    assert all(key == normalize_label(key) for key in RAW_TO_CONTEXT)       # looked up normalised
    assert not set(MPCAT40_TO_CONTEXT) & set(GOAL_MPCAT40_INDEX)            # no goal class is context


def test_the_publishers_goal_ids_outrank_the_class_table_and_an_unlisted_goal_class_instance_is_dropped():
    objects = {7: note("armchair", "chair"), 8: note("chair", "chair"), 9: note("bed", "bed"), 0: note("window", "window")}
    labels = labels_of(objects, {"chair": frozenset({7})})
    assert labels == {7: "chair", 9: "bed"}             # 8 is a chair the publisher left out; bed has no episodes here


def test_context_words_come_from_the_raw_name_first_then_the_class_and_never_from_a_goal_class():
    objects = {1: note("doorway", "door"), 2: note("door", "door"), 3: note("frige", "appliances"),
               4: note("staircase", "stairs"), 5: note("stair", "stairs"), 6: note("toaster", "appliances"),
               10: note("refrigerator", "cabinet")}        # a goal class wins over a context-looking raw name
    assert labels_of(objects) == {1: "doorway", 2: "door", 3: "refrigerator", 4: "staircase", 5: "stairs",
                                  10: "cabinet"}


def test_instance_zero_is_never_reported_because_it_renders_as_background():
    assert labels_of({0: note("chair", "chair")}) == {}


def test_a_goal_id_the_annotations_lack_or_annotate_as_another_class_is_refused():
    with pytest.raises(ValueError, match="same id space"):
        labels_of({7: note("chair", "chair")}, {"chair": frozenset({99})})
    with pytest.raises(ValueError, match="annotated as mpcat40 'stool'"):
        labels_of({7: note("bar stool", "stool")}, {"chair": frozenset({7})})


def test_the_scene_report_and_the_identities_name_the_mp3d_table():
    objects = {7: note("chair", "chair"), 8: note("chair", "chair"), 9: note("wall", "wall")}
    source = mp3d_source(objects, {"chair": frozenset({7})})
    report = source.scene_report
    assert report["reported_instances"] == 1 and report["category_table_sha256"] == table_sha256()
    assert report["named_for_a_goal_category_but_not_a_goal"] == {"chair": 1}
    assert report["unmapped_raw_names"] == {"wall": 1}
    detector = GroundTruthDetector(source)
    assert "Matterport3D" in detector.health()["model"]
    assert detector.health()["metadata"]["checkpoint_sha256"] == table_sha256()
    assert GroundTruthSegmenter(detector).health()["metadata"]["checkpoint_sha256"] == table_sha256()


def test_the_hm3d_oracle_identity_and_record_are_what_they_were():
    """The seam is the only change on HM3D: same table, same hash, same strings in the frozen record."""
    source = GroundTruthSource(hm3d_label_mapper())
    source.begin_scene("00800-TEEsavR23oF", {2: "chair"}, {})
    assert source.categories is hm3d_categories.HM3D_TABLE
    health = GroundTruthDetector(source).health()
    assert health["model"] == "habitat-sim semantic sensor (HM3D-Semantics annotations)"
    assert health["metadata"]["checkpoint_sha256"] == hm3d_categories.table_sha256()
    _, method = build_policy(arguments(), hm3d_label_mapper(), ground_truth=source)
    assert method["ground_truth"]["context_labels"] == "HM3D-Semantics raw names via ground_truth/hm3d_categories.py"


# ── the shared bridge ────────────────────────────────────────────────────────

class _Category:
    """An MP3D category: habitat-sim's default name is the mpcat40 class."""

    def __init__(self, raw, mpcat40):
        self.raw, self.mpcat40 = raw, mpcat40

    def name(self, mapping=""):
        return self.raw if mapping == "raw" else self.mpcat40


def bridge_with(objects):
    simulator = HabitatRGBDSimulator(PROTOCOL.camera(), PROTOCOL.actions(), 0.18, False, height_m=0.88,
                                     semantic=True, scene_dataset_config="mp3d.scene_dataset_config.json")
    simulator._sim = SimpleNamespace(semantic_scene=SimpleNamespace(objects=[
        SimpleNamespace(semantic_id=i, id="0_0_%d" % i, category=_Category(raw, mp)) for i, (raw, mp) in objects.items()]))
    simulator._semantic_objects = simulator._read_semantic_objects("scene.glb")
    return simulator


def test_the_bridge_names_instances_under_a_requested_scheme_and_its_default_is_unchanged():
    simulator = bridge_with({0: ("window", "window"), 3: ("armchair", "chair"), 4: ("Fridge ", "appliances")})
    assert simulator.semantic_objects() == {3: "chair", 4: "appliances"}            # default name(), id 0 dropped
    assert simulator.semantic_objects("raw") == {3: "armchair", 4: "fridge"}
    assert simulator.semantic_objects("mpcat40") == simulator.semantic_objects()
    assert simulator.semantic_objects() is simulator._semantic_objects               # the HM3D path: the cached table


# ── the dataset ──────────────────────────────────────────────────────────────

def write_single_file_split(root, rows):
    """val_mini's layout: episodes and goals_by_category in <split>.json.gz, and no content/."""
    dataset = write_release(root, PROTOCOL, {SCENE: rows})        # scenes and a first index file
    episodes_dir = dataset.episodes_dir
    for shard in (episodes_dir / "content").glob("*.json.gz"):
        shard.unlink()
    relative = "mp3d/%s/%s.glb" % (SCENE, SCENE)
    payload = {"episodes": [dict(r, scene_id=relative, episode_id=str(9 + i)) for i, r in enumerate(rows)],
               "goals_by_category": {"%s.glb_%s" % (SCENE, r["object_category"]): [goal(r["object_category"])]
                                     for r in rows},
               "category_to_task_category_id": {c: i for i, c in enumerate(CATEGORIES)},
               "category_to_mp3d_category_id": dict(GOAL_MPCAT40_INDEX)}
    with gzip.open(episodes_dir / ("%s.json.gz" % PROTOCOL.split), "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)
    return MP3DDataset(episodes_dir, dataset.scenes_dir, PROTOCOL, full=False)


def test_a_split_shipped_as_one_file_is_read_like_the_sharded_release(tmp_path):
    dataset = write_single_file_split(tmp_path, [row("chair"), row("bed"), row("chair")])
    assert dataset.episode_ids() == ("%s/000000" % SCENE, "%s/000001" % SCENE, "%s/000002" % SCENE)
    assert dataset.episodes["%s/000001" % SCENE].source_episode_id == "10"
    assert dataset.goal_instances(SCENE) == {"bed": frozenset({7}), "chair": frozenset({7})}
    assert dataset.category_to_mp3d_category_id == GOAL_MPCAT40_INDEX


def test_the_dataset_locates_the_annotations_and_the_preflight_names_a_missing_one(tmp_path):
    dataset = with_official_release_table(write_release(tmp_path, PROTOCOL, {SCENE: [row()]}))
    episode = dataset.episodes["%s/000000" % SCENE]
    with pytest.raises(FileNotFoundError, match="mp3d_habitat.zip"):
        dataset.semantic_assets(episode)
    with pytest.raises(FileNotFoundError, match="--fetch-semantic-config"):
        dataset.semantic_scene_dataset_config()
    assert "--fetch-semantic-config" in mp3d_run.ground_truth_issues(dataset)[0]
    install_config(dataset)
    assert "1 of 1 selected scenes lack their Matterport3D annotations" in mp3d_run.ground_truth_issues(dataset)[0]
    annotate(dataset)
    house, ply = dataset.semantic_assets(episode)
    assert house.name == SCENE + ".house" and ply.name == SCENE + "_semantic.ply"
    assert mp3d_run.ground_truth_issues(dataset) == []
    dataset.category_to_mp3d_category_id["chair"] = 4
    assert "mpcat40.tsv" in mp3d_run.ground_truth_issues(dataset)[0]


# ── the environment ──────────────────────────────────────────────────────────

class SemanticSimulator(FakeSimulator):
    """The fake bridge with an MP3D semantic sensor: both names per instance and an id frame that moves."""

    TABLE = {7: ("chair", "chair"), 11: ("chair", "chair"), 12: ("doorway", "door"), 13: ("wall", "wall")}

    def __init__(self, protocol):
        super().__init__(protocol)
        self.last_semantic = None
        self.namings = []

    def semantic_objects(self, naming=None):
        self.namings.append(naming)
        return {i: (raw if naming == "raw" else mp) for i, (raw, mp) in self.TABLE.items()}

    def _frame(self):
        rgb, depth, pose = super()._frame()
        ids = np.zeros(rgb.shape[:2], np.int32)
        ids[1:3, 1:4] = 7
        ids[3:5, 5:8] = 11 if len(self.actions_taken) % 2 else 12
        self.last_semantic = ids
        return rgb, depth, pose


def test_the_environment_publishes_every_frame_and_labels_the_scene_once_from_both_names(tmp_path):
    dataset = write_release(tmp_path, PROTOCOL, {SCENE: [row(), row()]})
    annotate(dataset)
    source = GroundTruthSource(MAPPER, MP3D_TABLE)
    simulator = SemanticSimulator(PROTOCOL)
    env = MP3DEnv(dataset, PROTOCOL, simulator=simulator, distance_factory=scripted_distances((4.0, 4.0)),
                  ground_truth=source)
    _, observation = env.reset("%s/000000" % SCENE)
    assert source.labels == {7: "chair", 12: "doorway"}
    assert source.lookup(observation.rgb) is simulator.last_semantic
    observation = env.step(A.TURN_LEFT)
    assert source.lookup(observation.rgb) is simulator.last_semantic and source.frames == 2
    env.reset("%s/000001" % SCENE)
    assert sorted(simulator.namings) == ["mpcat40", "raw"]           # the same scene: not relabelled
    diagnostics = env.evaluation_diagnostics()["ground_truth"]
    assert diagnostics["frames"] == 3 and diagnostics["scene_report"]["named_for_a_goal_category_but_not_a_goal"] == {"chair": 1}


def test_the_environment_needs_a_semantic_bridge_and_is_unchanged_without_a_source(tmp_path):
    dataset = write_release(tmp_path, PROTOCOL, {SCENE: [row()]})
    with pytest.raises(FileNotFoundError, match="mp3d.scene_dataset_config.json"):
        MP3DEnv(dataset, PROTOCOL, ground_truth=GroundTruthSource(MAPPER, MP3D_TABLE))   # never habitat-sim's "default"
    config = install_config(dataset)
    with pytest.raises(EnvContractError, match="semantic sensor"):
        MP3DEnv(dataset, PROTOCOL, simulator=FakeSimulator(PROTOCOL), ground_truth=GroundTruthSource(MAPPER, MP3D_TABLE))
    real = MP3DEnv(dataset, PROTOCOL, ground_truth=GroundTruthSource(MAPPER, MP3D_TABLE))
    assert real._simulator.semantic and real._simulator.scene_dataset_config == str(config)
    plain = MP3DEnv(dataset, PROTOCOL)
    assert not plain._simulator.semantic and plain._simulator.scene_dataset_config is None
    env = MP3DEnv(dataset, PROTOCOL, simulator=FakeSimulator(PROTOCOL), distance_factory=scripted_distances((4.0, 4.0)))
    env.reset("%s/000000" % SCENE)
    assert "ground_truth" not in env.evaluation_diagnostics()


# ── the entrypoint ───────────────────────────────────────────────────────────

def test_the_entrypoint_runs_ground_truth_perception_and_records_the_mp3d_table(tmp_path):
    dataset = write_release(tmp_path, PROTOCOL, {SCENE: [row()]})
    annotate(dataset)
    index = dataset.episodes_dir / "val.json.gz"
    payload = json.loads(gzip.open(index, "rt").read())
    payload["category_to_mp3d_category_id"] = dict(GOAL_MPCAT40_INDEX)
    with gzip.open(index, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream)
    args = mp3d_run.parser().parse_args([
        "--version", "v1", "--split", "val", "--episodes-dir", str(dataset.episodes_dir),
        "--scenes-dir", str(dataset.scenes_dir), "--scenes", SCENE, "--skip-scene-hashes",
        "--agent", "falcon-yolo", "--perception", "ground-truth", "--allow-shared-gpu"])
    env, policy, _, config, issues = mp3d_run.prepare(args)
    try:
        assert not [i for i in issues if "ground-truth" in i.lower() or "Method" in i], issues
        assert config["perception"] == PERCEPTION_GROUND_TRUTH and config["publishable"] is False
        assert isinstance(policy.detector, GroundTruthDetector)
        assert config["method"]["ground_truth"]["category_table_sha256"] == table_sha256()
        assert "mp3d_categories" in config["method"]["ground_truth"]["context_labels"]
        assert env._ground_truth is policy.detector.source
    finally:
        env.close()


def test_the_semantic_config_is_fetched_once_and_must_carry_the_semantic_frame(tmp_path):
    dataset = write_release(tmp_path, PROTOCOL, {SCENE: [row()]})
    served = tmp_path / "served.json"
    served.write_text(json.dumps(OFFICIAL_CONFIG))
    installed = fetch_semantic_config(dataset.scenes_dir, PROTOCOL, url=served.as_uri(), log=lambda m: None)
    assert installed == dataset.semantic_scene_dataset_config()
    assert json.loads(installed.read_text())["stages"]["default_attributes"]["semantic_up"] == [0, 1, 0]
    served.write_text("{}")                                         # already installed: not fetched again
    assert fetch_semantic_config(dataset.scenes_dir, PROTOCOL, url=served.as_uri(), log=lambda m: None) == installed
    installed.unlink()
    with pytest.raises(RuntimeError, match="no semantic frame"):
        fetch_semantic_config(dataset.scenes_dir, PROTOCOL, url=served.as_uri(), log=lambda m: None)
