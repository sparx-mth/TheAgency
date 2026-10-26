"""The ground-truth (oracle) perception: the semantic sensor standing in for the detector and the segmenter.

What is proved here, without habitat-sim: the instance boxes and masks read
off an id frame are exact; the publisher's goal ids outrank the name table
and an unlisted goal-named instance is dropped; the source answers only for
the frame it was given and ends the run otherwise; the detector and the
segmenter satisfy the method's contracts and the perception cycle consumes
them unchanged (alias folding, masks on the target's boxes); the dataset
locates the annotation files and the config refuses a missing one by name;
the environment publishes every frame and builds the scene's labels once; the
policy factory builds the oracle pair, turns segmentation on and marks the
run non-publishable; and ``--jobs max`` is bounded by scenes, cores and VRAM.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from sparx_agency.core.mapping.objects.detection_dedup import deduplicate_detections
from sparx_agency.core.planning.objnav.errors import EnvContractError, ObjNavInternalError
from sparx_agency.core.planning.objnav.labels.datasets.hm3d import hm3d_label_mapper
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction as A
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.core.planning.object_search.interfaces import RECOVERABLE_DETECTOR_CAUSES
from sparx_agency.core.planning.object_search.policy import RPTSearchPolicy, RPTSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_options import (
    PERCEPTION_GROUND_TRUTH, PERCEPTION_SERVICE, detector_flags, perception_flags)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.ground_truth.hm3d_categories import (
    GOAL_LABELS, RAW_TO_VOCABULARY, instance_labels, table_sha256, vocabulary_label)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.ground_truth.perception import (
    DEFAULT_MIN_PIXELS, GROUND_TRUTH_BACKEND, GroundTruthDetector, GroundTruthSegmenter)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.ground_truth.semantic_frame import (
    instance_extents, instance_mask)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.ground_truth.source import GroundTruthSource
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.assets import (
    check_installation, semantic_download_instructions, semantic_download_uids)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.dataset import ANNOTATED_SCENE_DATASET_CONFIG
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.env import HM3DEnv
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import HM3D_V1
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.run import ground_truth_issues
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.scene_pool import max_workers
from sparx_agency.tasks.planning.objnav_benchmark_runtime.policy_factory import build_policy
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_hm3d_env import (
    FakeSimulator, PROTOCOL, SCENE, goal, row, scripted_distances, write_release)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import FakeLLM

MAPPER = hm3d_label_mapper()
VOCABULARY = MAPPER.vocabulary()


def frame(shape=(6, 8), **instances):
    """An id frame with each ``name=(x1, y1, x2, y2)`` instance painted in paint order; names are ``i<id>``."""
    ids = np.zeros(shape, dtype=np.int32)
    for name, (x1, y1, x2, y2) in instances.items():
        ids[y1:y2, x1:x2] = int(name[1:])
    return ids


def rgb_for(ids):
    return np.zeros(ids.shape + (3,), dtype=np.uint8)


def source_with(semantic_objects, goal_instances=None, scene="00800-TEEsavR23oF"):
    source = GroundTruthSource(MAPPER)
    source.begin_scene(scene, semantic_objects, goal_instances or {})
    return source


# ── the frame ────────────────────────────────────────────────────────────────

def test_instance_extents_are_tight_exclusive_boxes_with_areas_and_ignore_the_unannotated_zero():
    ids = frame(i2=(1, 1, 4, 3), i3=(4, 3, 8, 6))
    ids[4, 2] = 5                                      # a one-pixel instance
    extents = instance_extents(ids)
    assert set(extents) == {2, 3, 5}
    assert extents[2].xyxy == (1.0, 1.0, 4.0, 3.0) and extents[2].pixels == 6
    assert extents[3].xyxy == (4.0, 3.0, 8.0, 6.0) and extents[3].pixels == 12
    assert extents[5].xyxy == (2.0, 4.0, 3.0, 5.0) and extents[5].pixels == 1
    assert set(instance_extents(ids, min_pixels=2)) == {2, 3}
    assert instance_extents(np.zeros((0, 0), np.int32)) == {}
    with pytest.raises(ValueError, match="2-D integer"):
        instance_extents(np.zeros((6, 8), np.float32))


def test_an_instance_mask_is_its_own_pixels_only():
    ids = frame(i2=(0, 0, 8, 6), i3=(2, 2, 4, 4))     # 3 painted over 2
    assert int(instance_mask(ids, 3).sum()) == 4
    assert int(instance_mask(ids, 2).sum()) == 48 - 4
    assert not instance_mask(ids, 9).any()


# ── the categories ───────────────────────────────────────────────────────────

def test_every_table_word_is_in_the_hm3d_vocabulary_and_each_goal_category_labels_as_its_primary_prompt():
    assert set(RAW_TO_VOCABULARY.values()) <= set(VOCABULARY)
    assert GOAL_LABELS == {"chair": "chair", "bed": "bed", "plant": "potted plant", "toilet": "toilet",
                           "tv_monitor": "tv", "sofa": "couch"}
    assert vocabulary_label("Office_Chair ") == "chair" and vocabulary_label("tv stand") is None
    assert len(table_sha256()) == 64


def test_the_publishers_goal_ids_outrank_the_name_table_and_an_unlisted_goal_named_instance_is_dropped():
    objects = {1: "wall", 2: "chair", 5: "chair", 7: "armchair", 8: "sofa", 9: "door", 10: "bed"}
    labels = instance_labels(objects, {"chair": frozenset({2}), "bed": frozenset({10})}, VOCABULARY)
    assert labels == {2: "chair", 10: "bed", 8: "couch", 9: "door"}
    # 5 and 7 are named for a listed category but are not its goals: a false STOP waiting to happen if emitted.
    assert 5 not in labels and 7 not in labels
    # Without a listing the name table speaks for the category.
    assert instance_labels(objects, {}, VOCABULARY)[7] == "armchair"


def test_a_goal_id_the_annotations_do_not_contain_is_refused():
    with pytest.raises(ValueError, match="same id space"):
        instance_labels({1: "chair"}, {"chair": frozenset({1, 42})}, VOCABULARY)


def test_a_word_outside_the_vocabulary_is_never_emitted():
    labels = instance_labels({1: "door", 2: "chair"}, {}, ("chair",))
    assert labels == {2: "chair"}


# ── the source ───────────────────────────────────────────────────────────────

def test_the_source_answers_for_the_published_frame_by_identity_or_pixels_and_ends_the_run_otherwise():
    source = source_with({2: "chair"})
    ids = frame(i2=(1, 1, 4, 3))
    rgb = rgb_for(ids)
    source.publish(rgb, ids)
    assert source.lookup(rgb) is ids
    assert source.lookup(rgb.copy()) is ids                       # pixel-identical: the same frame
    with pytest.raises(ObjNavInternalError) as caught:
        source.lookup(rgb + 1)
    assert caught.value.__cause__ is None                        # identity drift, not a frame failure: fatal
    assert not isinstance(caught.value.__cause__, RECOVERABLE_DETECTOR_CAUSES)
    assert source.diagnostics()["lookup_misses"] == 1 and source.diagnostics()["frames"] == 1


def test_the_source_refuses_a_frame_before_a_scene_and_a_frame_of_another_size():
    source = GroundTruthSource(MAPPER)
    with pytest.raises(ObjNavInternalError, match="begin_scene"):
        source.publish(rgb_for(frame()), frame())
    source.begin_scene("s", {2: "chair"}, {})
    with pytest.raises(ObjNavInternalError, match="does not match"):
        source.publish(rgb_for(frame()), frame(shape=(5, 5)))


def test_the_scene_report_accounts_for_every_instance():
    source = source_with({1: "wall", 2: "chair", 5: "chair", 9: "door"}, {"chair": frozenset({2})})
    report = source.scene_report
    assert report["annotated_instances"] == 4 and report["reported_instances"] == 2
    assert report["reported_by_label"] == {"chair": 1, "door": 1}
    assert report["goal_instances"] == {"chair": 1}
    assert report["named_for_a_goal_category_but_not_a_goal"] == {"chair": 1}
    assert report["unmapped_raw_names"] == {"wall": 1}
    assert report["category_table_sha256"] == table_sha256()


# ── the detector and the segmenter ────────────────────────────────────────────

def test_the_detector_reports_every_labelled_instance_once_at_confidence_one_with_its_tight_box():
    source = source_with({2: "chair", 3: "sofa", 4: "door", 6: "wall"})
    ids = frame(i2=(1, 1, 4, 3), i3=(4, 3, 8, 6), i4=(0, 0, 1, 6), i6=(7, 0, 8, 1))
    rgb = rgb_for(ids)
    source.publish(rgb, ids)
    detector = GroundTruthDetector(source, min_pixels=1)
    detections = detector.detect(rgb)
    assert [(d.cls, d.conf, d.xyxy) for d in detections] == [
        ("chair", 1.0, (1.0, 1.0, 4.0, 3.0)), ("couch", 1.0, (4.0, 3.0, 8.0, 6.0)), ("door", 1.0, (0.0, 0.0, 1.0, 6.0))]
    assert all(d.cls in VOCABULARY for d in detections)
    assert detector.last_diagnostics == {"mode": GROUND_TRUTH_BACKEND, "visible_instances": 4, "reported": 3,
                                         "below_min_pixels": 0}
    assert detector.last_detections == tuple(detections) and detector.last_inference_ms is not None
    # The method's own alias folding applies afterwards and leaves the boxes alone.
    folded = deduplicate_detections(detections)
    assert [d.cls for d in folded] == ["chair", "sofa", "door"]
    assert [d.xyxy for d in folded] == [d.xyxy for d in detections]


def test_the_resolution_floor_is_declared_and_applied():
    source = source_with({2: "chair", 5: "toilet"})
    ids = frame(i2=(1, 1, 4, 3))
    ids[4, 2] = 5
    rgb = rgb_for(ids)
    source.publish(rgb, ids)
    detector = GroundTruthDetector(source, min_pixels=2)
    assert [d.cls for d in detector.detect(rgb)] == ["chair"]
    assert detector.last_diagnostics["below_min_pixels"] == 1
    assert detector.health()["metadata"]["detector_config"] == {"conf_thresh": 0.0, "min_pixels": 2, "label_table": "hm3d"}
    assert GroundTruthDetector(source).min_pixels == DEFAULT_MIN_PIXELS
    with pytest.raises(ValueError):
        GroundTruthDetector(source, min_pixels=0)


def test_the_identities_are_shaped_like_the_services_and_pinned_to_the_table():
    source = source_with({2: "chair"})
    detector = GroundTruthDetector(source)
    identity = detector.health()
    assert set(identity) == {"model", "device", "classes", "metadata"}
    assert tuple(identity["classes"]) == VOCABULARY
    assert identity["metadata"]["backend"] == GROUND_TRUTH_BACKEND
    assert identity["metadata"]["checkpoint_sha256"] == table_sha256()
    segmenter = GroundTruthSegmenter(detector).health()
    assert set(segmenter) == {"model", "device", "metadata"}
    assert segmenter["metadata"]["backend"] == GROUND_TRUTH_BACKEND and "segmenter_config" in segmenter["metadata"]


def test_the_segmenter_returns_the_exact_instance_masks_in_box_order_and_nothing_of_an_occluder():
    source = source_with({2: "chair", 3: "sofa"})
    ids = frame(i3=(0, 0, 8, 6), i2=(2, 1, 5, 4))       # the chair painted over the sofa: the sofa's box is the whole frame
    rgb = rgb_for(ids)
    source.publish(rgb, ids)
    detector = GroundTruthDetector(source, min_pixels=1)
    detections = detector.detect(rgb)
    segmenter = GroundTruthSegmenter(detector)
    masks = segmenter.segment(rgb, [detections[1].xyxy, detections[0].xyxy])       # sofa first, then chair
    assert len(masks) == 2 and all(m.shape == ids.shape and m.dtype == bool for m in masks)
    assert int(masks[0].sum()) == 48 - 9 and not masks[0][1:4, 2:5].any()         # the sofa without the chair's pixels
    assert int(masks[1].sum()) == 9 and masks[1][1:4, 2:5].all()
    assert tuple(segmenter.last_masks) == tuple(masks)
    assert segmenter.segment(rgb, []) == []


def test_a_box_the_detector_did_not_emit_resolves_to_the_dominant_labelled_instance_or_an_empty_mask():
    source = source_with({2: "chair", 9: "wall"})
    ids = frame(i9=(0, 0, 8, 6), i2=(2, 1, 5, 4))
    rgb = rgb_for(ids)
    source.publish(rgb, ids)
    detector = GroundTruthDetector(source, min_pixels=1)
    detector.detect(rgb)
    segmenter = GroundTruthSegmenter(detector)
    loose, elsewhere = segmenter.segment(rgb, [(1.0, 0.0, 6.0, 5.0), (6.0, 4.0, 8.0, 6.0)])
    assert int(loose.sum()) == 9                     # the chair, the only labelled instance inside
    assert not elsewhere.any()                       # only wall there: nothing the oracle reports


def test_the_segmenter_refuses_a_frame_the_detector_has_not_read_and_the_detector_refuses_another_frame():
    source = source_with({2: "chair"})
    ids = frame(i2=(1, 1, 4, 3))
    rgb = rgb_for(ids)
    source.publish(rgb, ids)
    detector = GroundTruthDetector(source, min_pixels=1)
    with pytest.raises(ObjNavInternalError, match="has not read"):
        GroundTruthSegmenter(detector).segment(rgb, [(1.0, 1.0, 4.0, 3.0)])
    with pytest.raises(ObjNavInternalError, match="not the simulator's current frame"):
        detector.detect(rgb + 7)


# ── the method consumes the oracle unchanged ─────────────────────────────────

def test_the_perception_cycle_takes_the_oracles_boxes_and_masks_as_it_takes_the_services():
    source = source_with({2: "chair", 4: "door"}, {"chair": frozenset({2})})
    camera = HM3D_V1.camera()
    k = camera.intrinsics
    ids = np.zeros((k.height, k.width), np.int32)
    ids[200:280, 280:360] = 2
    ids[0:400, 0:30] = 4
    rgb = rgb_for(ids)
    source.publish(rgb, ids)
    detector = GroundTruthDetector(source)
    settings = RPTSettings(map_size_m=20.0, segmentation=True, doorway_peek={"enabled": False})
    policy = RPTSearchPolicy(detector, FakeLLM(), settings, segmenter=GroundTruthSegmenter(detector))
    episode = ObjNavEpisode("synthetic/0", "synthetic", "hm3d_v1", "val", "chair", camera, HM3D_V1.actions(), 500)
    policy.reset(episode, MAPPER.target_labels("chair"))
    depth = np.full(ids.shape, 3.0, np.float32)
    observation = ObjNavObservation(rgb, depth, AgentPose(0, 0, 0, 0), camera, "chair", 0)
    policy.perception.observe(observation)
    cycle = policy.perception
    assert [d.cls for d in cycle.detections] == ["chair", "door"]
    assert cycle.detector_evidence["mode"] == GROUND_TRUTH_BACKEND
    assert len(cycle.masks) == 1                                  # the target's box, and only the target's
    (detection, mask), = cycle.masks.items()
    assert detection.cls == "chair" and int(mask.sum()) == 80 * 80
    assert cycle.segmenter_evidence["boxes"] == 1 and cycle.counts["segmented_frames"] == 1
    assert cycle.counts["detector_failures"] == 0 and cycle.counts["segmenter_failures"] == 0


# ── the dataset and the assets ───────────────────────────────────────────────

def annotate(dataset, scene_folder, *, config=True, files=True):
    """Write the annotation files a release would carry beside ``scene_folder``'s mesh."""
    stem = scene_folder.split("-", 1)[1]
    release = dataset.scenes_dir / dataset.protocol.scene_root_name
    written = []
    if config:
        path = release / ANNOTATED_SCENE_DATASET_CONFIG
        path.write_text(json.dumps({"stages": {"paths": {".glb": ["*/*/*.basis.glb"]}}}))
        written.append(path)
    if files:
        folder = release / dataset.protocol.split / scene_folder
        for suffix in (".semantic.glb", ".semantic.txt"):
            path = folder / (stem + suffix)
            path.write_bytes(b"HM3D Semantic Annotations\n1,C9C373,\"door\",0\n")
            written.append(path)
    return written


def test_goal_instances_are_the_publishers_object_ids_per_listed_category(tmp_path):
    dataset = write_release(tmp_path, PROTOCOL, {SCENE: [row(), row(category="bed")]})
    assert dataset.goal_instances(SCENE) == {"chair": frozenset({goal()["object_id"]}),
                                             "bed": frozenset({goal()["object_id"]})}
    with pytest.raises(KeyError):
        dataset.goal_instances("00999-nothing")


def test_the_dataset_locates_the_annotations_and_names_the_archive_of_a_missing_one(tmp_path):
    dataset = write_release(tmp_path, PROTOCOL, {SCENE: [row()]})
    episode = next(iter(dataset.episodes.values()))
    with pytest.raises(FileNotFoundError, match="hm3d_val_semantic_configs_v0.1"):
        dataset.annotated_scene_dataset_config()
    with pytest.raises(FileNotFoundError, match="hm3d_val_semantic_annots_v0.1"):
        dataset.semantic_assets(episode)
    config, mesh, descriptor = annotate(dataset, SCENE)
    assert dataset.annotated_scene_dataset_config() == config
    assert dataset.semantic_assets(episode) == (mesh, descriptor)
    assert mesh.name.endswith(".semantic.glb") and descriptor.name.endswith(".semantic.txt")


def test_the_preflight_names_what_is_missing_and_the_one_command_that_provisions_it(tmp_path):
    dataset = write_release(tmp_path, PROTOCOL, {SCENE: [row()]})
    issues = ground_truth_issues(dataset)
    assert len(issues) == 3
    assert "annotated scene dataset config" in issues[0]
    assert "1 of 1 selected scenes lack" in issues[1]
    assert "hm3d.assets --version v1 --instructions" in issues[2]
    annotate(dataset, SCENE)
    assert ground_truth_issues(dataset) == []


def test_the_installation_check_reports_the_semantic_half_separately(tmp_path):
    dataset = write_release(tmp_path, PROTOCOL, {SCENE: [row()]})
    report = check_installation(dataset.episodes_dir, dataset.scenes_dir, PROTOCOL, semantic=True)
    assert report["semantic_ready"] is False and report["semantic_missing_count"] == 3
    assert any("Semantic annotations" in problem for problem in report["problems"])
    assert "semantic_ready" not in check_installation(dataset.episodes_dir, dataset.scenes_dir, PROTOCOL)
    annotate(dataset, SCENE)
    report = check_installation(dataset.episodes_dir, dataset.scenes_dir, PROTOCOL, semantic=True)
    assert report["semantic_missing_count"] == 0            # one scene of twenty: ready says no, missing says none
    assert semantic_download_uids(HM3D_V1) == ("hm3d_val_semantic_annots_v0.1", "hm3d_val_semantic_configs_v0.1")
    text = semantic_download_instructions(HM3D_V1, tmp_path)
    assert "hm3d_val_semantic_annots_v0.1 hm3d_val_semantic_configs_v0.1" in text
    assert ANNOTATED_SCENE_DATASET_CONFIG in text and "--check --semantic" in text and "<token-secret>" in text


# ── the environment ──────────────────────────────────────────────────────────

class SemanticSimulator(FakeSimulator):
    """The fake bridge with a semantic sensor: a fixed annotation table and an id frame that moves with the steps."""

    def __init__(self, protocol):
        super().__init__(protocol)
        self.last_semantic = None
        self.semantic_queries = 0

    def semantic_objects(self):
        self.semantic_queries += 1
        return {goal()["object_id"]: "chair", 11: "chair", 12: "door", 13: "wall"}

    def _frame(self):
        rgb, depth, pose = super()._frame()
        ids = np.zeros(rgb.shape[:2], np.int32)
        ids[1:3, 1:4] = goal()["object_id"]
        ids[3:5, 5:8] = 11 if len(self.actions_taken) % 2 else 12
        self.last_semantic = ids
        return rgb, depth, pose


def test_the_environment_publishes_every_frame_and_builds_the_scene_labels_once(tmp_path):
    dataset = write_release(tmp_path, PROTOCOL, {SCENE: [row(), row()]})
    annotate(dataset, SCENE)
    source = GroundTruthSource(MAPPER)
    simulator = SemanticSimulator(PROTOCOL)
    env = HM3DEnv(dataset, PROTOCOL, simulator=simulator, distance_factory=scripted_distances((4.0, 4.0)),
                  ground_truth=source)
    _, observation = env.reset("%s/000000" % SCENE)
    assert source.scene_key == SCENE and source.labels == {goal()["object_id"]: "chair", 12: "door"}
    assert source.lookup(observation.rgb) is simulator.last_semantic
    observation = env.step(A.TURN_LEFT)
    assert source.lookup(observation.rgb) is simulator.last_semantic and source.frames == 2
    env.reset("%s/000001" % SCENE)
    assert simulator.semantic_queries == 1                  # the same scene: the labels are not rebuilt
    diagnostics = env.evaluation_diagnostics()["ground_truth"]
    assert diagnostics["frames"] == 3 and diagnostics["scene_report"]["reported_instances"] == 2
    assert diagnostics["scene_report"]["named_for_a_goal_category_but_not_a_goal"] == {"chair": 1}


def test_the_environment_refuses_ground_truth_without_a_semantic_bridge_or_without_the_annotations(tmp_path):
    dataset = write_release(tmp_path, PROTOCOL, {SCENE: [row()]})
    with pytest.raises(EnvContractError, match="semantic sensor"):
        HM3DEnv(dataset, PROTOCOL, simulator=FakeSimulator(PROTOCOL), ground_truth=GroundTruthSource(MAPPER))
    with pytest.raises(FileNotFoundError, match="annotated scene dataset config"):
        HM3DEnv(dataset, PROTOCOL, ground_truth=GroundTruthSource(MAPPER))     # the real bridge needs the config


def test_without_a_source_the_environment_is_exactly_what_it_was(tmp_path):
    dataset = write_release(tmp_path, PROTOCOL, {SCENE: [row()]})
    env = HM3DEnv(dataset, PROTOCOL, simulator=FakeSimulator(PROTOCOL), distance_factory=scripted_distances((4.0, 4.0)))
    env.reset("%s/000000" % SCENE)
    assert "ground_truth" not in env.evaluation_diagnostics()


# ── the policy factory and the flags ─────────────────────────────────────────

def arguments(**overrides):
    values = dict(agent="falcon-yolo", policy_config=None, explorer=None, seed=3, segmentation=None, segmenter_url=None,
                  detector_url="http://127.0.0.1:1", detector_backend="yolo_world", detector_timeout_s=30.0,
                  perception=PERCEPTION_GROUND_TRUTH, ground_truth_min_pixels=None)
    values.update(overrides)
    return SimpleNamespace(**values)


def test_the_factory_builds_the_oracle_pair_with_masks_on_and_marks_the_run_non_publishable():
    source = source_with({2: "chair"})
    policy, method = build_policy(arguments(), MAPPER, defaults={"body_height_m": 0.88, "body_radius_m": 0.18},
                                  ground_truth=source)
    assert isinstance(policy.detector, GroundTruthDetector) and isinstance(policy.segmenter, GroundTruthSegmenter)
    assert policy.detector.source is source and policy.settings.segmentation is True
    assert method["publishable"] is False and method["perception"] == PERCEPTION_GROUND_TRUTH
    assert method["detector"]["metadata"]["backend"] == GROUND_TRUTH_BACKEND and method["detector_url"] is None
    assert method["ground_truth"]["bypassed"] == ["detector", "segmenter"]
    assert method["ground_truth"]["min_pixels"] == DEFAULT_MIN_PIXELS
    assert method["segmenter"]["metadata"]["backend"] == GROUND_TRUTH_BACKEND


def test_the_factory_honours_a_policy_config_that_keeps_the_box_pipeline_and_the_declared_floor(tmp_path):
    config = tmp_path / "policy.json"
    config.write_text(json.dumps({"segmentation": False}))
    policy, method = build_policy(arguments(policy_config=config, ground_truth_min_pixels=16), MAPPER,
                                  ground_truth=source_with({2: "chair"}))
    assert policy.segmenter is None and policy.settings.segmentation is False
    assert method["ground_truth"]["bypassed"] == ["detector"] and method["ground_truth"]["min_pixels"] == 16


def test_the_factory_refuses_contradictions():
    source = source_with({2: "chair"})
    with pytest.raises(RuntimeError, match="publishes the simulator's semantic frames"):
        build_policy(arguments(), MAPPER)                                          # no source to read
    with pytest.raises(ValueError, match="--perception is 'service'"):
        build_policy(arguments(perception=PERCEPTION_SERVICE), MAPPER, ground_truth=source)
    with pytest.raises(ValueError, match="--segmenter-url contradicts"):
        build_policy(arguments(segmenter_url="http://127.0.0.1:2"), MAPPER, ground_truth=source)
    with pytest.raises(ValueError, match="--ground-truth-min-pixels is the oracle's"):
        build_policy(arguments(perception=PERCEPTION_SERVICE, ground_truth_min_pixels=8), MAPPER)


def test_a_service_runs_record_gains_no_key_so_runs_recorded_before_the_oracle_still_resume(monkeypatch):
    """The resume lock compares the recorded config verbatim: the oracle's keys exist only in the oracle's record."""
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.services import http_detector

    class ServiceStandIn:
        def __init__(self, url, vocabulary, timeout_s=30.0, expected_backend=None):
            self.timeout_s = timeout_s

        def health(self):
            return {"model": "yolo", "device": "cpu", "classes": list(VOCABULARY),
                    "metadata": {"backend": "yolo_world", "checkpoint_sha256": "x", "detector_config": {"conf_thresh": 0.05}}}

        def detect(self, rgb):
            return []

    monkeypatch.setattr(http_detector, "HttpDetector", ServiceStandIn)
    _, method = build_policy(arguments(perception=PERCEPTION_SERVICE), MAPPER)
    assert "perception" not in method and "ground_truth" not in method
    assert method["detector_url"] == "http://127.0.0.1:1" and method["publishable"] is True


def test_the_campaign_forwards_the_perception_flags_and_a_service_run_forwards_none():
    assert perception_flags(arguments()) == ["--perception", "ground-truth"]
    assert perception_flags(arguments(ground_truth_min_pixels=32)) == ["--perception", "ground-truth",
                                                                       "--ground-truth-min-pixels", "32"]
    assert perception_flags(arguments(perception=PERCEPTION_SERVICE)) == []
    assert detector_flags(arguments())[-2:] == ["--perception", "ground-truth"]


# ── --jobs max ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("scenes,cpus,free,expected", [
    (20, 24, 3872, 8),      # the development box under another campaign: VRAM-bound
    (20, 24, 23000, 20),    # the card free: one child per scene, never more
    (20, 4, 23000, 4),      # a small box: core-bound
    (20, 24, 600, 1),       # no VRAM to speak of: still one
    (3, 24, 23000, 3),
])
def test_jobs_max_is_bounded_by_scenes_cores_and_vram(scenes, cpus, free, expected):
    assert max_workers(scenes, cpu_count=cpus, free_vram_mib=free, renderer_mib=400) == expected
    with pytest.raises(ValueError):
        max_workers(0, cpu_count=cpus, free_vram_mib=free, renderer_mib=400)
