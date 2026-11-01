from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.common.types.perception import Detection2D
from sparx_agency.core.mapping.detection.closed_open import ClosedOpenConfig, ClosedOpenDetector
from sparx_agency.core.mapping.detection.registry import default_detection_registry
from sparx_agency.core.mapping.detection.rfdetr import RfDetrConfig

COCO = {"chair", "couch", "bed", "toilet", "tv", "potted plant", "sink"}
FRAME = np.zeros((100, 100, 3), dtype=np.uint8)


def box(label, score, xyxy=(10, 10, 50, 50)):
    return Detection2D(label=label, score=score, bbox_xyxy=xyxy, frame_w=100, frame_h=100)


class FakeDetector:
    def __init__(self, detections, owns=None):
        self.detections, self.owns, self.prompts = detections, owns, []

    def set_prompts(self, prompts):
        self.prompts = list(prompts)

    @property
    def answered_prompts(self):
        return tuple(p for p in self.prompts if p in self.owns)

    def detect(self, rgb):
        return list(self.detections)

    def warmup(self):
        pass


def hybrid(closed=(), opened=(), **config):
    detector = ClosedOpenDetector(ClosedOpenConfig(**config), closed=FakeDetector(closed, COCO), open_=FakeDetector(opened))
    return detector


def test_prompts_are_split_by_owner():
    detector = hybrid(open_exclude=("armchair",))
    detector.set_prompts(["chair", "bed", "door", "stool", "armchair"])
    assert detector.closed.prompts == ["chair", "bed", "door", "stool", "armchair"]
    assert detector.open_prompts == ["door", "stool"]


def test_all_coco_prompts_leave_the_open_detector_idle():
    detector = hybrid(closed=[box("bed", 0.9)], opened=[box("door", 0.9)])
    detector.set_prompts(["bed", "toilet"])
    assert detector.open_prompts == []
    assert [d.label for d in detector.detect(FRAME)] == ["bed"]


def test_detections_are_merged_by_score():
    detector = hybrid(closed=[box("bed", 0.6)], opened=[box("door", 0.9)])
    detector.set_prompts(["bed", "door"])
    assert [d.label for d in detector.detect(FRAME)] == ["door", "bed"]


def test_distractor_vetoes_overlapping_chair_only():
    detector = hybrid(closed=[box("chair", 0.8), box("chair", 0.7, (60, 60, 95, 95)), box("bed", 0.8)],
                      opened=[box("stool", 0.5)], distractors=("stool",))
    detector.set_prompts(["chair", "bed", "stool"])
    labels = [(d.label, d.bbox_xyxy) for d in detector.detect(FRAME)]
    assert ("chair", (10, 10, 50, 50)) not in labels
    assert ("chair", (60, 60, 95, 95)) in labels and any(label == "bed" for label, _ in labels)


def test_weak_distractor_does_not_veto():
    detector = hybrid(closed=[box("chair", 0.8)], opened=[box("stool", 0.1)], distractors=("stool",))
    detector.set_prompts(["chair", "stool"])
    assert any(d.label == "chair" for d in detector.detect(FRAME))


def test_empty_prompts_rejected():
    with pytest.raises(ValueError):
        hybrid().set_prompts([" "])


def test_rfdetr_config_validation():
    with pytest.raises(ValueError):
        RfDetrConfig(weights_path=" ")
    with pytest.raises(ValueError):
        RfDetrConfig(conf_thresh=1.5)


def test_registry_lists_the_new_backends():
    assert {"rfdetr", "closed_open"} <= set(default_detection_registry().names())
