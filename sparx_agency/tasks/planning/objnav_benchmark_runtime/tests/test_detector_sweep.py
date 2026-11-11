"""The sweep's matching: IoU, one box per object, true and false positives, recall, the suggested threshold."""
import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep import evaluate
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.kinds import kind_of_annotation

BOX = [100.0, 100.0, 200.0, 200.0]


def gt(kind, xyxy=BOX, pixels=5000):
    return {"kind": kind, "name": kind, "xyxy": xyxy, "pixels": pixels}


def det(cls, conf, xyxy=BOX):
    return {"cls": cls, "conf": conf, "xyxy": xyxy}


def test_iou_of_identical_and_disjoint_boxes():
    assert evaluate.iou(BOX, BOX) == pytest.approx(1.0)
    assert evaluate.iou(BOX, [300.0, 300.0, 400.0, 400.0]) == 0.0


def test_sofa_and_couch_on_one_object_count_once():
    kept = evaluate.merged_by_kind([det("sofa", 0.9), det("couch", 0.8)])
    assert [d["cls"] for d in kept] == ["sofa"]


def test_a_detection_on_the_object_is_a_true_positive_and_another_kind_a_false_positive():
    result = evaluate.match_frame([det("bed", 0.9), det("sofa", 0.9, [300.0, 300.0, 400.0, 400.0])], [gt("bed")], 0.5)
    assert result["bed"] == [1, 0, 1] and result["sofa"] == [0, 1, 0]


def test_below_the_threshold_nothing_is_reported():
    assert not evaluate.match_frame([det("bed", 0.2)], [gt("bed")], 0.5)


def test_a_tiny_object_cannot_be_matched():
    assert evaluate.match_frame([det("bed", 0.9)], [gt("bed", pixels=100)], 0.5)["bed"] == [0, 1, 0]


def test_suggested_threshold_is_the_lowest_one_that_reaches_the_target():
    row = {t: (p, 0.5, 1, 1) for t, p in zip(evaluate.THRESHOLDS, (0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.0))}
    assert evaluate.suggested_threshold(row, 0.8) == 0.4
    assert evaluate.suggested_threshold(row, 1.1) is None


@pytest.mark.parametrize("name,kind", [("l-shaped sofa", "sofa"), ("bedside cabinet", None), ("armchair", "armchair"), ("dining chair", "chair"), ("toilet paper dispenser", None)])
def test_annotation_names_map_to_kinds(name, kind):
    assert kind_of_annotation(name) == kind
