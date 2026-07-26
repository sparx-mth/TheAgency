"""Test the label-string parsing for the torch-vs-TRT comparison (no torch/TRT)."""
import pytest

from sparx_agency.tasks.mapping.yolo_world_trt.compare_torch_vs_trt import parse_labels


def test_parse_labels_comma_and_whitespace():
    assert parse_labels("weapon, chair, refrigerator") == \
        ["weapon", "chair", "refrigerator"]


def test_parse_labels_tolerates_semicolons_and_gaps():
    assert parse_labels("  weapon ;chair ,, refrigerator ") == \
        ["weapon", "chair", "refrigerator"]


def test_parse_labels_empty_raises():
    with pytest.raises(ValueError):
        parse_labels("   ,  ; ")
