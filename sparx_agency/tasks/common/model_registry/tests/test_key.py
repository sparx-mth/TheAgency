"""Naming-scheme tests: pure parsing/derivation, no filesystem or hardware."""
import pytest

from sparx_agency.tasks.common.model_registry.key import ModelKey, parse_resolution


def test_parse_resolution_string():
    assert parse_resolution("546x364") == (546, 364)
    assert parse_resolution("546X364") == (546, 364)
    assert parse_resolution(" 546 x 364 ") == (546, 364)


def test_parse_resolution_tuple_and_none():
    assert parse_resolution((546, 364)) == (546, 364)
    assert parse_resolution(None) is None


def test_parse_resolution_rejects_missing_x():
    with pytest.raises(ValueError):
        parse_resolution("546364")


def test_stem_with_role_and_resolution():
    key = ModelKey(model_id="da3_metric_large", role="depth_only",
                   precision="fp16", height=546, width=364)
    assert key.stem() == "da3_metric_large.depth_only.fp16.546x364"
    assert key.filename() == "da3_metric_large.depth_only.fp16.546x364.engine"


def test_stem_without_role_or_resolution():
    key = ModelKey(model_id="yolo_world_s", precision="fp16")
    assert key.stem() == "yolo_world_s.fp16"


def test_relpath_uses_target_tag():
    key = ModelKey(model_id="da3_metric_large", role="depth_only",
                   precision="fp16", height=546, width=364, target_tag="orin_sm87")
    assert str(key.relpath()) == "engines/orin_sm87/da3_metric_large.depth_only.fp16.546x364.engine"
    assert str(key.relpath("rtx4090_sm89")) == \
        "engines/rtx4090_sm89/da3_metric_large.depth_only.fp16.546x364.engine"


def test_relpath_without_target_tag_raises():
    key = ModelKey(model_id="da3_metric_large")
    with pytest.raises(ValueError):
        key.relpath()
