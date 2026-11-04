"""Operator defaults never select a different backend/checkpoint silently."""
import pytest

from sparx_agency.core.mapping.detection.yolo_world import YoloWorldConfig
from sparx_agency.tasks.mapping.scene_graph.serve.detection_server import (
    _build_real_detector, parse_args)


def test_http_default_is_the_core_x_v2_default():
    args = parse_args([])
    assert args.backend == "yolo_world"
    assert args.model == YoloWorldConfig().model_path == "yolov8x-worldv2.pt"


@pytest.mark.parametrize("model", ["yolov8s-worldv2.pt", "yolov8l-worldv2.pt"])
def test_explicit_yolo_baselines_remain_selectable(model):
    assert parse_args(["--model", model]).model == model
    assert YoloWorldConfig(model_path=model).model_path == model


def test_llmdet_remains_explicit_and_selectable():
    args = parse_args(["--backend", "llmdet", "--model", "snapshot", "--conf", "0.6"])
    assert (args.backend, args.model, args.conf) == ("llmdet", "snapshot", 0.6)
    with pytest.raises(SystemExit):
        parse_args(["--backend", "llmdet", "--conf", "0.6"])


def test_missing_default_x_does_not_fall_back_to_available_s(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "yolov8s-worldv2.pt").write_bytes(b"not a real checkpoint")
    with pytest.raises(SystemExit, match="yolov8x-worldv2.pt"):
        _build_real_detector(parse_args(["--device", "cpu"]))


@pytest.mark.parametrize("model", ["", "   "])
def test_empty_explicit_model_is_not_replaced_with_default(model):
    with pytest.raises(SystemExit):
        parse_args(["--model", model])
