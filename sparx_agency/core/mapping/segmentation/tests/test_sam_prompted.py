"""Contract tests for the SAM-family box-prompted segmenter: no torch, no checkpoint.

Everything here runs in the plain ``.venv`` (no ``ultralytics``); the model is
replaced by a fake that records what it was asked and answers in the shape
ultralytics returns. The one thing the fake cannot prove -- that MobileSAM
leaves the occluder out of a sofa's mask -- was measured by hand on the
development laptop and recorded in the module docstring.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sparx_agency.core.mapping.interfaces.segmentation_model import SegmentationModel
from sparx_agency.core.mapping.segmentation import SamPromptedConfig, SamPromptedSegmenter


class _Tensor:
    def __init__(self, array):
        self._array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _FakeSam:
    """Answers like ``ultralytics.SAM``: a Results list whose masks.data is (N, H, W) in box order."""

    def __init__(self, shape, answer=None):
        self.shape = shape
        self.calls = []
        self.answer = answer

    def to(self, device):
        self.device = device

    def predict(self, bgr, bboxes, imgsz, device, verbose):
        self.calls.append({"bgr": bgr, "bboxes": bboxes, "imgsz": imgsz, "device": device})
        if self.answer is not None:
            return [self.answer]
        h, w = self.shape
        data = np.zeros((len(bboxes), h, w), dtype=bool)
        for index, (x1, y1, x2, y2) in enumerate(bboxes):
            data[index, int(y1):int(y2), int(x1):int(x2)] = True
        masks = SimpleNamespace(data=_Tensor(data))
        return [SimpleNamespace(masks=masks)]


def segmenter(fake):
    seg = SamPromptedSegmenter(SamPromptedConfig(model_path="mobile_sam.pt", device="cpu"))
    seg._model = fake
    return seg


def frame(h=48, w=64):
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = 200
    return rgb


def test_it_is_a_segmentation_model_and_imports_without_torch():
    seg = SamPromptedSegmenter()
    assert isinstance(seg, SegmentationModel)
    assert seg.cfg.model_path == "mobile_sam.pt" and seg.cfg.device == "cpu"


@pytest.mark.parametrize("values", [{"model_path": " "}, {"device": ""}, {"imgsz": 0}, {"max_boxes": 0}])
def test_invalid_config_fails_early(values):
    with pytest.raises(ValueError):
        SamPromptedConfig(**values)


def test_masks_come_back_one_per_box_in_box_order_as_bool_hw():
    fake = _FakeSam((48, 64))
    masks = segmenter(fake).segment(frame(), [(10, 10, 20, 30), (40, 5, 60, 45)])
    assert len(masks) == 2
    assert all(mask.shape == (48, 64) and mask.dtype == bool for mask in masks)
    assert masks[0].sum() == 10 * 20 and masks[0][10:30, 10:20].all()
    assert masks[1].sum() == 20 * 40 and masks[1][5:45, 40:60].all()
    call = fake.calls[0]
    assert call["imgsz"] == 1024 and call["device"] == "cpu"
    # RGB in, BGR to ultralytics.
    assert call["bgr"][..., 2].max() == 200 and call["bgr"][..., 0].max() == 0


def test_no_boxes_is_no_call_and_no_masks():
    fake = _FakeSam((48, 64))
    assert segmenter(fake).segment(frame(), []) == []
    assert fake.calls == []


def test_a_box_the_model_cannot_resolve_is_an_empty_mask_not_a_missing_entry():
    fake = _FakeSam((48, 64), answer=SimpleNamespace(masks=None))
    masks = segmenter(fake).segment(frame(), [(10, 10, 20, 30)])
    assert len(masks) == 1 and masks[0].shape == (48, 64) and not masks[0].any()


def test_a_reply_that_does_not_line_up_with_the_boxes_raises():
    data = np.zeros((3, 48, 64), dtype=bool)
    masks = SimpleNamespace(data=_Tensor(data))
    fake = _FakeSam((48, 64), answer=SimpleNamespace(masks=masks))
    with pytest.raises(RuntimeError, match="3 boxes|for 1 boxes"):
        segmenter(fake).segment(frame(), [(10, 10, 20, 30)])


@pytest.mark.parametrize("box", [(-1, 0, 10, 10), (0, 0, 70, 10), (10, 10, 10, 20), (0, 0, 10, float("nan"))])
def test_boxes_outside_the_frame_or_empty_are_refused(box):
    with pytest.raises(ValueError):
        segmenter(_FakeSam((48, 64))).segment(frame(), [box])


def test_more_boxes_than_the_cap_raise_instead_of_dropping_some():
    seg = SamPromptedSegmenter(SamPromptedConfig(max_boxes=2))
    seg._model = _FakeSam((48, 64))
    with pytest.raises(ValueError, match="max_boxes"):
        seg.segment(frame(), [(0, 0, 5, 5), (5, 5, 10, 10), (10, 10, 15, 15)])


def test_wrong_image_type_is_refused():
    with pytest.raises(ValueError):
        segmenter(_FakeSam((48, 64))).segment(np.zeros((48, 64), np.uint8), [(0, 0, 5, 5)])
    with pytest.raises(ValueError):
        segmenter(_FakeSam((48, 64))).segment(np.zeros((48, 64, 3), np.float32), [(0, 0, 5, 5)])


def test_a_missing_checkpoint_fails_before_any_torch_import(tmp_path):
    seg = SamPromptedSegmenter(SamPromptedConfig(model_path=str(tmp_path / "absent.pt")))
    with pytest.raises(FileNotFoundError, match="no download"):
        seg.segment(frame(), [(0, 0, 5, 5)])
