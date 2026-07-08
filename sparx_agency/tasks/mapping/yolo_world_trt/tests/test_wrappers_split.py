"""Tests for the backbone/head cut logic (pure python -- no torch/ultralytics).

The wrappers reuse a loaded model's ``.f`` / ``.i`` / ``.save`` metadata, so the
*routing* decisions (where to cut, which backbone outputs the head needs) can be
checked with a lightweight fake of the ultralytics layer list. The actual tensor
math is guarded at export time by the numerical parity gate in ``export_onnx.py``.
"""
from sparx_agency.tasks.mapping.yolo_world_trt import wrappers


class _Layer:
    """Minimal stand-in for an ultralytics module (name = its 'type')."""

    def __init__(self, i, f, kind):
        self.i = i
        self.f = f
        self._kind = kind
        # Rename the instance's class so type(m).__name__ == kind.
        self.__class__ = type(kind, (_Layer,), {})


def _fake_yolo_world():
    """A tiny YOLOv8-world-like layer list: backbone 0-4, then text-fused head."""
    return [
        _Layer(0, -1, "Conv"),
        _Layer(1, -1, "Conv"),
        _Layer(2, -1, "C2f"),
        _Layer(3, -1, "Conv"),
        _Layer(4, -1, "SPPF"),          # last backbone layer
        _Layer(5, -1, "C2fAttn"),       # first text-aware -> cut here
        _Layer(6, [-1, 2], "Concat"),   # head references backbone layer 2
        _Layer(7, -1, "ImagePoolingAttn"),
        _Layer(8, [7, 5, 2], "WorldDetect"),
    ]


def test_find_cut_is_first_text_layer():
    assert wrappers.find_cut(_fake_yolo_world()) == 5


def test_backbone_outputs_include_cut_minus_one_and_referenced():
    layers = _fake_yolo_world()
    cut = wrappers.find_cut(layers)
    outs = wrappers.backbone_output_indices(layers, cut)
    # cut-1 == 4 (feed into first head layer) and layer 2 (referenced by 6 & 8).
    assert outs == [2, 4]


def test_find_cut_raises_without_text_layer():
    plain = [_Layer(0, -1, "Conv"), _Layer(1, -1, "C2f")]
    try:
        wrappers.find_cut(plain)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "text-aware" in str(e)
