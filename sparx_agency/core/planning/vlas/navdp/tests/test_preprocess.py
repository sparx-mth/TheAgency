"""The one NavDP preprocessing implementation, checked against what it replaced.

These assertions are the contract two fine-tune pipelines and the verification
tool now share: the same resize/pad, the same depth band, and -- the one that was
actually wrong before this module existed -- the same channel order as the
deployed server.
"""
import numpy as np
import pytest

from sparx_agency.core.planning.vlas.navdp.errors import NavDPError
from sparx_agency.core.planning.vlas.navdp.preprocess import (
    ENCODER_COLOR_ORDER,
    IMAGE_SIZE,
    preprocess_depth,
    preprocess_rgb,
    resize_pad,
)


def _frame(h=480, w=640):
    """A frame whose three channels are constant and distinguishable."""
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[..., 0] = 10      # R
    rgb[..., 1] = 120     # G
    rgb[..., 2] = 250     # B
    return rgb


# ── resize_pad ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("shape", [(480, 640, 3), (480, 640), (640, 480, 3), (224, 224, 3)])
def test_resize_pad_always_returns_an_exact_square(shape):
    out = resize_pad(np.zeros(shape, np.uint8), IMAGE_SIZE)
    assert out.shape[:2] == (IMAGE_SIZE, IMAGE_SIZE)
    assert out.ndim == len(shape)


def test_resize_pad_centre_pads_rather_than_stretching():
    # A landscape frame gets zero bands top and bottom, not a squashed image.
    filled = resize_pad(np.full((480, 640, 3), 255, np.uint8), IMAGE_SIZE)
    assert filled[0, IMAGE_SIZE // 2].max() == 0        # top band is padding
    assert filled[IMAGE_SIZE // 2, IMAGE_SIZE // 2].min() > 0   # centre is image


# ── colour order ─────────────────────────────────────────────────────────
def test_default_encoder_order_is_what_the_deployed_server_sends():
    # navdp_trt_server.py does cvtColor(RGB2BGR) before process_image.
    assert ENCODER_COLOR_ORDER == "bgr"


def test_rgb_input_is_swapped_for_the_bgr_encoder():
    out = preprocess_rgb(_frame(), input_order="rgb", layout="hwc")
    centre = out[IMAGE_SIZE // 2, IMAGE_SIZE // 2]
    assert centre[0] == pytest.approx(250 / 255.0, abs=2e-2)   # B first
    assert centre[2] == pytest.approx(10 / 255.0, abs=2e-2)    # R last


def test_bgr_input_reaches_a_bgr_encoder_untouched():
    bgr = _frame()[:, :, ::-1].copy()
    out = preprocess_rgb(bgr, input_order="bgr", layout="hwc")
    same = preprocess_rgb(_frame(), input_order="rgb", layout="hwc")
    np.testing.assert_allclose(out, same, atol=1e-6)


def test_asking_for_the_other_order_actually_swaps():
    as_bgr = preprocess_rgb(_frame(), input_order="rgb", encoder_order="bgr", layout="hwc")
    as_rgb = preprocess_rgb(_frame(), input_order="rgb", encoder_order="rgb", layout="hwc")
    np.testing.assert_allclose(as_rgb, as_bgr[:, :, ::-1], atol=1e-6)


# ── layout ───────────────────────────────────────────────────────────────
def test_rgb_layouts_are_transposes_of_each_other():
    chw = preprocess_rgb(_frame(), layout="chw")
    hwc = preprocess_rgb(_frame(), layout="hwc")
    assert chw.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert hwc.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    np.testing.assert_allclose(chw.transpose(1, 2, 0), hwc, atol=1e-6)
    assert chw.dtype == np.float32 and chw.flags["C_CONTIGUOUS"]


def test_depth_layouts():
    depth = np.full((480, 640), 2.0, np.float32)
    assert preprocess_depth(depth, layout="chw").shape == (1, IMAGE_SIZE, IMAGE_SIZE)
    assert preprocess_depth(depth, layout="hwc").shape == (IMAGE_SIZE, IMAGE_SIZE, 1)


# ── depth band ───────────────────────────────────────────────────────────
def test_depth_outside_the_band_reads_as_no_measurement():
    depth = np.full((240, 240), 9.0, np.float32)     # beyond the 5 m ceiling
    depth[:120] = 0.05                               # below the 0.1 m floor
    out = preprocess_depth(depth, layout="hwc")
    assert np.count_nonzero(out) == 0


def test_non_finite_depth_is_zeroed_before_the_resize_can_spread_it():
    depth = np.full((240, 240), 2.0, np.float32)
    depth[100, 100] = np.nan
    depth[101, 101] = np.inf
    out = preprocess_depth(depth, layout="hwc")
    assert np.isfinite(out).all()
    # One bad sample must not wipe the frame.
    assert np.count_nonzero(out) > out.size * 0.5


def test_depth_inside_the_band_survives():
    out = preprocess_depth(np.full((240, 240), 2.0, np.float32), layout="hwc")
    assert out[IMAGE_SIZE // 2, IMAGE_SIZE // 2, 0] == pytest.approx(2.0, abs=1e-3)


# ── argument validation ──────────────────────────────────────────────────
@pytest.mark.parametrize("kwargs", [
    {"input_order": "rgba"},
    {"encoder_order": "gbr"},
    {"layout": "whc"},
])
def test_unknown_order_or_layout_raises_navdp_error(kwargs):
    with pytest.raises(NavDPError):
        preprocess_rgb(_frame(), **kwargs)


def test_unknown_depth_layout_raises_navdp_error():
    with pytest.raises(NavDPError):
        preprocess_depth(np.zeros((10, 10), np.float32), layout="whc")
