"""Unit tests for the ROS-free camera-intrinsic image resample."""
import math

import numpy as np
import pytest

from sparx_agency.core.common.intrinsic_remap import build_remap, remap_raw_image

# Real XTEND depth calibration (the sim_adapter target).
XT = dict(fx=253.066668300591147, fy=287.535109816403349,
          cx=236.140442706411449, cy=81.734160313465040, w=504, h=294)


def _square_render(w, h, hfov):
    """Gazebo-style square-pixel render intrinsics (centred principal point)."""
    f = w / (2.0 * math.tan(hfov / 2.0))
    return f, f, w / 2.0, h / 2.0


def test_xtend_from_640x480_render_is_in_bounds():
    sfx, sfy, scx, scy = _square_render(640, 480, math.radians(100.0))
    row, col = build_remap(sfx, sfy, scx, scy, 640, 480,
                           XT["fx"], XT["fy"], XT["cx"], XT["cy"],
                           XT["w"], XT["h"])
    assert col.shape == (504,) and row.shape == (294,)
    assert 0 <= col.min() and col.max() < 640
    assert 0 <= row.min() and row.max() < 480


def test_principal_point_maps_to_render_centre():
    # An integer target principal point must sample the source principal point.
    row, col = build_remap(50.0, 40.0, 50.0, 50.0, 100, 100,
                           25.0, 40.0, 20.0, 10.0, 40, 20)
    assert col[20] == 50  # u = dst_cx  -> src_cx
    assert row[10] == 50  # v = dst_cy  -> src_cy


def test_equal_focal_reduces_to_a_pure_crop():
    # When src and dst share a focal length, the resample IS a crop whose
    # offset relocates the (centred) source PP to the target PP.
    f = 390.7
    row, col = build_remap(f, f, 300.0, 300.0, 600, 600,
                           f, f, 222.273, 108.548, 504, 392)
    assert col[0] == int(round(300.0 - 222.273))  # 78
    assert row[0] == int(round(300.0 - 108.548))  # 191
    assert np.array_equal(col, np.arange(78, 78 + 504))


def test_too_narrow_render_raises():
    # 600x600 @ 1.3098 rad (75 deg) cannot contain the XTEND's ~93 deg HFOV.
    sfx, sfy, scx, scy = _square_render(600, 600, 1.3098)
    with pytest.raises(ValueError):
        build_remap(sfx, sfy, scx, scy, 600, 600,
                    XT["fx"], XT["fy"], XT["cx"], XT["cy"], XT["w"], XT["h"])


def test_gather_is_pixel_exact_32fc1():
    row, col = build_remap(50.0, 50.0, 50.0, 50.0, 100, 100,
                           50.0, 50.0, 40.0, 30.0, 60, 40)
    img = (np.arange(100)[:, None] * 1000 + np.arange(100)[None, :]).astype(np.float32)
    data, step = remap_raw_image(img.tobytes(), 100, 100, 100 * 4, row, col)
    out = np.frombuffer(data, np.float32).reshape(40, 60)
    expected = (row[:, None] * 1000 + col[None, :]).astype(np.float32)
    assert step == 60 * 4
    assert np.array_equal(out, expected)


def test_gather_is_pixel_exact_rgb8():
    row, col = build_remap(50.0, 50.0, 50.0, 50.0, 100, 100,
                           50.0, 50.0, 40.0, 30.0, 60, 40)
    rgb = np.zeros((100, 100, 3), np.uint8)
    rgb[..., 0] = np.arange(100)[:, None] % 256
    rgb[..., 1] = np.arange(100)[None, :] % 256
    rgb[..., 2] = 7
    data, step = remap_raw_image(rgb.tobytes(), 100, 100, 100 * 3, row, col)
    out = np.frombuffer(data, np.uint8).reshape(40, 60, 3)
    assert step == 60 * 3
    assert np.array_equal(out[..., 0], (row[:, None] % 256).astype(np.uint8) + np.zeros((1, 60), np.uint8))
    assert np.array_equal(out[..., 1], (col[None, :] % 256).astype(np.uint8) + np.zeros((40, 1), np.uint8))
    assert np.all(out[..., 2] == 7)
