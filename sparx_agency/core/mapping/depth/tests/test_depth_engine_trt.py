"""
test_depth_engine_trt.py
=========================
Unit tests for DepthEngineTRT and its helper utilities.

All tests run WITHOUT a GPU or a real .engine file.
TRT / pycuda are mocked so these tests are safe on any machine.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from sparx_agency.core.mapping.depth.depth_engine_trt import (
    DepthEngineTRTConfig,
    DepthEngineTRT,
    _letterbox,
    _resize_depth,
)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

class TestDepthEngineTRTConfig:
    def test_empty_engine_path_raises(self):
        with pytest.raises(ValueError, match="engine_path must be set"):
            DepthEngineTRTConfig(engine_path="")

    def test_negative_min_range_raises(self):
        with pytest.raises(ValueError, match="min_range_m must be"):
            DepthEngineTRTConfig(engine_path="/dummy.engine", min_range_m=-1.0)

    def test_max_less_than_min_raises(self):
        with pytest.raises(ValueError, match="max_range_m must be >"):
            DepthEngineTRTConfig(
                engine_path="/dummy.engine",
                min_range_m=5.0,
                max_range_m=5.0,
            )

    def test_bad_input_shape_raises(self):
        with pytest.raises(ValueError, match="input_shape must be"):
            DepthEngineTRTConfig(
                engine_path="/dummy.engine",
                input_shape=(3, 518, 518),   # missing N dim
            )

    def test_valid_config_succeeds(self):
        cfg = DepthEngineTRTConfig(engine_path="/some/path.engine")
        assert cfg.min_range_m > 0
        assert cfg.max_range_m > cfg.min_range_m


# ---------------------------------------------------------------------------
# Engine file validation (no GPU needed)
# ---------------------------------------------------------------------------

class TestDepthEngineTRTEngineLoad:
    def test_missing_engine_raises_file_not_found(self):
        cfg = DepthEngineTRTConfig(engine_path="/nonexistent/path/model.engine")
        engine = DepthEngineTRT(cfg)
        with pytest.raises(FileNotFoundError, match="TRT engine not found"):
            engine._load_engine()

    def test_engine_not_loaded_until_infer_called(self):
        cfg = DepthEngineTRTConfig(engine_path="/nonexistent/path/model.engine")
        engine = DepthEngineTRT(cfg)
        # Should NOT raise during construction
        assert not engine._loaded


# ---------------------------------------------------------------------------
# Preprocessing (_letterbox + _preprocess)
# ---------------------------------------------------------------------------

class TestPreprocess:
    """Validates the preprocessing pipeline purely in numpy (no GPU)."""

    def _make_rgb(self, h: int = 720, w: int = 1280) -> np.ndarray:
        return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

    def test_preprocess_output_shape(self):
        """_preprocess must return (1, 3, H_in, W_in) for the configured shape."""
        cfg = DepthEngineTRTConfig(
            engine_path="/dummy.engine",
            input_shape=(1, 3, 518, 518),
        )
        engine = DepthEngineTRT(cfg)
        rgb = self._make_rgb(720, 1280)
        out = engine._preprocess(rgb)
        assert out.shape == (1, 3, 518, 518), f"Got {out.shape}"

    def test_preprocess_value_range(self):
        """All pixel values after normalisation must be in [0, 1]."""
        cfg = DepthEngineTRTConfig(engine_path="/dummy.engine")
        engine = DepthEngineTRT(cfg)
        rgb = self._make_rgb(480, 640)
        out = engine._preprocess(rgb)
        assert out.min() >= 0.0, "Values below 0"
        assert out.max() <= 1.0, "Values above 1"

    def test_preprocess_dtype(self):
        cfg = DepthEngineTRTConfig(engine_path="/dummy.engine")
        engine = DepthEngineTRT(cfg)
        out = engine._preprocess(self._make_rgb())
        assert out.dtype == np.float32

    def test_preprocess_contiguous(self):
        """Output must be C-contiguous for CUDA memcpy."""
        cfg = DepthEngineTRTConfig(engine_path="/dummy.engine")
        engine = DepthEngineTRT(cfg)
        out = engine._preprocess(self._make_rgb())
        assert out.flags["C_CONTIGUOUS"]


# ---------------------------------------------------------------------------
# _letterbox
# ---------------------------------------------------------------------------

class TestLetterbox:
    def test_shape_is_exact_target(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = _letterbox(img, 518, 518)
        assert result.shape == (518, 518, 3)

    def test_dtype_uint8(self):
        img = np.ones((100, 200, 3), dtype=np.uint8) * 128
        result = _letterbox(img, 64, 64)
        assert result.dtype == np.uint8

    def test_square_input_no_padding(self):
        img = np.full((100, 100, 3), 100, dtype=np.uint8)
        result = _letterbox(img, 64, 64)
        # With a square source the entire canvas should be filled (no black bars)
        assert result.min() > 0

    def test_wide_input_has_top_bottom_padding(self):
        """16:9 → square target must have black top/bottom padding."""
        img = np.full((100, 400, 3), 200, dtype=np.uint8)
        result = _letterbox(img, 200, 200, fill_value=0)
        # Top row should be padding (0)
        assert result[0, 100, 0] == 0


# ---------------------------------------------------------------------------
# _resize_depth
# ---------------------------------------------------------------------------

class TestResizeDepth:
    def test_output_shape(self):
        d = np.ones((518, 518), dtype=np.float32)
        out = _resize_depth(d, 720, 1280)
        assert out.shape == (720, 1280)

    def test_dtype_preserved(self):
        d = np.random.rand(100, 100).astype(np.float32)
        out = _resize_depth(d, 50, 50)
        assert out.dtype == np.float32

    def test_uniform_depth_preserved(self):
        """Resizing a uniform depth map should return the same constant."""
        d = np.full((100, 100), 5.0, dtype=np.float32)
        out = _resize_depth(d, 200, 200)
        np.testing.assert_allclose(out, 5.0, atol=1e-4)
