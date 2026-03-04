"""
depth_engine_trt.py
====================
TensorRT backend for DepthAnything V3.

Architecture reference: zibochen6/ros2-depth-anything-v3-trt

Design rules (spatial_logic.md):
  - Validates ``trt_engine`` file existence before loading.
  - Output tensor shape is validated before returning depth map.
  - Raises errors on invalid state; no silent fallbacks.

Complexity: O(HxW) per call (dominated by letterbox + GPU memcpy).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from sparx_agency.core.mapping.interfaces.depth_model import DepthModel


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DepthEngineTRTConfig:
    """Configuration for the TensorRT DepthAnything V3 engine.

    Args:
        engine_path: Absolute path to the pre-built ``.engine`` file.
        input_shape: NCHW shape expected by the engine.
            Default matches the official Depth-Anything-V3 518x518 export.
        min_range_m: Minimum valid depth in metres (values clipped below this).
        max_range_m: Maximum valid depth in metres (values clipped above this).
        device_id: CUDA device index.
    """
    engine_path: str = ""                          # required; empty → raises on __post_init__
    input_shape: Tuple[int, int, int, int] = (1, 3, 518, 518)
    min_range_m: float = 0.3
    max_range_m: float = 20.0
    device_id: int = 0

    def __post_init__(self) -> None:
        if not self.engine_path:
            raise ValueError("DepthEngineTRTConfig: engine_path must be set.")
        if self.min_range_m <= 0.0:
            raise ValueError("min_range_m must be > 0.")
        if self.max_range_m <= self.min_range_m:
            raise ValueError("max_range_m must be > min_range_m.")
        if len(self.input_shape) != 4:
            raise ValueError("input_shape must be (N, C, H, W).")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DepthEngineTRT(DepthModel):
    """TensorRT inference wrapper for DepthAnything V3.

    Implements the ``DepthModel`` ABC, making it a drop-in replacement for
    ``DepthAnythingV2DepthModel`` in ``MappingPipeline``.

    The engine is loaded lazily the first time ``infer_depth()`` is called so
    that the object can be constructed in CPU-only environments (tests).

    Args:
        cfg: Engine configuration.

    Raises:
        FileNotFoundError: If ``cfg.engine_path`` does not exist at load time.
        RuntimeError: If TensorRT binding shapes are incompatible or if
            the output tensor shape is unexpected.
    """

    def __init__(self, cfg: DepthEngineTRTConfig) -> None:
        self.cfg = cfg
        self._engine = None      # tensorrt.ICudaEngine
        self._context = None     # tensorrt.IExecutionContext
        self._bindings: list = []
        self._d_input = None     # pycuda device buffer
        self._d_output = None    # pycuda device buffer
        self._h_output: Optional[np.ndarray] = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer_depth(self, rgb: np.ndarray) -> np.ndarray:
        """Run depth estimation on an RGB image.

        Args:
            rgb: HxWx3 uint8 or float32 image (RGB order).

        Returns:
            depth_m: HxW float32 depth map in metres, clipped to
                [min_range_m, max_range_m].

        Raises:
            FileNotFoundError: Engine file missing.
            RuntimeError: Unexpected output shape from the engine.
        """
        self._ensure_loaded()

        inp_batch = self._preprocess(rgb)       # (1, 3, H_in, W_in)
        raw_out = self._run_inference(inp_batch) # (H_in, W_in) float32

        # Letterbox inversion: calculate the active region in the 518x518 output
        # that actually corresponds to the original image (ignoring black bars).
        expected_h, expected_w = self.cfg.input_shape[2], self.cfg.input_shape[3]
        if raw_out.shape != (expected_h, expected_w):
            raise RuntimeError(
                f"Engine output shape {raw_out.shape} != "
                f"expected ({expected_h}, {expected_w})."
            )

        orig_h, orig_w = rgb.shape[:2]
        active_region = self._get_active_subregion(orig_h, orig_w, expected_h, expected_w)
        y0, x0, y1, x1 = active_region
        
        # Crop the active region
        raw_cropped = raw_out[y0:y1, x0:x1]

        # Resize cropped depth back to original resolution
        depth_resized = _resize_depth(raw_cropped, orig_h, orig_w)

        depth_m = np.clip(depth_resized, self.cfg.min_range_m, self.cfg.max_range_m)
        return depth_m.astype(np.float32)

    def _get_active_subregion(
        self, 
        src_h: int, 
        src_w: int, 
        target_h: int, 
        target_w: int
    ) -> Tuple[int, int, int, int]:
        """Calculate the [y0, x0, y1, x1] box of the scaled image inside the letterbox.
        
        Complexity: O(1).
        """
        scale = min(target_w / src_w, target_h / src_h)
        new_w = int(round(src_w * scale))
        new_h = int(round(src_h * scale))

        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        
        return (pad_top, pad_left, pad_top + new_h, pad_left + new_w)

    def unload(self) -> None:
        """Release CUDA resources and engine."""
        import pycuda.driver as cuda  # type: ignore

        if self._d_input is not None:
            self._d_input.free()
        if self._d_output is not None:
            self._d_output.free()
        self._context = None
        self._engine = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the TRT engine if not already loaded."""
        if not self._loaded:
            self._load_engine()
            self._loaded = True

    def _load_engine(self) -> None:
        """Load and initialise the TensorRT engine.

        Complexity: O(1) (I/O bound).

        Raises:
            FileNotFoundError: Engine file not found.
            RuntimeError: TRT context creation failure.
        """
        engine_path = Path(self.cfg.engine_path)
        if not engine_path.exists():
            raise FileNotFoundError(
                f"TRT engine not found: {engine_path}. "
                "Build it with trtexec or use DepthAnythingV2DepthModel "
                "for the HuggingFace backend."
            )

        import tensorrt as trt          # type: ignore
        import pycuda.autoinit          # type: ignore  # noqa: F401
        import pycuda.driver as cuda    # type: ignore

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as f:
            engine_data = f.read()

        self._engine = runtime.deserialize_cuda_engine(engine_data)
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TRT engine: {engine_path}")

        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Failed to create TRT execution context.")

        self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        """Allocate pinned host + device buffers for input and output.

        Complexity: O(1).
        """
        import tensorrt as trt          # type: ignore
        import pycuda.driver as cuda    # type: ignore

        N, C, H, W = self.cfg.input_shape
        input_size = int(N * C * H * W)
        output_size = int(N * H * W)  # depth is single-channel

        self._d_input = cuda.mem_alloc(input_size * np.dtype(np.float32).itemsize)
        self._d_output = cuda.mem_alloc(output_size * np.dtype(np.float32).itemsize)
        self._h_output = np.empty((N, H, W), dtype=np.float32)

        self._bindings = [int(self._d_input), int(self._d_output)]

    def _preprocess(self, rgb: np.ndarray) -> np.ndarray:
        """Letterbox-resize and normalise an RGB image to NCHW float32.

        Args:
            rgb: H x W x 3 uint8 or float32.

        Returns:
            Normalised tensor of shape (1, 3, H_in, W_in) float32 in [0, 1].

        Complexity: O(H_in * W_in).
        """
        _, _, H_in, W_in = self.cfg.input_shape
        resized = _letterbox(rgb, H_in, W_in)   # (H_in, W_in, 3) uint8

        arr = resized.astype(np.float32) / 255.0
        # HWC → CHW → NCHW
        arr = arr.transpose(2, 0, 1)[np.newaxis, :]  # (1, 3, H_in, W_in)
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _run_inference(self, inp: np.ndarray) -> np.ndarray:
        """Copy input to device, run TRT inference, copy output back.

        Args:
            inp: NCHW float32 array ready for the engine.

        Returns:
            depth_raw: (H_in, W_in) float32 depth (raw metric metres).

        Complexity: O(H_in * W_in) data transfer + amortised GPU inference.
        """
        import pycuda.driver as cuda    # type: ignore

        cuda.memcpy_htod(self._d_input, inp)
        self._context.execute_v2(self._bindings)
        cuda.memcpy_dtoh(self._h_output, self._d_output)

        N, H, W = self._h_output.shape
        return self._h_output[0]  # (H, W)


# ---------------------------------------------------------------------------
# Pure-numpy utilities (no GPU dependency — facilitates unit testing)
# ---------------------------------------------------------------------------

def _letterbox(
    img: np.ndarray,
    target_h: int,
    target_w: int,
    fill_value: int = 0,
) -> np.ndarray:
    """Scale ``img`` to fit (target_h, target_w) while preserving aspect ratio.

    Pads with ``fill_value`` to reach the exact target size.

    Args:
        img: HxWx3 uint8 image.
        target_h: Desired output height.
        target_w: Desired output width.
        fill_value: Pixel value used for padding.

    Returns:
        Letterboxed image of shape (target_h, target_w, 3) uint8.

    Complexity: O(target_h * target_w).
    """
    try:
        import cv2  # type: ignore
        _cv2_available = True
    except ImportError:
        _cv2_available = False

    src_h, src_w = img.shape[:2]
    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))

    if _cv2_available:
        if img.dtype != np.uint8:
            img_u8 = (np.clip(img, 0, 255)).astype(np.uint8)
        else:
            img_u8 = img
        resized = cv2.resize(img_u8, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        # Fallback: nearest-neighbour via numpy (test-only path)
        y_idx = (np.arange(new_h) * (src_h / new_h)).astype(np.int32)
        x_idx = (np.arange(new_w) * (src_w / new_w)).astype(np.int32)
        resized = img[y_idx[:, None], x_idx[None, :]]
        resized = resized.astype(np.uint8)

    canvas = np.full((target_h, target_w, 3), fill_value, dtype=np.uint8)
    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas


def _resize_depth(depth: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize a (H, W) depth map to (target_h, target_w) using bilinear interpolation.

    Args:
        depth: Source depth map as 2-D float32.
        target_h: Output height.
        target_w: Output width.

    Returns:
        Resized depth map as float32.

    Complexity: O(target_h * target_w).
    """
    try:
        import cv2  # type: ignore
        return cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        # Fallback: nearest-neighbour
        src_h, src_w = depth.shape
        y_idx = (np.arange(target_h) * (src_h / target_h)).astype(np.int32)
        x_idx = (np.arange(target_w) * (src_w / target_w)).astype(np.int32)
        return depth[y_idx[:, None], x_idx[None, :]].astype(np.float32)
