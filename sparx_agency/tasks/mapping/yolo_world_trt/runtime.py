"""Run a built YOLO-World TensorRT engine and expose it as a ``DetectionModel``.

:class:`YoloWorldTRTEngine` is the low-level runner (deserialize, allocate, H2D /
execute / D2H, like the DA3 bench runner) and :class:`YoloTRTDetector` wraps it
behind the core
:class:`~sparx_agency.core.mapping.interfaces.detection_model.DetectionModel` ABC
-- the TensorRT analog of ``DepthEngineTRT`` -- so the existing
``yolo_detector_node`` can consume it unchanged.

Prompts are **baked into the engine** at export time, so :meth:`set_prompts` only
accepts a subset of the baked class list; a prompt outside it raises (re-prompting
to a new class needs a rebuild -- the deliberate trade for a DLA-able CNN). The
engine's manifest (``<engine>.json``) supplies the baked class order, input size,
and thresholds.

``tensorrt`` + ``pycuda`` are imported lazily, so this module imports cleanly on a
box without them (the postprocess / preprocess it relies on are pure numpy).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from sparx_agency.core.common.types.perception import Detection2D
from sparx_agency.core.mapping.interfaces.detection_model import DetectionModel
from sparx_agency.tasks.mapping.yolo_world_trt import postprocess, preprocess


class YoloWorldTRTEngine:
    """Static-shape TensorRT runner for one YOLO-World engine (single input/output)."""

    def __init__(self, engine_path: str):
        import pycuda.autoinit  # noqa: F401  (creates the CUDA context)
        import pycuda.driver as cuda
        import tensorrt as trt

        self._cuda = cuda
        self.engine_path = str(engine_path)
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Failed to deserialize engine: %s" % engine_path)
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        self._in = None
        self._out = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self.context.set_tensor_address(name, int(dev))
            slot = {"name": name, "shape": shape, "dtype": dtype,
                    "host": host, "device": dev}
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._in = slot
            else:
                self._out = slot
        if self._in is None or self._out is None:
            raise RuntimeError("Engine must have exactly one input and one output.")
        _, _, self.input_h, self.input_w = self._in["shape"]

    def infer(self, chw_nchw: np.ndarray) -> np.ndarray:
        """Run one ``[1,3,H,W]`` float32 tensor through the engine -> raw output."""
        cuda = self._cuda
        np.copyto(self._in["host"], np.ascontiguousarray(chw_nchw).ravel())
        cuda.memcpy_htod_async(self._in["device"], self._in["host"], self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self._out["host"], self._out["device"], self.stream)
        self.stream.synchronize()
        return self._out["host"].reshape(self._out["shape"])


def load_manifest(engine_path: str) -> dict:
    """Load ``<engine>.json`` (prompts, imgsz, thresholds); {} if absent."""
    p = Path(str(engine_path) + ".json")
    return json.loads(p.read_text()) if p.exists() else {}


class YoloTRTDetector(DetectionModel):
    """``DetectionModel`` backed by a TensorRT YOLO-World engine (DLA or GPU).

    Example:
        >>> det = YoloTRTDetector("yolo_world_s.fp16.dla0.engine")   # doctest: +SKIP
        >>> det.set_prompts(["refrigerator"])                        # subset of baked
        >>> boxes = det.detect(rgb_hwc_uint8)                        # doctest: +SKIP
    """

    def __init__(self, engine_path: str, conf_thresh: Optional[float] = None,
                 iou_thresh: Optional[float] = None, max_det: Optional[int] = None):
        self.engine = YoloWorldTRTEngine(engine_path)
        man = load_manifest(engine_path)
        self._baked: List[str] = [str(p).strip().lower() for p in man.get("prompts", [])]
        if not self._baked:
            raise ValueError(
                "Engine manifest has no baked prompts (%s.json). Rebuild via "
                "export_onnx --prompts ..." % engine_path)
        self.imgsz = tuple(man.get("imgsz_hw", [self.engine.input_h, self.engine.input_w]))
        self.conf_thresh = float(conf_thresh if conf_thresh is not None
                                 else man.get("conf_thresh", 0.25))
        self.iou_thresh = float(iou_thresh if iou_thresh is not None
                                else man.get("iou_thresh", 0.5))
        self.max_det = int(max_det if max_det is not None else man.get("max_det", 100))
        # By default keep every baked class active (the label filter is downstream).
        self._active = set(self._baked)

    def set_prompts(self, prompts: Sequence[str]) -> None:
        """Restrict detection to a subset of the baked classes (cheap; no reload).

        Raises if any prompt was not baked into the engine -- open-vocab freedom
        was traded away for the DLA-able static CNN at export time.
        """
        cleaned = [str(p).strip().lower() for p in prompts if str(p).strip()]
        if not cleaned:
            raise ValueError("set_prompts: at least one non-empty prompt required.")
        unknown = [p for p in cleaned if p not in set(self._baked)]
        if unknown:
            raise ValueError(
                "prompts %s not baked into this engine (baked: %s). Re-export with "
                "these in --prompts and rebuild." % (unknown, self._baked))
        self._active = set(cleaned)

    @property
    def prompts(self) -> List[str]:
        return [p for p in self._baked if p in self._active]

    def detect(self, rgb: np.ndarray) -> List[Detection2D]:
        """Detect the active prompts in an RGB frame; see the ABC contract."""
        img = np.asarray(rgb)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("detect expects HxWx3 RGB, got shape %s" % (img.shape,))
        padded, transform = preprocess.letterbox(img, self.imgsz)
        raw = self.engine.infer(preprocess.to_engine_tensor(padded))
        dets = postprocess.decode(raw, self._baked, transform,
                                  conf_thresh=self.conf_thresh,
                                  iou_thresh=self.iou_thresh, max_det=self.max_det)
        return [d for d in dets if d.label in self._active]
