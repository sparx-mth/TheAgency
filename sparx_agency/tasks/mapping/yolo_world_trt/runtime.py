"""Run the open-set YOLO-World split (backbone DLA engine + head GPU engine).

:class:`TwoStageYoloTRT` chains the two engines on one CUDA stream: the backbone's
output feature buffers are *shared* (same device address) as the head's feature
inputs, so no copy happens between DLA and GPU stages. The head's ``txt_feats``
input is set once per re-prompt (dynamic ``N``); the per-frame call only moves the
image in and the detections out.

:class:`YoloTRTDetector` wraps it behind the core ``DetectionModel`` ABC. The
text branch (torch/CLIP via :class:`TextEmbedder`) runs **only** inside
:meth:`set_prompts`; :meth:`detect` is torch-free (TensorRT + numpy), so the frame
loop stays lean. For a torch-free runtime you can instead precompute embeddings
offline and call :meth:`set_text_features`.

``tensorrt`` + ``pycuda`` are imported lazily so this module imports on a box
without them (its preprocess / postprocess are pure numpy).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from sparx_agency.core.common.types.perception import Detection2D
from sparx_agency.core.mapping.interfaces.detection_model import DetectionModel
from sparx_agency.tasks.mapping.yolo_world_trt import postprocess, preprocess
from sparx_agency.tasks.mapping.yolo_world_trt.text_embed import TextEmbedder


def load_manifest(engine_path: str) -> dict:
    """Load ``<engine>.json`` (IO spec, dynamic bounds); {} if absent."""
    p = Path(str(engine_path) + ".json")
    return json.loads(p.read_text()) if p.exists() else {}


def _prod(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


class TwoStageYoloTRT:
    """Backbone(DLA) -> Head(GPU) TensorRT chain with shared feature buffers."""

    def __init__(self, backbone_engine: str, head_engine: str):
        import pycuda.driver as cuda
        import tensorrt as trt

        # Attach to the CUDA *primary* context (the one the runtime API / PyTorch
        # use) rather than the separate context ``pycuda.autoinit`` would create.
        # Sharing the primary context is what lets these engines run in a process
        # that also drives torch on the GPU -- the CLIP text branch when
        # ``text_device`` is a GPU, or a side-by-side PyTorch baseline. With
        # autoinit's own context, a torch GPU op makes torch's context current and
        # the next ``enqueueV3`` fails with 'invalid resource handle' /
        # 'cuTensor permutate execute failed'. The context is released in __del__.
        cuda.init()
        self._cuda_ctx = cuda.Device(0).retain_primary_context()
        self._cuda_ctx.push()

        self._cuda = cuda
        self.stream = cuda.Stream()
        self.bman = load_manifest(backbone_engine)
        self.hman = load_manifest(head_engine)

        logger = trt.Logger(trt.Logger.WARNING)
        rt = trt.Runtime(logger)
        self.b_engine = rt.deserialize_cuda_engine(Path(backbone_engine).read_bytes())
        self.h_engine = rt.deserialize_cuda_engine(Path(head_engine).read_bytes())
        if self.b_engine is None or self.h_engine is None:
            raise RuntimeError("Failed to deserialize one of the engines.")
        self.b_ctx = self.b_engine.create_execution_context()
        self.h_ctx = self.h_engine.create_execution_context()

        self._trt = trt
        self._feat_dev = {}          # name -> device buffer (backbone out = head in)
        self._alloc_backbone(cuda, trt)
        self._alloc_head(cuda, trt)

        self.imgsz = tuple(self.bman.get("imgsz_hw", (self.input_h, self.input_w)))
        self._txt_shape = None
        self._labels: List[str] = []

    # ── allocation ───────────────────────────────────────────────────
    def _alloc_backbone(self, cuda, trt):
        eng, ctx = self.b_engine, self.b_ctx
        self._b_in = None
        for i in range(eng.num_io_tensors):
            name = eng.get_tensor_name(i)
            shape = tuple(eng.get_tensor_shape(name))
            dtype = trt.nptype(eng.get_tensor_dtype(name))
            dev = cuda.mem_alloc(int(np.prod(shape)) * np.dtype(dtype).itemsize)
            ctx.set_tensor_address(name, int(dev))
            if eng.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
                self._b_in = {"name": name, "shape": shape, "host": host, "device": dev}
                _, _, self.input_h, self.input_w = shape
            else:
                self._feat_dev[name] = {"shape": shape, "device": dev}

    def _alloc_head(self, cuda, trt):
        eng, ctx = self.h_engine, self.h_ctx
        n_max = int(self.hman["head"]["n_max"])
        axis = int(self.hman["txt_n_axis"])
        txt_ex = list(self.hman["txt_example_shape"])
        out_ex = list(self.hman["head"]["output_example_shape"])
        self._txt_axis = axis
        self._txt_base = txt_ex
        self._out_ex = out_ex
        # Max buffers: txt with N=n_max, output with class dim = 4 + n_max.
        txt_max = list(txt_ex); txt_max[axis] = n_max
        out_max = list(out_ex); out_max[1] = out_ex[1] - _txt_n(txt_ex, axis) + n_max

        for i in range(eng.num_io_tensors):
            name = eng.get_tensor_name(i)
            dtype = trt.nptype(eng.get_tensor_dtype(name))
            is_in = eng.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            if is_in and name in self._feat_dev:
                ctx.set_tensor_address(name, int(self._feat_dev[name]["device"]))
            elif is_in:                                  # txt_feats
                host = cuda.pagelocked_empty(_prod(txt_max), dtype)
                dev = cuda.mem_alloc(host.nbytes)
                ctx.set_tensor_address(name, int(dev))
                self._txt = {"name": name, "host": host, "device": dev, "dtype": dtype}
            else:                                        # output
                host = cuda.pagelocked_empty(_prod(out_max), dtype)
                dev = cuda.mem_alloc(host.nbytes)
                ctx.set_tensor_address(name, int(dev))
                self._out = {"name": name, "host": host, "device": dev}

    # ── text (re-prompt only) ────────────────────────────────────────
    def set_text_features(self, embeddings: np.ndarray, labels: Sequence[str]) -> None:
        """Upload the text embeddings for the current prompt list (H2D once)."""
        arr = np.ascontiguousarray(np.asarray(embeddings, dtype=self._txt["dtype"]))
        labels = list(labels)
        shape = list(self._txt_base)
        shape[self._txt_axis] = len(labels)
        if _prod(shape) != arr.size:
            raise ValueError("txt embeddings size %d != expected %s for N=%d"
                             % (arr.size, shape, len(labels)))
        self._txt["host"][:arr.size] = arr.ravel()
        self._cuda.memcpy_htod_async(self._txt["device"], self._txt["host"][:arr.size],
                                     self.stream)
        self.stream.synchronize()
        self._txt_shape = tuple(shape)
        self._labels = [str(l).strip().lower() for l in labels]

    # ── per-frame inference ──────────────────────────────────────────
    def infer(self, chw_nchw: np.ndarray) -> np.ndarray:
        """Run one ``[1,3,H,W]`` image tensor through backbone+head -> raw output."""
        if self._txt_shape is None:
            raise RuntimeError("set_text_features / set_prompts before infer().")
        cuda, trt = self._cuda, self._trt
        np.copyto(self._b_in["host"], np.ascontiguousarray(chw_nchw).ravel())
        self.b_ctx.set_input_shape(self._b_in["name"], self._b_in["shape"])
        cuda.memcpy_htod_async(self._b_in["device"], self._b_in["host"], self.stream)
        self.b_ctx.execute_async_v3(stream_handle=self.stream.handle)

        for name, slot in self._feat_dev.items():
            self.h_ctx.set_input_shape(name, slot["shape"])
        self.h_ctx.set_input_shape(self._txt["name"], self._txt_shape)
        out_shape = tuple(self.h_ctx.get_tensor_shape(self._out["name"]))
        self.h_ctx.execute_async_v3(stream_handle=self.stream.handle)

        n = _prod(out_shape)
        cuda.memcpy_dtoh_async(self._out["host"][:n], self._out["device"], self.stream)
        self.stream.synchronize()
        return self._out["host"][:n].reshape(out_shape)

    def __del__(self):
        """Release our push of the CUDA primary context (best-effort)."""
        ctx = getattr(self, "_cuda_ctx", None)
        if ctx is not None:
            try:
                ctx.pop()
            except Exception:       # noqa: BLE001 - interpreter teardown races
                pass


def _txt_n(shape, axis):
    return int(shape[axis])


class YoloTRTDetector(DetectionModel):
    """Open-set ``DetectionModel`` backed by the backbone(DLA)+head(GPU) TRT split.

    Example:
        >>> det = YoloTRTDetector(".../yolo_world_s.backbone.fp16.dla0.engine",
        ...                       ".../yolo_world_s.head.fp16.gpu.engine",
        ...                       text_weights="yolov8s-worldv2.pt")   # doctest: +SKIP
        >>> det.set_prompts(["refrigerator", "chair"])   # any prompts, no rebuild
        >>> boxes = det.detect(rgb_hwc_uint8)
    """

    def __init__(self, backbone_engine: str, head_engine: str,
                 text_weights: Optional[str] = None, text_device: str = "cpu",
                 conf_thresh: Optional[float] = None, iou_thresh: Optional[float] = None,
                 max_det: Optional[int] = None):
        self.stage = TwoStageYoloTRT(backbone_engine, head_engine)
        self.text_weights = text_weights
        self.text_device = text_device
        self._embedder: Optional[TextEmbedder] = None
        man = self.stage.hman
        self.conf_thresh = float(conf_thresh if conf_thresh is not None
                                 else man.get("conf_thresh", 0.25))
        self.iou_thresh = float(iou_thresh if iou_thresh is not None
                                else man.get("iou_thresh", 0.5))
        self.max_det = int(max_det if max_det is not None else man.get("max_det", 100))

    def set_prompts(self, prompts: Sequence[str]) -> None:
        """Encode + upload arbitrary open-vocab prompts (torch text branch, rare)."""
        cleaned = [str(p).strip() for p in prompts if str(p).strip()]
        if not cleaned:
            raise ValueError("set_prompts: at least one non-empty prompt required.")
        if not self.text_weights:
            raise RuntimeError(
                "set_prompts needs text_weights (the .pt for the CLIP text branch). "
                "Or precompute embeddings and call set_text_features().")
        if self._embedder is None:
            self._embedder = TextEmbedder(self.text_weights, self.text_device)
        self.set_text_features(self._embedder.embed(cleaned), cleaned)

    def set_text_features(self, embeddings: np.ndarray, labels: Sequence[str]) -> None:
        """Torch-free path: upload embeddings you computed offline for ``labels``."""
        self.stage.set_text_features(embeddings, labels)

    @property
    def prompts(self) -> List[str]:
        return list(self.stage._labels)

    def detect(self, rgb: np.ndarray) -> List[Detection2D]:
        """Detect the current prompts in an RGB frame; see the ABC contract."""
        img = np.asarray(rgb)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("detect expects HxWx3 RGB, got shape %s" % (img.shape,))
        if not self.stage._labels:
            raise RuntimeError("detect called before set_prompts / set_text_features.")
        padded, transform = preprocess.letterbox(img, self.stage.imgsz)
        raw = self.stage.infer(preprocess.to_engine_tensor(padded))
        return postprocess.decode(raw, self.stage._labels, transform,
                                  conf_thresh=self.conf_thresh,
                                  iou_thresh=self.iou_thresh, max_det=self.max_det)
