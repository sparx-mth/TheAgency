"""Minimal TensorRT 10.x engine runner for the NavDP point-goal engines.

Mirrors the *IO mechanics* of the repo's live TRT-10 model
(``core.mapping.depth.depth_anything_v3.DA3TensorRTModel``): name-based tensor
IO (``num_io_tensors`` / ``get_tensor_name`` / ``get_tensor_shape`` /
``set_tensor_address``) and ``execute_async_v3``. It deliberately does NOT copy
that class's module-level ``import tensorrt`` / ``import pycuda.autoinit`` /
``import cv2`` or its ``tuple[...]`` annotations -- those would break ``core``'s
two hard rules: stay importable with only numpy present, and stay Python-3.8
compatible (the FALCON Noetic adapter imports ``core`` under 3.8). So TensorRT
and pycuda are imported lazily inside methods, exactly like
``core.planning.navdp.client`` lazy-imports ``requests``/``PIL``.

Differences from ``DA3TensorRTModel`` that matter here:
  * Retained CUDA primary context (``Device.retain_primary_context()``) with a
    mandatory ``push()/pop()`` around every call, instead of ``pycuda.autoinit``.
    This is single-context (so a torch fallback in the same process shares it
    safely) and survives the Flask server's worker thread.
  * Multiple named inputs (the NavDP engines take 2-4 inputs each) and partial
    feeds: conditioning tensors that are constant across the 10-step diffusion
    loop are uploaded once and left resident on the device.
  * A version/compute-capability lock against the sibling ``.json`` manifest, so
    an engine built for a different GPU or TensorRT build fails loud at load.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from sparx_agency.core.planning.navdp.trt.errors import NavDPError


class TRTEngineRunner:
    """Load one serialized TensorRT engine and run fixed-shape inference.

    Args:
        engine_path: path to a serialized ``.engine`` file.
        device_id: CUDA device index.
        verify_manifest: if True and a sibling ``<engine>.json`` exists, check
            the engine's TensorRT version and target compute capability against
            the importing runtime/GPU and raise :class:`NavDPError` on mismatch.

    Raises:
        NavDPError: engine file missing, deserialization failure, or a manifest
            version / compute-capability mismatch.
    """

    def __init__(self, engine_path, device_id=0, verify_manifest=True):
        self.engine_path = Path(engine_path)
        self.device_id = int(device_id)
        if not self.engine_path.exists():
            raise NavDPError("TRT engine not found: %s" % self.engine_path)
        self._engine = None
        self._context = None
        self._ctx = None        # retained CUDA primary context
        self._stream = None
        self._inputs = {}       # name -> {"host", "device", "shape"}
        self._outputs = {}      # name -> {"host", "device", "shape"}
        if verify_manifest:
            self._verify_manifest()
        self._load()

    # ------------------------------------------------------------------
    # Manifest version lock
    # ------------------------------------------------------------------
    def _verify_manifest(self):
        """Fail loud if the engine was built for another TRT build / GPU.

        Serialized engines only deserialize under the exact TensorRT build and
        GPU compute capability they were built with; this turns the otherwise
        cryptic deserialize failure into an actionable message.
        """
        meta_path = self.engine_path.with_suffix(self.engine_path.suffix + ".json")
        if not meta_path.exists():
            return
        meta = json.loads(meta_path.read_text())
        import tensorrt as trt  # noqa: F401  (lazy: keep core numpy-only at import)
        import pycuda.driver as cuda

        cuda.init()
        sm_major, sm_minor = cuda.Device(self.device_id).compute_capability()
        sm = sm_major * 10 + sm_minor
        want_sm = meta.get("sm")
        if want_sm is not None and int(want_sm) != sm:
            raise NavDPError(
                "Engine %s was built for SM %s but this GPU is SM %s; rebuild it "
                "on this device (engines are not portable across GPUs)."
                % (self.engine_path.name, want_sm, sm))
        want_trt = meta.get("trt_version")
        # Engines deserialize only under the EXACT build that wrote them, incl.
        # the patch level (JetPack point releases bump it), so compare the full
        # version -- a patch-only mismatch must raise this actionable error
        # rather than fall through to the generic deserialize-None failure.
        if want_trt and str(want_trt) != str(trt.__version__):
            raise NavDPError(
                "Engine %s was built with TensorRT %s but the runtime imports "
                "%s; build and run with the same TensorRT build."
                % (self.engine_path.name, want_trt, trt.__version__))

    # ------------------------------------------------------------------
    # Load + allocate
    # ------------------------------------------------------------------
    def _load(self):
        """Deserialize the engine and allocate one host+device buffer per tensor."""
        import tensorrt as trt
        import pycuda.driver as cuda

        cuda.init()
        self._ctx = cuda.Device(self.device_id).retain_primary_context()
        self._ctx.push()
        try:
            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            self._engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
            if self._engine is None:
                raise NavDPError("Failed to deserialize TRT engine: %s" % self.engine_path)
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise NavDPError("Failed to create execution context: %s" % self.engine_path)
            self._stream = cuda.Stream()
            self._allocate(trt, cuda)
        finally:
            self._ctx.pop()

    def _allocate(self, trt, cuda):
        """Allocate pagelocked host + device memory for every IO tensor."""
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = tuple(int(d) for d in self._engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                raise NavDPError(
                    "Engine %s tensor %r has dynamic shape %r; the NavDP engines "
                    "must be built fully static." % (self.engine_path.name, name, shape))
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            host = cuda.pagelocked_empty(int(trt.volume(shape)), dtype)
            device = cuda.mem_alloc(host.nbytes)
            self._context.set_tensor_address(name, int(device))
            slot = {"host": host, "device": device, "shape": shape, "dtype": dtype}
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._inputs[name] = slot
            else:
                self._outputs[name] = slot

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @property
    def input_names(self):
        """Names of the engine's input tensors."""
        return list(self._inputs.keys())

    @property
    def output_names(self):
        """Names of the engine's output tensors."""
        return list(self._outputs.keys())

    def upload(self, feeds):
        # type: (Dict[str, np.ndarray]) -> None
        """Copy ``feeds`` to the device and leave them resident (no execute).

        Used for conditioning tensors that are constant across the diffusion
        loop (``rgbd_embed`` / ``goal_embed``): upload once, then call
        :meth:`infer` each step with only the changing inputs.
        """
        import pycuda.driver as cuda

        self._ctx.push()
        try:
            for name in feeds:
                if name not in self._inputs:
                    raise NavDPError("Unknown input tensor %r for %s"
                                     % (name, self.engine_path.name))
                slot = self._inputs[name]
                arr = np.ascontiguousarray(feeds[name], dtype=slot["dtype"]).ravel()
                if arr.size != slot["host"].size:
                    raise NavDPError("Input %r expects %d elements but got %d"
                                     % (name, slot["host"].size, arr.size))
                np.copyto(slot["host"], arr)
                cuda.memcpy_htod_async(slot["device"], slot["host"], self._stream)
            self._stream.synchronize()
        finally:
            self._ctx.pop()

    def infer(self, feeds):
        # type: (Dict[str, np.ndarray]) -> Dict[str, np.ndarray]
        """Run the engine, uploading only the named inputs in ``feeds``.

        Inputs not present in ``feeds`` keep whatever is already resident on the
        device -- this is how the 10-step diffusion loop avoids re-uploading the
        constant ``rgbd_embed`` / ``goal_embed`` conditioning every step.

        Args:
            feeds: mapping of input tensor name -> contiguous float array whose
                element count matches that tensor's volume.

        Returns:
            Mapping of output tensor name -> float32 array reshaped to the
            engine's output shape (a fresh copy, safe to keep across calls).
        """
        import pycuda.driver as cuda

        for name in feeds:
            if name not in self._inputs:
                raise NavDPError("Unknown input tensor %r for %s (have %r)"
                                 % (name, self.engine_path.name, self.input_names))
        self._ctx.push()
        try:
            for name, slot in self._inputs.items():
                if name in feeds:
                    arr = np.ascontiguousarray(feeds[name], dtype=slot["dtype"]).ravel()
                    if arr.size != slot["host"].size:
                        raise NavDPError(
                            "Input %r expects %d elements but got %d"
                            % (name, slot["host"].size, arr.size))
                    np.copyto(slot["host"], arr)
                    cuda.memcpy_htod_async(slot["device"], slot["host"], self._stream)
            if not self._context.execute_async_v3(stream_handle=self._stream.handle):
                # execute_async_v3 returns False on a failed enqueue; raise rather
                # than copy back stale device buffers as if they were valid.
                raise NavDPError("execute_async_v3 failed for %s" % self.engine_path.name)
            results = {}
            for name, slot in self._outputs.items():
                cuda.memcpy_dtoh_async(slot["host"], slot["device"], self._stream)
            self._stream.synchronize()
            for name, slot in self._outputs.items():
                results[name] = np.array(slot["host"], dtype=np.float32).reshape(slot["shape"])
            return results
        finally:
            self._ctx.pop()
