"""Minimal TensorRT 10.x/11.x engine runner shared by every VLA policy.

Mirrors the *IO mechanics* of the repo's live TRT model
(``core.mapping.depth.depth_anything_v3.DA3TensorRTModel``): name-based tensor
IO (``num_io_tensors`` / ``get_tensor_name`` / ``get_tensor_shape`` /
``set_tensor_address``) and ``execute_async_v3``. It deliberately does NOT copy
that class's module-level ``import tensorrt`` / ``import pycuda.autoinit`` /
``import cv2`` or its ``tuple[...]`` annotations -- those would break ``core``'s
two hard rules: stay importable with only numpy present, and stay Python-3.8
compatible (the FALCON Noetic adapter imports ``core`` under 3.8). So TensorRT
and pycuda are imported lazily inside methods, exactly like the policy clients
lazy-import ``requests``/``PIL``.

Differences from ``DA3TensorRTModel`` that matter here:
  * Retained CUDA primary context (``Device.retain_primary_context()``) with a
    mandatory ``push()/pop()`` around every call, instead of ``pycuda.autoinit``.
    This is single-context (so a torch fallback in the same process shares it
    safely) and survives the Flask server's worker thread.
  * Multiple named inputs and *partial feeds*: a conditioning tensor that is
    constant across an iterative decode loop -- NavDP's ``rgbd_embed`` /
    ``goal_embed`` across 10 diffusion steps, FlowNav's ``global_cond`` across
    K-1 Euler steps -- is uploaded once with :meth:`upload` and left resident on
    the device, then :meth:`infer` re-sends only what changed.
  * A version/compute-capability lock against the sibling ``.json`` manifest, so
    an engine built for a different GPU or TensorRT build fails loud at load.

Per-policy error types
----------------------
Callers catch policy-specific exceptions (``NavDPError``, ``FlowNavError``), so
the runner does not hardcode one: pass ``error_cls`` and every failure in this
module is raised as that type. It defaults to
:class:`~sparx_agency.core.planning.vlas.common.errors.VlaError`, which every
policy error derives from -- so ``except VlaError`` catches all of them and
``except NavDPError`` still catches exactly what it did before.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional  # noqa: F401  (used in type comments)

import numpy as np

from sparx_agency.core.planning.vlas.common.errors import VlaError


class TRTEngineRunner:
    """Load one serialized TensorRT engine and run fixed-shape inference.

    Args:
        engine_path: path to a serialized ``.engine`` file.
        device_id: CUDA device index.
        verify_manifest: if True and a sibling ``<engine>.json`` exists, check
            the engine's TensorRT version and target compute capability against
            the importing runtime/GPU and raise ``error_cls`` on mismatch.
        error_cls: exception type raised for every failure here. Pass the calling
            policy's own error (e.g. ``NavDPError``) so callers can keep catching
            it by name; defaults to :class:`VlaError`, its base.

    Raises:
        error_cls: engine file missing, deserialization failure, or a manifest
            version / compute-capability mismatch.
    """

    def __init__(self, engine_path, device_id=0, verify_manifest=True,
                 error_cls=VlaError):
        self.engine_path = Path(engine_path)
        self.device_id = int(device_id)
        self._err = error_cls
        if not self.engine_path.exists():
            raise self._err("TRT engine not found: %s" % self.engine_path)
        self._engine = None
        self._context = None
        self._ctx = None        # retained CUDA primary context
        self._stream = None
        self._inputs = {}       # name -> {"host", "device", "shape", "dtype"}
        self._outputs = {}      # name -> {"host", "device", "shape", "dtype"}
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
            raise self._err(
                "Engine %s was built for SM %s but this GPU is SM %s; rebuild it "
                "on this device (engines are not portable across GPUs)."
                % (self.engine_path.name, want_sm, sm))
        want_trt = meta.get("trt_version")
        # Engines deserialize only under the EXACT build that wrote them, incl.
        # the patch level (JetPack point releases bump it), so compare the full
        # version -- a patch-only mismatch must raise this actionable error
        # rather than fall through to the generic deserialize-None failure.
        if want_trt and str(want_trt) != str(trt.__version__):
            raise self._err(
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
                raise self._err("Failed to deserialize TRT engine: %s" % self.engine_path)
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise self._err("Failed to create execution context: %s" % self.engine_path)
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
                raise self._err(
                    "Engine %s tensor %r has dynamic shape %r; the VLA engines "
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

        Used for conditioning tensors that are constant across an iterative
        decode loop (NavDP's ``rgbd_embed`` / ``goal_embed``, FlowNav's
        ``global_cond``): upload once, then call :meth:`infer` each step with
        only the changing inputs.
        """
        import pycuda.driver as cuda

        self._ctx.push()
        try:
            for name in feeds:
                if name not in self._inputs:
                    raise self._err("Unknown input tensor %r for %s"
                                    % (name, self.engine_path.name))
                slot = self._inputs[name]
                arr = np.ascontiguousarray(feeds[name], dtype=slot["dtype"]).ravel()
                if arr.size != slot["host"].size:
                    raise self._err("Input %r expects %d elements but got %d"
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
        device -- this is how an iterative decode loop avoids re-uploading the
        constant conditioning every step.

        Args:
            feeds: mapping of input tensor name -> contiguous array whose element
                count matches that tensor's volume.

        Returns:
            Mapping of output tensor name -> float32 array reshaped to the
            engine's output shape (a fresh copy, safe to keep across calls).
        """
        import pycuda.driver as cuda

        for name in feeds:
            if name not in self._inputs:
                raise self._err("Unknown input tensor %r for %s (have %r)"
                                % (name, self.engine_path.name, self.input_names))
        self._ctx.push()
        try:
            for name, slot in self._inputs.items():
                if name in feeds:
                    arr = np.ascontiguousarray(feeds[name], dtype=slot["dtype"]).ravel()
                    if arr.size != slot["host"].size:
                        raise self._err(
                            "Input %r expects %d elements but got %d"
                            % (name, slot["host"].size, arr.size))
                    np.copyto(slot["host"], arr)
                    cuda.memcpy_htod_async(slot["device"], slot["host"], self._stream)
            if not self._context.execute_async_v3(stream_handle=self._stream.handle):
                # execute_async_v3 returns False on a failed enqueue; raise rather
                # than copy back stale device buffers as if they were valid.
                raise self._err("execute_async_v3 failed for %s" % self.engine_path.name)
            for name, slot in self._outputs.items():
                cuda.memcpy_dtoh_async(slot["host"], slot["device"], self._stream)
            self._stream.synchronize()
            results = {}
            for name, slot in self._outputs.items():
                results[name] = np.array(slot["host"], dtype=np.float32).reshape(slot["shape"])
            return results
        finally:
            self._ctx.pop()
