"""Run a built engine, static or dynamic, for benchmarking and gating.

The repo already has a runtime for engines it *deploys*:
``core.planning.vlas.common.trt.engine_runner.TRTEngineRunner``. That one is
numpy-only at import, Python-3.8-clean and deliberately refuses any engine with
a dynamic dimension, because the systems it serves want deterministic latency
and cannot afford a profile switch inside a control loop. Prefer it in
production.

This runner exists for the other half of the job. A network-agnostic optimizer
has to be able to *measure* whatever it just built -- including the dynamic
engines it now knows how to produce -- without forcing the deployment runtime to
grow a feature its callers do not want. It is a build-time and bench-time tool,
so it may import torch-adjacent things freely and may allocate for the profile's
maximum shape.

Buffers are sized once from the profile's ``max`` and reused, so a per-call
shape change costs a ``set_input_shape`` and nothing else. Output shapes are
read back after the shapes are set, which is the only correct order for a
dynamic engine: before that, the context does not know them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np


class EngineRunner(object):
    """Load one serialized engine and run it with numpy in and numpy out.

    Args:
        engine_path: path to a serialized ``.engine``.
        device_id: CUDA device index.

    Raises:
        RuntimeError: engine missing, deserialization failure, or an execution
            failure. Never returns stale device buffers as if they were a result.
    """

    def __init__(self, engine_path, device_id=0):
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise RuntimeError("engine not found: %s" % self.engine_path)
        self.device_id = int(device_id)
        self._engine = None
        self._context = None
        self._ctx = None
        self._stream = None
        self._inputs = {}
        self._outputs = {}
        self._load()

    # ------------------------------------------------------------------
    def _load(self):
        """Deserialize and allocate one host+device buffer per IO tensor."""
        import tensorrt as trt
        import pycuda.driver as cuda

        cuda.init()
        self._ctx = cuda.Device(self.device_id).retain_primary_context()
        self._ctx.push()
        try:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self._engine = runtime.deserialize_cuda_engine(
                self.engine_path.read_bytes())
            if self._engine is None:
                raise RuntimeError(
                    "failed to deserialize %s -- an engine loads only under the "
                    "exact TensorRT build and compute capability that wrote it"
                    % self.engine_path.name)
            self._context = self._engine.create_execution_context()
            self._stream = cuda.Stream()
            self._allocate(trt, cuda)
        finally:
            self._ctx.pop()

    def _max_shape(self, trt, name):
        """The largest shape this tensor can take, across every profile."""
        shape = tuple(int(d) for d in self._engine.get_tensor_shape(name))
        if all(d >= 0 for d in shape):
            return shape
        best = shape
        for index in range(self._engine.num_optimization_profiles):
            bounds = self._engine.get_tensor_profile_shape(name, index)
            best = tuple(int(d) for d in bounds[-1])       # the MAX bound
        return best

    def _allocate(self, trt, cuda):
        """Allocate for the maximum shape so a per-call resize is free."""
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            dtype = trt.nptype(self._engine.get_tensor_dtype(name))
            is_input = (self._engine.get_tensor_mode(name)
                        == trt.TensorIOMode.INPUT)
            shape = self._max_shape(trt, name) if is_input else None
            if shape is None:
                declared = tuple(int(d) for d in
                                 self._engine.get_tensor_shape(name))
                # An output whose shape depends on the input is sized by the
                # largest input profile; fall back to the declared volume when
                # it is already static.
                shape = declared if all(d >= 0 for d in declared) else None
            slot = {"dtype": dtype, "shape": shape, "host": None, "device": None,
                    "nbytes": 0}
            (self._inputs if is_input else self._outputs)[name] = slot
        self._reserve(trt, cuda)

    def _reserve(self, trt, cuda):
        """Ensure every slot with a known shape has device memory."""
        for name, slot in list(self._inputs.items()) + list(self._outputs.items()):
            if slot["shape"] is None:
                continue
            count = int(np.prod(slot["shape"]))
            nbytes = count * np.dtype(slot["dtype"]).itemsize
            if slot["nbytes"] >= nbytes and slot["device"] is not None:
                continue
            slot["host"] = cuda.pagelocked_empty(count, slot["dtype"])
            slot["device"] = cuda.mem_alloc(max(nbytes, 1))
            slot["nbytes"] = nbytes

    # ------------------------------------------------------------------
    @property
    def input_names(self):
        """Names of the engine's input tensors."""
        return list(self._inputs)

    @property
    def output_names(self):
        """Names of the engine's output tensors."""
        return list(self._outputs)

    def infer(self, feeds):
        """Run the engine on ``feeds`` and return every output as float32.

        Args:
            feeds: mapping of input tensor name -> array. For a dynamic engine
                the array's own shape selects the shape for this call, so the
                caller never has to state it separately.

        Returns:
            Mapping of output tensor name -> array, reshaped to the shape the
            engine reports for this call.

        Raises:
            RuntimeError: on an unknown input name or a failed enqueue.
        """
        import tensorrt as trt
        import pycuda.driver as cuda

        unknown = [n for n in feeds if n not in self._inputs]
        if unknown:
            raise RuntimeError("unknown input(s) %r for %s; have %r"
                               % (unknown, self.engine_path.name,
                                  self.input_names))
        self._ctx.push()
        try:
            for name, array in feeds.items():
                slot = self._inputs[name]
                array = np.ascontiguousarray(array, dtype=slot["dtype"])
                if tuple(array.shape) != tuple(slot["shape"] or ()):
                    slot["shape"] = tuple(array.shape)
                    self._reserve(trt, cuda)
                self._context.set_input_shape(name, tuple(array.shape))
                # Buffers are sized for the profile's max, so a smaller call
                # fills only a prefix of them.
                count = int(array.size)
                np.copyto(slot["host"][:count], array.ravel())
                cuda.memcpy_htod_async(slot["device"], slot["host"][:count],
                                       self._stream)
                self._context.set_tensor_address(name, int(slot["device"]))

            results = self._bind_outputs(trt, cuda)
            if not self._context.execute_async_v3(
                    stream_handle=self._stream.handle):
                raise RuntimeError("execute_async_v3 failed for %s"
                                   % self.engine_path.name)
            for name, slot in self._outputs.items():
                count = int(np.prod(results[name]))
                cuda.memcpy_dtoh_async(slot["host"][:count], slot["device"],
                                       self._stream)
            self._stream.synchronize()
            for name, shape in results.items():
                slot = self._outputs[name]
                count = int(np.prod(shape))
                results[name] = np.array(slot["host"][:count],
                                         dtype=np.float32).reshape(shape)
            return results
        finally:
            self._ctx.pop()

    def _bind_outputs(self, trt, cuda):
        """Size and bind outputs once the input shapes are set. Returns shapes."""
        shapes = {}
        for name, slot in self._outputs.items():
            shape = tuple(int(d) for d in self._context.get_tensor_shape(name))
            if tuple(slot["shape"] or ()) != shape:
                slot["shape"] = shape
                self._reserve(trt, cuda)
            self._context.set_tensor_address(name, int(slot["device"]))
            shapes[name] = shape
        return shapes
