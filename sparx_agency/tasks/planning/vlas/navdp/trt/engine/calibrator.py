"""INT8 entropy calibrator for the multi-input NavDP engines (stretch path).

INT8 is a deferred, on-target stretch goal behind the accuracy gate; FP16 is the
shippable default. When enabled, this calibrator feeds representative inputs to
TensorRT's entropy calibration. It handles multiple named inputs per engine
(``get_batch(names)`` must return device pointers in the order TensorRT asks):

  * encoder: real ``process_image`` / ``process_depth`` captures.
  * denoise / critic: tensors captured from a full FP16 TRT-in-loop run (so the
    ``last_actions`` distribution matches what the quantized graph actually
    sees), NOT a single FP32 reference step.

Calibration data is provided as an ``.npz`` mapping each engine input name to a
stacked array ``(num_samples, *shape)``. Imports TensorRT/pycuda lazily so the
module is importable for inspection without a CUDA stack.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def make_calibrator(input_arrays, cache_path, batch_size=1):
    # type: (Dict[str, np.ndarray], str, int) -> object
    """Construct a TensorRT IInt8EntropyCalibrator2 over ``input_arrays``.

    Args:
        input_arrays: mapping input-tensor-name -> ``(M, *shape)`` float32 stack.
        cache_path: where to read/write the calibration cache.
        batch_size: calibration batch size (1 for the static single-drone graphs).

    Returns:
        A TensorRT calibrator instance.
    """
    import tensorrt as trt
    import pycuda.driver as cuda

    class _NavDPInt8Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self._arrays = {k: np.ascontiguousarray(v, np.float32)
                            for k, v in input_arrays.items()}
            self._n = min(a.shape[0] for a in self._arrays.values())
            self._bs = int(batch_size)
            self._pos = 0
            self._cache = Path(cache_path)
            self._dev = {k: cuda.mem_alloc(a[0].nbytes * self._bs)
                         for k, a in self._arrays.items()}

        def get_batch_size(self):
            return self._bs

        def get_batch(self, names):
            if self._pos + self._bs > self._n:
                return None
            ptrs = []  # type: List[int]
            for name in names:
                arr = self._arrays[name][self._pos:self._pos + self._bs]
                cuda.memcpy_htod(self._dev[name], np.ascontiguousarray(arr, np.float32))
                ptrs.append(int(self._dev[name]))
            self._pos += self._bs
            return ptrs

        def read_calibration_cache(self):
            return self._cache.read_bytes() if self._cache.exists() else None

        def write_calibration_cache(self, cache):
            self._cache.write_bytes(cache)

    return _NavDPInt8Calibrator()
