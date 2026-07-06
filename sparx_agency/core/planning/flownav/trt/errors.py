"""Error type for the FlowNav TensorRT runtime.

Kept in its own module so the whole ``core.planning.flownav.trt`` package can
import the error without pulling in numpy/TensorRT/pycuda. The runtime raises
loudly (never silently falls back) so a missing engine, a version-locked engine,
or a malformed checkpoint surfaces immediately instead of degrading to a slow or
wrong path -- see the repo "prefer raising errors over silent fallbacks" rule.
"""
from __future__ import annotations


class FlowNavError(RuntimeError):
    """Raised on any FlowNav TensorRT runtime failure the caller must handle.

    Examples: engine/manifest file missing, an engine built for a different GPU
    compute capability or TensorRT version than the importing runtime, a
    checkpoint whose weights do not match the exported graphs, or a request for
    an unsupported batch size / sample count.
    """
