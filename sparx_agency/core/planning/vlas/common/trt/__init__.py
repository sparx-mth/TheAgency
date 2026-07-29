"""Shared TensorRT runtime primitives for every VLA policy.

Holds the single :class:`~sparx_agency.core.planning.vlas.common.trt.engine_runner.TRTEngineRunner`
that NavDP and FlowNav (and any future TRT-backed policy) both use. Before the
VLAs consolidation each policy shipped its own ~230-line copy that differed only
in docstrings and the name of the exception it raised.

Numpy-only at import; ``tensorrt`` and ``pycuda`` are lazy-imported inside the
runner's methods.
"""
from sparx_agency.core.planning.vlas.common.trt.engine_runner import TRTEngineRunner

__all__ = ["TRTEngineRunner"]
