"""FlowNav TensorRT runtime (ROS-free, numpy-only at import).

This package is the *runtime* half of the FlowNav TensorRT integration: given
pre-built engines it runs image-goal inference with the same numerics as the
PyTorch reference. It is intentionally importable with only ``numpy`` present
(``tensorrt``/``pycuda`` are lazy-imported inside :class:`TRTEngineRunner`) and
stays Python-3.8 compatible, because the FALCON Noetic (ROS1) adapter imports
``core`` under Python 3.8.

The *builder* half (ONNX export, hardware detection, engine build, benchmark,
the torch-vs-TRT comparison) lives under ``sparx_agency/tasks/planning/flownav``
because it imports the external PyTorch FlowNav model + torch/onnx and is
dev/host-only.

Public API:
    * :class:`FlowNavTRTPolicy` -- numpy + TensorRT FlowNav image-goal inference.
    * :class:`TRTEngineRunner` -- minimal TRT fixed-shape engine runner.
    * :class:`FlowMatchEulerScheduler` -- deterministic flow-matching integrator.
    * :class:`FlowNavError` -- raised on any runtime failure (no silent fallbacks).
"""
from sparx_agency.core.planning.flownav.trt.engine_runner import TRTEngineRunner
from sparx_agency.core.planning.flownav.trt.errors import FlowNavError
from sparx_agency.core.planning.flownav.trt.policy import FlowNavTRTPolicy
from sparx_agency.core.planning.flownav.trt.scheduler import FlowMatchEulerScheduler

__all__ = [
    "FlowNavTRTPolicy",
    "TRTEngineRunner",
    "FlowMatchEulerScheduler",
    "FlowNavError",
]
