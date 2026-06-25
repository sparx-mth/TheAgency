"""NavDP point-goal TensorRT runtime (ROS-free, numpy-only at import).

This package is the *runtime* half of the NavDP TensorRT integration: given
pre-built engines it runs point-goal inference with the same numerics as the
PyTorch reference. It is intentionally importable with only ``numpy`` present
(``tensorrt``/``pycuda`` are lazy-imported inside :class:`TRTEngineRunner`) and
stays Python-3.8 compatible, because the FALCON Noetic (ROS1) adapter imports
``core`` under Python 3.8.

The *builder* half (ONNX export, hardware detection, engine build, benchmark,
the TRT-backed server) lives under ``sparx_agency/tasks/planning/navdp`` because
it imports the external PyTorch NavDP model + torch/onnx and is dev/host-only.

Public API:
    * :class:`NavDPTRTPolicy` -- drop-in for ``NavDP_Policy.predict_pointgoal_action``.
    * :class:`TRTEngineRunner` -- minimal TRT-10 fixed-shape engine runner.
    * :class:`NumpyDDPMScheduler` -- diffusers-faithful DDPM sampler in numpy.
    * :class:`NavDPPointEncoder` -- numpy point-goal linear encoder.
    * :class:`NavDPError` -- raised on any runtime failure (no silent fallbacks).
"""
from sparx_agency.core.planning.navdp.trt.engine_runner import TRTEngineRunner
from sparx_agency.core.planning.navdp.trt.errors import NavDPError
from sparx_agency.core.planning.navdp.trt.point_encoder import NavDPPointEncoder
from sparx_agency.core.planning.navdp.trt.policy import NavDPTRTPolicy
from sparx_agency.core.planning.navdp.trt.scheduler import NumpyDDPMScheduler

__all__ = [
    "NavDPTRTPolicy",
    "TRTEngineRunner",
    "NumpyDDPMScheduler",
    "NavDPPointEncoder",
    "NavDPError",
]
