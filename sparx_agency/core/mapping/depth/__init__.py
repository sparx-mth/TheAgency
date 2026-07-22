"""Depth backends + depth/bbox geometry.

The heavy backends (``DepthAnythingV2DepthModel`` -> torch,
``DepthEngineTRT`` -> tensorrt/pycuda) are imported **lazily** (PEP 562) so that
lightweight, dependency-free modules in this package -- notably
``depth_bbox_fusion`` (numpy-only geometry) -- can be imported on hosts without
torch/TensorRT (e.g. the Python-3.8 Noetic adapter). ``from ...depth import
DepthEngineTRT`` still works; it just imports torch/tensorrt at that point.
"""
from __future__ import annotations

import importlib

_LAZY = {
    "DepthAnythingV2DepthModel": ".depth_anything_v2",
    "DepthEngineTRT": ".depth_engine_trt",
    "DepthEngineTRTConfig": ".depth_engine_trt",
}

__all__ = list(_LAZY)


def __getattr__(name):
    """Import a heavy backend on first access (PEP 562)."""
    if name in _LAZY:
        module = importlib.import_module(_LAZY[name], __name__)
        return getattr(module, name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(list(globals().keys()) + __all__)
