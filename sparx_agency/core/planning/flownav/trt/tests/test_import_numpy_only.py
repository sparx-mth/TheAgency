"""Guard: importing the FlowNav TRT runtime must not pull heavy deps.

The FALCON Noetic adapter imports ``core`` under Python 3.8 with only numpy
available, so importing this package (and all its public symbols) must not import
``tensorrt``, ``pycuda``, or ``torch`` -- those stay lazy inside the engine
runner. This is the contract that keeps the runtime usable in the ROS1 container.
"""
import sys


FORBIDDEN = ("tensorrt", "pycuda", "torch", "onnx")


def test_import_does_not_pull_heavy_deps():
    for mod in FORBIDDEN:
        sys.modules.pop(mod, None)
    before = set(sys.modules)

    import sparx_agency.core.planning.flownav.trt as pkg  # noqa: F401
    from sparx_agency.core.planning.flownav.trt import (  # noqa: F401
        FlowMatchEulerScheduler, FlowNavError, FlowNavTRTPolicy, TRTEngineRunner,
    )

    pulled = (set(sys.modules) - before) & set(FORBIDDEN)
    assert not pulled, "importing the runtime pulled heavy deps: %s" % sorted(pulled)


def test_public_symbols_present():
    from sparx_agency.core.planning.flownav import trt
    for name in ("FlowNavTRTPolicy", "TRTEngineRunner",
                 "FlowMatchEulerScheduler", "FlowNavError"):
        assert hasattr(trt, name), "missing public symbol %s" % name
