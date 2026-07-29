"""The trt runtime package must import with only numpy present.

``core`` is imported by the FALCON Noetic adapter under Python 3.8 with no
torch/tensorrt/pycuda. This test guards the contract that importing the NavDP
trt runtime does NOT eagerly import any heavy dependency -- they must be lazy
(inside methods), mirroring how ``core.planning.vlas.navdp.client`` lazy-imports
``requests``/``PIL``.
"""
from __future__ import annotations

import sys


def test_import_does_not_pull_heavy_deps():
    # Importing the package (and all its public modules) must succeed and must
    # not have imported tensorrt / pycuda / torch as a side effect.
    before = set(sys.modules)
    import sparx_agency.core.planning.vlas.navdp.trt as trt_pkg  # noqa: F401
    from sparx_agency.core.planning.vlas.navdp.trt import (  # noqa: F401
        NavDPTRTPolicy, TRTEngineRunner, NumpyDDPMScheduler, NavDPPointEncoder,
        NavDPError,
    )
    newly = set(sys.modules) - before
    forbidden = {m for m in newly if m.split(".")[0] in ("tensorrt", "pycuda", "torch")}
    assert not forbidden, "trt runtime import pulled heavy deps: %r" % sorted(forbidden)


def test_public_symbols_present():
    import sparx_agency.core.planning.vlas.navdp.trt as trt_pkg
    for name in ("NavDPTRTPolicy", "TRTEngineRunner", "NumpyDDPMScheduler",
                 "NavDPPointEncoder", "NavDPError"):
        assert hasattr(trt_pkg, name)
