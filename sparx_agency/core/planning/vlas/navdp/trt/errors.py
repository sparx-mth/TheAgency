"""Error type for the NavDP TensorRT runtime.

The class itself now lives one level up, in
:mod:`sparx_agency.core.planning.vlas.navdp.errors`, so the HTTP client and this
runtime raise the *same* ``NavDPError`` rather than two distinct classes that
merely shared a name -- ``except NavDPError`` imported from one half used to miss
the other's. This module stays as the import site the builder tooling under
``tasks/planning/vlas/navdp`` already uses, and keeps the property that the whole
``core.planning.vlas.navdp.trt`` package's error can be caught without pulling in
numpy/TensorRT/pycuda.

The runtime raises loudly (never silently falls back) so a missing engine, a
version-locked engine, or a malformed checkpoint surfaces immediately instead of
degrading to a slow or wrong path -- see the repo "prefer raising errors over
silent fallbacks" rule.
"""
from sparx_agency.core.planning.vlas.navdp.errors import NavDPError

__all__ = ["NavDPError"]
