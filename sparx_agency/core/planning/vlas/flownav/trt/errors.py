"""Error type for the FlowNav TensorRT runtime.

The class itself now lives one level up, in
:mod:`sparx_agency.core.planning.vlas.flownav.errors`, alongside the client's
:class:`FlowNavClientError`, so both derive from the shared
:class:`~sparx_agency.core.planning.vlas.common.errors.VlaError`. This module
stays as the import site the builder tooling under ``tasks/planning/vlas/flownav``
already uses, and keeps the property that the whole
``core.planning.vlas.flownav.trt`` package's error can be caught without pulling
in numpy/TensorRT/pycuda.

The runtime raises loudly (never silently falls back) so a missing engine, a
version-locked engine, or a malformed checkpoint surfaces immediately instead of
degrading to a slow or wrong path -- see the repo "prefer raising errors over
silent fallbacks" rule.
"""
from sparx_agency.core.planning.vlas.flownav.errors import FlowNavError

__all__ = ["FlowNavError"]
