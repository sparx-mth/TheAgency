"""The FlowNav error types.

Kept in a dependency-free module at the package root so both halves of the
package -- the HTTP client (``client.py``) and the TensorRT runtime (``trt/``) --
draw their errors from one place.

FlowNav keeps *two* names because they mark genuinely different failure sites and
callers catch them separately: :class:`FlowNavClientError` for the wire contract
against the host server, :class:`FlowNavError` for the local TensorRT runtime.
Both derive from :class:`~sparx_agency.core.planning.vlas.common.errors.VlaError`
(and therefore still from ``RuntimeError``), so an arbiter driving several
policies can catch them together.
"""
from sparx_agency.core.planning.vlas.common.errors import VlaError


class FlowNavError(VlaError):
    """Raised on any FlowNav TensorRT runtime failure the caller must handle.

    Examples: engine/manifest file missing, an engine built for a different GPU
    compute capability or TensorRT version than the importing runtime, or a
    request for an unsupported batch size / sample count.
    """


class FlowNavClientError(VlaError):
    """Raised on a FlowNav transport / decoding failure the caller must handle."""
