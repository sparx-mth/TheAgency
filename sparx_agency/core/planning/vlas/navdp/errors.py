"""The NavDP error type.

Kept in its own dependency-free module at the package root so *both* halves of
the package raise the same class: the HTTP client (``client.py``) and the
TensorRT runtime (``trt/``). Before the VLAs consolidation each defined its own
``NavDPError``, so ``except NavDPError`` imported from one half silently failed
to catch the other's -- two distinct classes sharing a name.

Importing this module pulls in nothing but the standard library, so the FALCON
Noetic adapter (Python 3.8, no numpy guarantee at import) can catch NavDP
failures without loading the TRT runtime.
"""
from sparx_agency.core.planning.vlas.common.errors import VlaError


class NavDPError(VlaError):
    """Raised on any NavDP failure the caller must handle.

    Covers both layers: a transport/decoding failure against the NavDP HTTP
    server, and a TensorRT runtime failure (missing engine, manifest mismatch,
    bad tensor shape, an unsupported batch size / goal mode).
    """
