"""The error base every VLA policy raises from.

Kept dependency-free (standard library only) so any module -- including the
lightweight HTTP clients the FALCON Noetic container imports -- can raise and
catch these without pulling in numpy, TensorRT or pycuda.

Why a shared base
-----------------
Each policy owns a concrete error type (``NavDPError``, ``FlowNavError``,
``InternVlaError``) because callers catch them by name and the message text is
policy-specific. But an arbiter that drives *several* policies -- FALCON's
hybrid/fallback/combination planners do exactly this -- needs one thing to catch.
:class:`VlaError` is that thing.

:class:`VlaError` derives from ``RuntimeError``, which is what every one of these
error types derived from before they were unified, so existing
``except RuntimeError`` handlers keep behaving identically.

Python 3.8 compatible (the FALCON Noetic adapter imports ``core`` under 3.8).
"""


class VlaError(RuntimeError):
    """Base for any VLA policy failure the caller must handle.

    Raised (via a policy-specific subclass) on a transport failure, a malformed
    server response, a missing or mismatched TensorRT engine, or an unsupported
    batch size / goal mode. There are no silent fallbacks in this package: a
    policy that cannot produce a trajectory raises rather than returning a
    plausible-looking default that would be flown.
    """
