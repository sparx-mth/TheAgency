"""The error InternVLA-N1 raises, and the one thing to know about when it does not.

``InternVlaError`` completes the set :mod:`sparx_agency.core.planning.vlas.common.errors`
describes: one concrete type per policy, all deriving from ``VlaError``, so an
arbiter driving several policies can catch them together while
``except InternVlaError`` keeps catching exactly this one.

**N1 raises far less than NavDP does, on purpose.** A server that drops a frame,
times out, or answers 200 with a body this package cannot read does *not* raise
-- :class:`~sparx_agency.core.planning.vlas.internvla_n1.client.ModelClient`
returns ``StepResponse(success=False)`` and the policy turns that into a not-``ok``
:class:`~sparx_agency.core.planning.vlas.interfaces.policy.PolicyResult` carrying
``transport_failed``. That is deliberate: the model runs behind an external
InternNav server at the far end of a socket, an episode is minutes long, and a
runner that already has a committed route should hold it rather than take an
exception mid-flight. The runner decides; this package reports.

What *does* raise is a caller error -- a step with no frame, a goal of the wrong
kind -- because there is no route to hold and no next frame that would fix it.

Python 3.8 compatible; standard library only.
"""
from sparx_agency.core.planning.vlas.common.errors import VlaError


class InternVlaError(VlaError):
    """InternVLA-N1 was asked for something it cannot answer.

    Raised for a malformed call into the policy (a missing RGB frame, an
    unsupported goal type). Transport and server failures are reported as a
    not-``ok`` result instead -- see the module docstring for why.
    """
