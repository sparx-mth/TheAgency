"""VLA-agnostic runtime pieces shared by every policy in this package.

What lives here is code that was, or would otherwise be, copy-pasted per policy:

* :mod:`~sparx_agency.core.planning.vlas.common.errors` -- the :class:`VlaError`
  base every policy's error type derives from.
* :mod:`~sparx_agency.core.planning.vlas.common.image_codec` -- RGB/depth PNG
  encoding, the wire format every HTTP policy server speaks.
* :mod:`~sparx_agency.core.planning.vlas.common.http_client` -- the base class
  for a policy's HTTP wire contract (URL handling, logging, trajectory decode).
* :mod:`~sparx_agency.core.planning.vlas.common.trt` -- the single TensorRT
  engine runner, previously duplicated per policy.
* :mod:`~sparx_agency.core.planning.vlas.common.yaw_search` -- when a
  forward-looking policy answers "stop", turn until it can see somewhere to go.
  Holding position without also looking around is a deadlock: the view never
  changes, so neither does the answer.
* :mod:`~sparx_agency.core.planning.vlas.common.turn_in_place` -- fly a
  policy's discrete turn as a real rotation that ends *stopped*. Its sibling
  above decides where to look; this gets there and says when the aircraft has
  arrived, so the next observation is taken from a standstill -- which is the
  regime a VLN policy was trained in, and not what a bent waypoint flown by a
  holonomic tracker produces.
* :mod:`~sparx_agency.core.planning.vlas.common.plan_commit` -- commit to one
  prediction and fly it as a route, instead of replacing it every frame. Not a
  policy concern and not a robot concern, which is why it is here: a policy
  answers per frame, an aircraft flies a route, and every VLA needs the same
  piece between the two.

What deliberately does **not** live here: schedulers, post-processing and policy
classes. NavDP's DDPM/DDIM sampling with critic ranking and FlowNav's
deterministic flow-matching Euler loop are different algorithms that happen to
have the same shape; merging them would create a parameter soup, not reuse.

Like the rest of ``core``, everything here is numpy-only at import and Python 3.8
compatible; ``tensorrt`` / ``pycuda`` / ``requests`` / ``PIL`` are lazy-imported
inside methods.
"""
