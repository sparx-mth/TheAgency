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

What deliberately does **not** live here: schedulers, post-processing and policy
classes. NavDP's DDPM/DDIM sampling with critic ranking and FlowNav's
deterministic flow-matching Euler loop are different algorithms that happen to
have the same shape; merging them would create a parameter soup, not reuse.

Like the rest of ``core``, everything here is numpy-only at import and Python 3.8
compatible; ``tensorrt`` / ``pycuda`` / ``requests`` / ``PIL`` are lazy-imported
inside methods.
"""
