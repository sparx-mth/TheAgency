"""Base for a VLA policy's HTTP wire contract (ROS-free, numpy-only at import).

Every learned policy in this stack runs as a separate process -- on a GPU box, or
on the host next to a ROS1 container that has neither torch nor TensorRT -- and is
reached over HTTP. Each policy owns its own routes and payloads (point-goal vs
image-goal vs language), so this class deliberately defines **no** ``step()``:
it holds only what was identical in every client.

``requests`` is imported lazily inside the methods so importing this module stays
numpy-only (the FALCON Noetic adapter imports ``core`` under Python 3.8 and must
not pull heavy deps at import).

Python 3.8 compatible.
"""
from __future__ import annotations

import numpy as np

from sparx_agency.core.planning.vlas.common.errors import VlaError


class HttpPolicyClient:
    """Shared plumbing for a policy server client: URL, timeout, logging, POST.

    Transport failures are *returned as* ``None``/``False`` rather than raised,
    because a dropped inference frame is normal operation for a policy running at
    video rate over loopback -- the caller re-sends on the next frame. Malformed
    *content* (a response that arrived but cannot be flown) does raise: see
    :meth:`first_batch_trajectory`.

    Args:
        url: server base URL, e.g. ``"http://127.0.0.1:8888"``.
        timeout_s: per-request timeout (seconds).
        logger: optional ``logger(fmt, *args)`` callable for transport warnings
            (e.g. ``rospy.logwarn``); defaults to a no-op.
    """

    #: Exception raised for malformed content. Subclasses override with their own.
    error_cls = VlaError

    def __init__(self, url, timeout_s=30.0, logger=None):
        self.url = url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self._log = logger or (lambda *a, **k: None)

    # ── transport ────────────────────────────────────────────────────
    def _post(self, route, what, **kwargs):
        """POST to ``<url><route>``; return the ``requests`` response or ``None``.

        Args:
            route: path beginning with ``/``, e.g. ``"/pointgoal_step"``.
            what: short label used in the warning log line, e.g. ``"step"``.
            **kwargs: forwarded to ``requests.post`` (``json``/``files``/``data``).

        Returns:
            The response on HTTP 200, else ``None`` (a warning is logged).
        """
        import requests

        try:
            r = requests.post(self.url + route, timeout=self.timeout_s, **kwargs)
        except Exception as e:                       # noqa: BLE001
            self._log("%s %s failed: %s", self.name(), what, e)
            return None
        if r.status_code != 200:
            self._log("%s %s HTTP %s", self.name(), what, r.status_code)
            return None
        return r

    def _post_json(self, route, what, **kwargs):
        """:meth:`_post` then ``.json()``; ``None`` on transport or decode failure."""
        r = self._post(route, what, **kwargs)
        if r is None:
            return None
        try:
            return r.json()
        except Exception as e:                       # noqa: BLE001
            self._log("%s %s: bad JSON: %s", self.name(), what, e)
            return None

    @classmethod
    def name(cls):
        """Short policy name used in log lines (defaults to the class name)."""
        return cls.__name__

    # ── decoding ─────────────────────────────────────────────────────
    @classmethod
    def first_batch_trajectory(cls, result, key="trajectory"):
        """Extract the chosen trajectory ``(T, >=2)`` from a step result.

        Servers return the executed trajectory batched as ``(batch, T, C)``; we
        take the first (and only) batch item. Columns are body-frame
        ``(forward, left[, yaw])`` at inference time. A server that has already
        picked a sample may return it unbatched as ``(T, C)``; both are accepted.

        Raises:
            cls.error_cls: the payload is missing or malformed. This one does
                raise -- a bad-shaped trajectory is a wire-contract break, and
                returning ``None`` would let the caller fly a stale path.
        """
        if not result or key not in result:
            raise cls.error_cls("%s result missing %r" % (cls.name(), key))
        traj = np.asarray(result[key], dtype=np.float32)
        if traj.ndim == 3:                # (batch, T, C) -> first batch item
            traj = traj[0]
        if traj.ndim != 2 or traj.shape[0] == 0 or traj.shape[1] < 2:
            raise cls.error_cls("bad %s trajectory shape %r" % (cls.name(), traj.shape))
        return traj
