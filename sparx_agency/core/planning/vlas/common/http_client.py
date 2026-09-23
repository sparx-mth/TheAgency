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


def _requests():
    """The ``requests`` module, imported on first use (see the module docstring)."""
    import requests
    return requests


class Attempt:
    """One HTTP attempt: the response if it arrived, else why it did not.

    ``_post`` collapses everything that is not a 200 into ``None``, which is the
    right answer for a policy step -- a drop and a 503 are equally "no
    trajectory this frame". It is the wrong answer for a *session* route, where
    201, 409 and a timeout mean three different things and only one of them is a
    failure. Rather than give the shared client a second, subtly different
    ``_post``, both are built on this.
    """

    __slots__ = ("response", "error")

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    @property
    def arrived(self):
        """True if the server answered at all, whatever it answered."""
        return self.response is not None

    @property
    def status(self):
        """HTTP status, or ``None`` if nothing arrived."""
        return None if self.response is None else self.response.status_code

    def timed_out(self):
        """True if the attempt failed specifically on a timeout."""
        if self.error is None:
            return False
        return isinstance(self.error, _requests().exceptions.Timeout)


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

    def __init__(self, url, timeout_s=30.0, logger=None, pooled=False):
        self.url = url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.logger = logger
        self._log = logger if callable(logger) else (lambda *a, **k: None)
        self._pooled = bool(pooled)
        self._session = None

    # ── logging ──────────────────────────────────────────────────────
    def _say(self, level, msg):
        """Log ``msg`` at ``level``, whatever flavour of logger was handed in.

        Two conventions meet here. ROS1 clients pass a bare callable
        (``rospy.logwarn``), which is what ``self._log(fmt, *args)`` serves.
        ROS2 clients pass a logger *object* and need severity dispatch -- and it
        must be **one call site per severity**: ``rclpy`` caches a severity per
        logging call site, so a single shared ``getattr(logger, level)(msg)``
        line raises ``ValueError: Logger severity cannot be changed between
        calls`` the first time a client logs at a second severity, out of
        whatever callback it was in. ``rclpy``'s logger also spells it
        ``warning``, not ``warn``, so a ``getattr`` lookup silently demotes every
        warning to info.
        """
        logger = self.logger
        if logger is None:
            print("[%s] %s" % (level.upper(), msg))
        elif callable(logger):
            logger("%s", msg)
        elif level == "error":
            logger.error(msg)
        elif level in ("warn", "warning"):
            logger.warning(msg)
        elif level == "debug":
            logger.debug(msg)
        else:
            logger.info(msg)

    # ── transport ────────────────────────────────────────────────────
    def _transport(self):
        """The object requests are issued through: a pooled session, or the module.

        Pooling is opt-in because it is a real behaviour change on a live flight
        path (connection reuse, keep-alive) and the ROS1 clients have flown
        without it for a long time. A client posting a large payload several
        times a second -- N1 base64s a pickled observation -- asks for it.
        """
        if not self._pooled:
            return _requests()
        if self._session is None:
            self._session = _requests().Session()
        return self._session

    def _attempt(self, method, route, what, timeout_s=None, quiet=False, **kwargs):
        """Issue one request and report what happened, without judging it.

        Args:
            method: ``"get"`` or ``"post"``.
            route: path beginning with ``/``.
            what: short label for the log line.
            timeout_s: per-call override; defaults to the client's timeout.
            quiet: do not log a transport failure (the caller expects them).
            **kwargs: forwarded to ``requests`` (``json``/``files``/``data``).

        Returns:
            An :class:`Attempt`. A non-200 is *not* a failure here -- see the
            class docstring.
        """
        timeout = self.timeout_s if timeout_s is None else float(timeout_s)
        try:
            r = getattr(self._transport(), method)(
                self.url + route, timeout=timeout, **kwargs)
        except Exception as e:                       # noqa: BLE001
            if not quiet:
                self._log("%s %s failed: %s", self.name(), what, e)
            return Attempt(error=e)
        return Attempt(response=r)

    def _post(self, route, what, **kwargs):
        """POST to ``<url><route>``; return the ``requests`` response or ``None``.

        Args:
            route: path beginning with ``/``, e.g. ``"/pointgoal_step"``.
            what: short label used in the warning log line, e.g. ``"step"``.
            **kwargs: forwarded to ``requests.post`` (``json``/``files``/``data``).

        Returns:
            The response on HTTP 200, else ``None`` (a warning is logged).
        """
        attempt = self._attempt("post", route, what, **kwargs)
        if not attempt.arrived:
            return None
        if attempt.status != 200:
            self._log("%s %s HTTP %s", self.name(), what, attempt.status)
            return None
        return attempt.response

    def _get(self, route, what, **kwargs):
        """GET ``<url><route>``; return the response on HTTP 200, else ``None``."""
        attempt = self._attempt("get", route, what, **kwargs)
        if not attempt.arrived:
            return None
        if attempt.status != 200:
            self._log("%s %s HTTP %s", self.name(), what, attempt.status)
            return None
        return attempt.response

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
