"""HTTP client for the FlowNav image-goal policy server (ROS-free).

FlowNav inference runs on TensorRT, which the FALCON Noetic adapter container has
no access to, so the policy runs as a HOST process (see
``tasks/planning/flownav/server/flownav_trt_server.py``) reached over
``--network host`` + ``127.0.0.1:<port>`` loopback. This module owns the *wire
contract only* -- the single integration seam between the in-container ROS node
and that host server. It holds no ROS and no geometry.

``requests`` and ``Pillow`` are imported lazily inside the methods so importing
this module stays numpy-only (the FALCON Noetic adapter imports ``core`` under
Python 3.8 and must not pull heavy deps at import).

Contract (leaner than NavDP's: image-goal, no depth, no point, no intrinsics)
----------------------------------------------------------------------------
``reset()`` -> ``POST /reset``
    Optional: clears the server's rolling RGB context buffer (call when the goal
    changes / a new episode starts). Stepping works without it.

``step(rgb, goal_rgb)`` -> ``POST /imagegoal_step`` (multipart)
    files: ``image`` = PNG(RGB uint8) current observation frame, ``goal_image`` =
    PNG(RGB uint8) target frame. The server keeps the short frame *history*
    (FlowNav conditions on ``context_size+1`` frames) and does all preprocessing.
    Returns the parsed JSON, e.g. ``{"trajectory": (T, 2), "all_trajectory":
    (N, T, 2), "distance": float}`` -- body-frame ``(forward, left)`` waypoints.
"""
from __future__ import annotations

import io

import numpy as np


class FlowNavClientError(RuntimeError):
    """Raised on a FlowNav transport / decoding failure the caller must handle."""


class FlowNavImageGoalClient:
    """Thin, swappable client for the FlowNav image-goal HTTP server.

    Args:
        url: server base URL, e.g. ``"http://127.0.0.1:8889"``.
        timeout_s: per-request timeout (seconds).
        logger: optional ``logger(fmt, *args)`` callable for transport warnings
            (e.g. ``rospy.logwarn``); defaults to a no-op.
    """

    def __init__(self, url, timeout_s=10.0, logger=None):
        self.url = url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self._log = logger or (lambda *a, **k: None)

    # ── reset ────────────────────────────────────────────────────────
    def reset(self):
        """POST ``/reset`` to clear the server's rolling context buffer.

        Returns:
            ``True`` if the server accepted the reset, else ``False``.
        """
        import requests

        try:
            r = requests.post(self.url + "/reset", json={"batch_size": 1},
                              timeout=self.timeout_s)
            return r.status_code == 200
        except Exception as e:                       # noqa: BLE001
            self._log("FlowNav reset failed: %s", e)
            return False

    # ── step ─────────────────────────────────────────────────────────
    def step(self, rgb, goal_rgb=None):
        """POST ``/imagegoal_step``; return the parsed JSON dict or ``None``.

        Args:
            rgb: HxWx3 uint8 current observation image in RGB order.
            goal_rgb: optional HxWx3 uint8 goal image in RGB order. If ``None``,
                the server uses its configured goal (``--goal-image`` at startup,
                a prior :meth:`set_goal`, or the last goal it received) -- this is
                how the in-container node avoids needing the goal file mounted.

        Returns:
            The decoded response JSON, or ``None`` on any HTTP/transport error.
        """
        import requests

        files = {"image": ("rgb.png", self._png(rgb), "image/png")}
        if goal_rgb is not None:
            files["goal_image"] = ("goal.png", self._png(goal_rgb), "image/png")
        try:
            r = requests.post(self.url + "/imagegoal_step", files=files,
                              timeout=self.timeout_s)
            if r.status_code != 200:
                self._log("FlowNav step HTTP %s", r.status_code)
                return None
            return r.json()
        except Exception as e:                       # noqa: BLE001
            self._log("FlowNav step failed: %s", e)
            return None

    def set_goal(self, goal_rgb):
        """POST ``/set_goal`` to set/replace the server's target goal image.

        Returns:
            ``True`` if the server accepted the goal, else ``False``.
        """
        import requests

        try:
            r = requests.post(
                self.url + "/set_goal",
                files={"goal_image": ("goal.png", self._png(goal_rgb), "image/png")},
                timeout=self.timeout_s)
            return r.status_code == 200
        except Exception as e:                       # noqa: BLE001
            self._log("FlowNav set_goal failed: %s", e)
            return False

    def get_goal(self):
        """GET ``/get_goal``; return the server's goal image (HxWx3 uint8 RGB) or None.

        Used by the display to show the target view when the goal lives on the
        server (``--goal-image``) and the node has no local copy.
        """
        import requests
        from PIL import Image as PILImage

        try:
            r = requests.get(self.url + "/get_goal", timeout=self.timeout_s)
            if r.status_code != 200:
                return None
            return np.asarray(PILImage.open(io.BytesIO(r.content)).convert("RGB"))
        except Exception as e:                       # noqa: BLE001
            self._log("FlowNav get_goal failed: %s", e)
            return None

    @staticmethod
    def _png(arr):
        """RGB uint8 array -> a seek-0 PNG :class:`io.BytesIO` buffer."""
        from PIL import Image as PILImage
        buf = io.BytesIO()
        PILImage.fromarray(np.ascontiguousarray(arr, dtype=np.uint8), "RGB").save(
            buf, format="PNG")
        buf.seek(0)
        return buf

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def best_trajectory(result):
        """Extract the chosen trajectory ``(T, >=2)`` from a step result.

        FlowNav executes sample 0; the server returns it as ``trajectory``.
        Columns are ``(forward, left)`` body-frame waypoints at inference time.

        Raises:
            FlowNavClientError: if the payload is missing or malformed.
        """
        if not result or "trajectory" not in result:
            raise FlowNavClientError("FlowNav result missing 'trajectory'")
        traj = np.asarray(result["trajectory"], dtype=np.float32)
        if traj.ndim == 3:                # (batch, T, C) -> first batch item
            traj = traj[0]
        if traj.ndim != 2 or traj.shape[0] == 0 or traj.shape[1] < 2:
            raise FlowNavClientError("bad FlowNav trajectory shape %r" % (traj.shape,))
        return traj
