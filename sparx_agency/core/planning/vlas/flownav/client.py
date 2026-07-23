"""HTTP client for the FlowNav image-goal policy server (ROS-free).

FlowNav inference runs on TensorRT, which the FALCON Noetic adapter container has
no access to, so the policy runs as a HOST process (see
``tasks/planning/vlas/flownav/serve/flownav_trt_server.py``) reached over
``--network host`` + ``127.0.0.1:<port>`` loopback. This module owns the *wire
contract only* -- the single integration seam between the in-container ROS node
and that host server. It holds no ROS and no geometry.

``requests`` and ``Pillow`` are imported lazily (inside
:class:`~sparx_agency.core.planning.vlas.common.http_client.HttpPolicyClient` and
:mod:`~sparx_agency.core.planning.vlas.common.image_codec`) so importing this
module stays numpy-only -- the FALCON Noetic adapter imports ``core`` under
Python 3.8 and must not pull heavy deps at import.

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

from sparx_agency.core.planning.vlas.common.http_client import HttpPolicyClient
from sparx_agency.core.planning.vlas.common.image_codec import png_to_rgb, rgb_to_png
from sparx_agency.core.planning.vlas.flownav.errors import FlowNavClientError

__all__ = ["FlowNavClientError", "FlowNavImageGoalClient"]


class FlowNavImageGoalClient(HttpPolicyClient):
    """Thin, swappable client for the FlowNav image-goal HTTP server.

    Args:
        url: server base URL, e.g. ``"http://127.0.0.1:8889"``.
        timeout_s: per-request timeout (seconds).
        logger: optional ``logger(fmt, *args)`` callable for transport warnings
            (e.g. ``rospy.logwarn``); defaults to a no-op.
    """

    error_cls = FlowNavClientError

    def __init__(self, url, timeout_s=10.0, logger=None):
        HttpPolicyClient.__init__(self, url, timeout_s=timeout_s, logger=logger)

    @classmethod
    def name(cls):
        """Short policy name used in transport log lines."""
        return "FlowNav"

    # ── reset ────────────────────────────────────────────────────────
    def reset(self):
        """POST ``/reset`` to clear the server's rolling context buffer.

        Returns:
            ``True`` if the server accepted the reset, else ``False``.
        """
        return self._post("/reset", "reset", json={"batch_size": 1}) is not None

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
        files = {"image": ("rgb.png", rgb_to_png(rgb), "image/png")}
        if goal_rgb is not None:
            files["goal_image"] = ("goal.png", rgb_to_png(goal_rgb), "image/png")
        return self._post_json("/imagegoal_step", "step", files=files)

    def set_goal(self, goal_rgb):
        """POST ``/set_goal`` to set/replace the server's target goal image.

        Returns:
            ``True`` if the server accepted the goal, else ``False``.
        """
        return self._post(
            "/set_goal", "set_goal",
            files={"goal_image": ("goal.png", rgb_to_png(goal_rgb), "image/png")},
        ) is not None

    def get_goal(self):
        """GET ``/get_goal``; return the server's goal image (HxWx3 uint8 RGB) or None.

        Used by the display to show the target view when the goal lives on the
        server (``--goal-image``) and the node has no local copy.
        """
        import requests

        try:
            r = requests.get(self.url + "/get_goal", timeout=self.timeout_s)
            if r.status_code != 200:
                return None
            return png_to_rgb(r.content)
        except Exception as e:                       # noqa: BLE001
            self._log("FlowNav get_goal failed: %s", e)
            return None

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def best_trajectory(result):
        """Extract the chosen trajectory ``(T, >=2)`` from a step result.

        FlowNav executes sample 0; the server returns it as ``trajectory``.
        Columns are ``(forward, left)`` body-frame waypoints at inference time.

        Raises:
            FlowNavClientError: if the payload is missing or malformed.
        """
        return FlowNavImageGoalClient.first_batch_trajectory(result)
