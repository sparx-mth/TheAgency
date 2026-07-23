"""HTTP client for the NavDP point-goal policy server (ROS-free).

NavDP runs as an HTTP service on a GPU box (`InternRobotics/NavDP
<https://github.com/InternRobotics/NavDP>`_; ``eval_*_wheeled.py`` exposes
``/navigator_reset`` and ``/pointgoal_step``). This module owns the *wire
contract only* -- the single integration seam between our stack and that server.
It holds no ROS and no geometry. ``requests`` and ``Pillow`` are imported lazily
(inside :class:`~sparx_agency.core.planning.vlas.common.http_client.HttpPolicyClient`
and :mod:`~sparx_agency.core.planning.vlas.common.image_codec`) so importing this
module stays numpy-only -- the FALCON Noetic adapter imports ``core`` under
Python 3.8 and must not pull heavy deps at import.

Contract
--------
``reset(intrinsics)`` -> ``POST /navigator_reset``
    JSON ``{"intrinsic": [[fx,0,cx],[0,fy,cy],[0,0,1]], "stop_threshold",
    "batch_size"}``. Call once before stepping so the server knows the camera
    model (the same intrinsics the RGB/depth frames are captured with).

``pointgoal_step(rgb, depth, gx, gy)`` -> ``POST /pointgoal_step`` (multipart)
    files: ``image`` = PNG(RGB uint8), ``depth`` = PNG(uint16, ``depth_m *
    DEPTH_SCALE``); data: ``goal_data`` = JSON ``{"goal_x":[gx], "goal_y":[gy],
    "click_px", "click_py", "altitude"}``. Returns the parsed JSON, e.g.
    ``{"trajectory": (1, T, >=2), "all_values": (1, K)}`` and optionally
    ``"all_trajectory"`` / ``"trajectory_mask"``.

Depth encoding (do NOT widen without matching the server)
---------------------------------------------------------
Depth is clipped to ``depth_max_m`` then scaled by :data:`DEPTH_SCALE` (10000)
into uint16, capping the honest range at ~6.55 m -- shared with every other VLA
client in :mod:`~sparx_agency.core.planning.vlas.common.image_codec`, whose
module docstring explains the phantom-wall failure mode in full.
"""
from __future__ import annotations

import json

from sparx_agency.core.planning.vlas.common.http_client import HttpPolicyClient
from sparx_agency.core.planning.vlas.common.image_codec import (
    DEPTH_SCALE,
    check_depth_cap,
    depth_to_png,
    rgb_to_png,
)
from sparx_agency.core.planning.vlas.navdp.errors import NavDPError

__all__ = ["DEPTH_SCALE", "NavDPError", "NavDPPointgoalClient"]


class NavDPPointgoalClient(HttpPolicyClient):
    """Thin, swappable client for the NavDP point-goal HTTP server.

    Args:
        url: server base URL, e.g. ``"http://127.0.0.1:8888"``.
        timeout_s: per-request timeout (seconds).
        depth_max_m: depth clip before uint16 encoding (see the module docstring).
        logger: optional ``logger(fmt, *args)`` callable for transport warnings
            (e.g. ``rospy.logwarn``); defaults to a no-op.

    Raises:
        ValueError: ``depth_max_m`` exceeds the uint16 encoding ceiling. Fail
            loud rather than silently overflow -- beyond ``65535/DEPTH_SCALE`` a
            far pixel wraps to a tiny value, a phantom wall right where the
            operator clicked the far floor.
    """

    error_cls = NavDPError

    def __init__(self, url, timeout_s=30.0, depth_max_m=5.0, logger=None):
        HttpPolicyClient.__init__(self, url, timeout_s=timeout_s, logger=logger)
        self.depth_max_m = check_depth_cap(depth_max_m)

    @classmethod
    def name(cls):
        """Short policy name used in transport log lines."""
        return "NavDP"

    # ── reset ────────────────────────────────────────────────────────
    def reset(self, intrinsics, stop_threshold=-999, batch_size=1):
        """POST ``/navigator_reset`` with the camera intrinsic matrix.

        Args:
            intrinsics: camera :class:`Intrinsics` for the RGB/depth stream.
            stop_threshold: server stop-critic threshold (``-999`` disables the
                server-side stop so it always returns a full trajectory).
            batch_size: NavDP batch size (1 for a single drone).

        Returns:
            ``True`` if the server accepted the reset, else ``False``.
        """
        K = [[intrinsics.fx, 0.0, intrinsics.cx],
             [0.0, intrinsics.fy, intrinsics.cy],
             [0.0, 0.0, 1.0]]
        return self._post("/navigator_reset", "reset",
                          json={"intrinsic": K,
                                "stop_threshold": stop_threshold,
                                "batch_size": batch_size}) is not None

    # ── step ─────────────────────────────────────────────────────────
    def pointgoal_step(self, rgb, depth, gx, gy, click_px=-1, click_py=-1,
                       altitude=None):
        """POST ``/pointgoal_step``; return the parsed JSON dict or ``None``.

        Args:
            rgb: HxWx3 uint8 image in RGB order.
            depth: HxW float metric depth (m), aligned to ``rgb``.
            gx, gy: body-frame point-goal ``(forward, left)`` in NavDP's range.
            click_px, click_py: clicked pixel (forwarded for the server overlay).
            altitude: optional drone altitude (m) the server may use to render its
                trajectory mask on the matching ground plane.

        Returns:
            The decoded response JSON, or ``None`` on any HTTP/transport error.
        """
        goal = {"goal_x": [float(gx)], "goal_y": [float(gy)],
                "click_px": int(click_px), "click_py": int(click_py)}
        if altitude is not None:
            goal["altitude"] = float(altitude)

        return self._post_json(
            "/pointgoal_step", "step",
            files={"image": ("rgb.png", rgb_to_png(rgb), "image/png"),
                   "depth": ("depth.png",
                             depth_to_png(depth, self.depth_max_m), "image/png")},
            data={"goal_data": json.dumps(goal)})

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def best_trajectory(result):
        """Extract the chosen trajectory ``(T, >=2)`` from a step result.

        The server returns ``trajectory`` batched as ``(batch, T, C)``; we take the
        first (and only) batch item. Columns are ``(forward, left[, yaw])`` in the
        body frame at inference time.

        Raises:
            NavDPError: if the payload is missing or malformed.
        """
        return NavDPPointgoalClient.first_batch_trajectory(result)
