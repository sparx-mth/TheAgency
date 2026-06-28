"""HTTP client for the NavDP point-goal policy server (ROS-free).

NavDP runs as an HTTP service on a GPU box (`InternRobotics/NavDP
<https://github.com/InternRobotics/NavDP>`_; ``eval_*_wheeled.py`` exposes
``/navigator_reset`` and ``/pointgoal_step``). This module owns the *wire
contract only* -- the single integration seam between our stack and that server.
It holds no ROS and no geometry. ``requests`` and ``Pillow`` are imported lazily
inside the methods so importing this module stays numpy-only (the FALCON Noetic
adapter imports ``core`` under Python 3.8 and must not pull heavy deps at import).

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
into uint16. uint16 caps at 65535, so ``65535 / 10000 = 6.5535`` m is the hard
ceiling -- a 7 m pixel would overflow and read back as ~0.45 m, a phantom wall
right where the operator clicked the far floor. NavDP's own ``process_depth``
zeroes depth beyond 5 m anyway, so 5 m is the honest default cap. Widen only by
also widening the encoding (uint32, or a smaller scale) on BOTH sides.
"""
from __future__ import annotations

import io
import json

import numpy as np

# depth_m -> uint16 multiplier; the server divides by the same value.
DEPTH_SCALE = 10000.0


class NavDPError(RuntimeError):
    """Raised on a NavDP transport / decoding failure the caller must handle."""


class NavDPPointgoalClient:
    """Thin, swappable client for the NavDP point-goal HTTP server.

    Args:
        url: server base URL, e.g. ``"http://127.0.0.1:8888"``.
        timeout_s: per-request timeout (seconds).
        depth_max_m: depth clip before uint16 encoding (see module docstring).
        logger: optional ``logger(fmt, *args)`` callable for transport warnings
            (e.g. ``rospy.logwarn``); defaults to a no-op.
    """

    def __init__(self, url, timeout_s=30.0, depth_max_m=5.0, logger=None):
        self.url = url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.depth_max_m = float(depth_max_m)
        # Fail loud rather than silently overflow: depth_m * DEPTH_SCALE must fit
        # in uint16. Beyond 65535/DEPTH_SCALE a far pixel wraps to a tiny value --
        # a phantom wall right where the operator clicked the far floor.
        max_cap = 65535.0 / DEPTH_SCALE
        if self.depth_max_m > max_cap:
            raise ValueError(
                "depth_max_m=%.3f exceeds the uint16 encoding ceiling %.4f m "
                "(DEPTH_SCALE=%g); lower it or widen the encoding on both sides."
                % (self.depth_max_m, max_cap, DEPTH_SCALE))
        self._log = logger or (lambda *a, **k: None)

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
        import requests

        K = [[intrinsics.fx, 0.0, intrinsics.cx],
             [0.0, intrinsics.fy, intrinsics.cy],
             [0.0, 0.0, 1.0]]
        try:
            r = requests.post(self.url + "/navigator_reset",
                              json={"intrinsic": K,
                                    "stop_threshold": stop_threshold,
                                    "batch_size": batch_size},
                              timeout=self.timeout_s)
            return r.status_code == 200
        except Exception as e:                       # noqa: BLE001
            self._log("NavDP reset failed: %s", e)
            return False

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
        import requests
        from PIL import Image as PILImage

        rgb_buf = io.BytesIO()
        PILImage.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), "RGB").save(
            rgb_buf, format="PNG")
        rgb_buf.seek(0)

        d_int = (np.clip(depth, 0.0, self.depth_max_m) * DEPTH_SCALE).astype(np.uint16)
        d_buf = io.BytesIO()
        PILImage.fromarray(d_int, mode="I;16").save(d_buf, format="PNG")
        d_buf.seek(0)

        goal = {"goal_x": [float(gx)], "goal_y": [float(gy)],
                "click_px": int(click_px), "click_py": int(click_py)}
        if altitude is not None:
            goal["altitude"] = float(altitude)

        try:
            r = requests.post(
                self.url + "/pointgoal_step",
                files={"image": ("rgb.png", rgb_buf, "image/png"),
                       "depth": ("depth.png", d_buf, "image/png")},
                data={"goal_data": json.dumps(goal)},
                timeout=self.timeout_s)
            if r.status_code != 200:
                self._log("NavDP step HTTP %s", r.status_code)
                return None
            return r.json()
        except Exception as e:                       # noqa: BLE001
            self._log("NavDP step failed: %s", e)
            return None

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
        if not result or "trajectory" not in result:
            raise NavDPError("NavDP result missing 'trajectory'")
        traj = np.asarray(result["trajectory"], dtype=np.float32)
        if traj.ndim == 3:                # (batch, T, C) -> first batch item
            traj = traj[0]
        if traj.ndim != 2 or traj.shape[0] == 0 or traj.shape[1] < 2:
            raise NavDPError("bad NavDP trajectory shape %r" % (traj.shape,))
        return traj
