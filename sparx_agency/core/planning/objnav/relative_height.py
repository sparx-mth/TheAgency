"""Relative base height from RGB-D registration and episodic planar odometry."""
from __future__ import annotations

import numpy as np

from sparx_agency.core.planning.objnav.camera_geometry import world_T_camera_optical


def _smoothed_luma(rgb):
    """RGB-to-gray and a reflected 3x3 binomial blur, without an OpenCV dependency."""
    # BT.601 weights scaled to 2**15; round to uint8 levels before blurring.
    gray = (rgb @ np.array([9798, 19235, 3735], dtype=np.uint32) + 16384) >> 15
    rows = np.pad(gray, ((1, 1), (0, 0)), mode="reflect")
    cols = np.pad(rows[:-2] + 2 * rows[1:-1] + rows[2:], ((0, 0), (1, 1)), mode="reflect")
    return ((cols[:, :-2] + 2 * cols[:, 1:-1] + cols[:, 2:] + 8) >> 4).astype(np.uint8)


class RelativeHeight:
    """Fit the one unknown translation across a stride; never use an altitude measurement.

    GPS, compass and commanded camera pitch already determine five of the six
    degrees of freedom. Reproject measured points into the next frame under
    candidate rises. Only an unambiguous match of geometry and appearance
    changes the estimate. In particular an untextured wall at constant
    depth says nothing about vertical motion.
    """

    NAME = "relative-rgbd-registration/v1"

    def __init__(self):
        self.height = 0.0
        self.previous = None
        self.accepted = self.unmeasured = 0

    def update(self, rgb, depth, camera, planar_pose, forward):
        # The second frame samples the same surface at subpixel offsets.
        gray = _smoothed_luma(rgb)
        transform = world_T_camera_optical(planar_pose, camera)
        previous = self.previous
        self.previous = gray, depth.copy(), transform
        if previous is None or not forward or np.linalg.norm(transform[:2, 3] - previous[2][:2, 3]) < 0.02:
            return self.height
        dz = self._displacement(previous, self.previous, camera)
        if dz is None:
            self.unmeasured += 1
        else:
            self.accepted += 1
            self.height += dz
        return self.height

    @staticmethod
    def _displacement(old, new, camera):
        gray, depth, transform = old
        now, new_depth, new_transform = new
        h, w = depth.shape
        stride = max(1, int(np.ceil(np.sqrt(w * h / 3000))))
        v, u = np.mgrid[0:h:stride, 0:w:stride]
        d = depth[v, u]
        valid = np.isfinite(d) & (d > camera.min_depth_m) & (d < camera.max_depth_m)
        v, u, d = v[valid], u[valid], d[valid]
        if len(d) < 32:
            return None
        k = camera.intrinsics
        points = np.column_stack(((u - k.cx) * d / k.fx, (v - k.cy) * d / k.fy, d))
        world = points @ transform[:3, :3].T + transform[:3, 3]
        optical = (world - new_transform[:3, 3]) @ new_transform[:3, :3]
        up = new_transform[2, :3]
        intensity = gray[v, u].astype(float)

        def score(rise):
            xyz = optical - rise * up
            z = xyz[:, 2]
            denom = np.maximum(z, 1e-6)
            x = np.rint(k.cx + k.fx * xyz[:, 0] / denom).astype(int)
            y = np.rint(k.cy + k.fy * xyz[:, 1] / denom).astype(int)
            ok = (z > camera.min_depth_m) & (z < camera.max_depth_m) & (x >= 0) & (x < w) & (y >= 0) & (y < h)
            idx = np.flatnonzero(ok)
            measured = new_depth[y[idx], x[idx]]
            idx = idx[(measured > camera.min_depth_m) & (measured < camera.max_depth_m)]
            if len(idx) < max(32, 0.25 * len(d)):
                return 0.0
            y, x = y[idx], x[idx]
            geometry = np.maximum(0, 1 - abs(new_depth[y, x] - z[idx]) / 0.10)
            appearance = np.maximum(0, 1 - abs(now[y, x] - intensity[idx]) / 32)
            return float(np.mean(geometry * appearance))

        coarse = np.linspace(-1.0, 1.0, 41)
        scores = np.array([score(z) for z in coarse])
        peak = coarse[np.argmax(scores)]
        fine = np.clip(peak + np.arange(-5, 6) * .01, -1.0, 1.0)
        refined = np.array([score(z) for z in fine])
        best = int(np.argmax(refined))
        # An isolated peak, not merely high agreement (a blank wall agrees at every height).
        other = scores[np.abs(coarse - fine[best]) >= .09]
        if refined[best] < .50 or refined[best] - other.max() < .035:
            return None
        return round(float(fine[best]), 2)
