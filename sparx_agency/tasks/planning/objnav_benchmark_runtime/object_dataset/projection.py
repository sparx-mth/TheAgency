"""Where on the floor plan a detector box is: its median depth projected through the camera pose. numpy only; habitat's yaw."""
import math

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.geometry import forward_xz


def box_floor_point(xyxy, depth, camera_xz_yaw, width, height, hfov_deg):
    """(x, z) of the box's middle from the median valid depth of its inner half, or None when it has no valid depth."""
    x0, y0, x1, y1 = xyxy
    centre_u, centre_v = (x0 + x1) / 2, (y0 + y1) / 2
    inner = depth[int(max(0, centre_v - (y1 - y0) / 4)):int(min(height, centre_v + (y1 - y0) / 4)) + 1,
                  int(max(0, centre_u - (x1 - x0) / 4)):int(min(width, centre_u + (x1 - x0) / 4)) + 1]
    valid = inner[np.isfinite(inner) & (inner > 0)]
    if not valid.size:
        return None
    metres = float(np.median(valid))
    focal = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    sideways = (centre_u - width / 2) / focal * metres
    cam_x, cam_z, yaw = camera_xz_yaw
    ahead, right = forward_xz(yaw), (math.cos(yaw), -math.sin(yaw))
    return (cam_x + metres * ahead[0] + sideways * right[0], cam_z + metres * ahead[1] + sideways * right[1])
