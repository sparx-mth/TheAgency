#!/usr/bin/env python3
"""ROS1 PointCloud2 -> (N,3) float32 numpy, fast path + safe fallback.

(ROS2 stacks use tasks/mapping/ros2/helpers.py with sensor_msgs_py; the
FALCON adapter runs in Noetic, so it uses sensor_msgs.point_cloud2 here.)
"""
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointField

_F32 = PointField.FLOAT32


def cloud_to_xyz(msg) -> np.ndarray:
    """Extract finite (x,y,z) as (N,3) float32.

    Uses a zero-copy structured view when the cloud is little-endian float32
    xyz (~100x faster than read_points for large clouds); otherwise falls back
    to pc2.read_points for exotic layouts.
    """
    try:
        f = {p.name: p for p in msg.fields}
        if (not msg.is_bigendian
                and all(k in f for k in ("x", "y", "z"))
                and all(f[k].datatype == _F32 for k in ("x", "y", "z"))):
            dt = np.dtype({"names": ["x", "y", "z"],
                           "formats": [np.float32, np.float32, np.float32],
                           "offsets": [f["x"].offset, f["y"].offset, f["z"].offset],
                           "itemsize": msg.point_step})
            n = msg.width * msg.height
            a = np.frombuffer(msg.data, dt, count=n)
            xyz = np.stack((a["x"], a["y"], a["z"]), axis=1)
        else:
            raise ValueError("non-float32/bigendian cloud")
    except Exception:
        xyz = np.array(list(pc2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True)), np.float32)
    if xyz.size == 0:
        return np.empty((0, 3), np.float32)
    return xyz[np.isfinite(xyz).all(axis=1)].astype(np.float32, copy=False)