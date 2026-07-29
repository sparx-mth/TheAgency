#!/usr/bin/env python3
"""Read the (x, y) of a ROS1 ``PointCloud2`` as a numpy array, quickly.

FALCON republishes its whole occupancy slab twice a second, so this is on a hot
path: a ``read_points`` generator over a few hundred thousand points costs more
than the rest of the recorder put together. When the cloud is the ordinary
little-endian float32 xyz that PCL produces -- which is every cloud FALCON emits
-- the buffer is viewed in place with a structured dtype and nothing is copied
until the final ``stack``.

A near-identical helper exists at ``tasks/planning/falcon/adapter/scripts/
cloud_utils.py``. It is not shared because catkin ``scripts/`` directories are
not importable packages: each node's siblings are on its path and nothing else
is, so a second catkin package cannot import the first one's helpers. This copy
is deliberately the narrower of the two -- the recorder only ever needs xy.
"""
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointField


def cloud_to_xy(msg):
    """Extract the finite ``(x, y)`` of every point.

    Args:
        msg: A ``sensor_msgs/PointCloud2``.

    Returns:
        An ``(N, 2)`` float32 array. Empty when the cloud is.
    """
    fields = {field.name: field for field in msg.fields}
    usable = (not msg.is_bigendian
              and all(name in fields for name in ("x", "y"))
              and all(fields[name].datatype == PointField.FLOAT32 for name in ("x", "y")))
    if usable:
        dtype = np.dtype({
            "names": ["x", "y"],
            "formats": [np.float32, np.float32],
            "offsets": [fields["x"].offset, fields["y"].offset],
            "itemsize": msg.point_step,
        })
        count = msg.width * msg.height
        view = np.frombuffer(msg.data, dtype, count=count)
        points = np.stack((view["x"], view["y"]), axis=1)
    else:
        points = np.array(
            [(x, y) for x, y, _z in pc2.read_points(
                msg, field_names=("x", "y", "z"), skip_nans=True)],
            dtype=np.float32)
    if points.size == 0:
        return np.empty((0, 2), np.float32)
    return points[np.isfinite(points).all(axis=1)].astype(np.float32, copy=False)
