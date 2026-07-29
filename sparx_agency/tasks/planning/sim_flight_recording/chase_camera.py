"""An external camera that follows the drone, for watching/recording a flight.

This is *not* the drone's own onboard camera (that one is part of the vehicle,
see ``robots/PEGASUS/adapters/vehicle.py``, and is what recordings capture).
This one exists purely so a human can see the aircraft in its environment.

It chases the vehicle at a fixed world-frame offset rather than sitting at a
fixed world position: a fixed position risks clipping into unknown room
geometry, since a scene's interior bounds aren't known in advance, whereas the
vehicle's own position is always valid collision-free airspace -- that is what
"it is flying" means.
"""
from __future__ import annotations

import numpy as np

DEFAULT_RESOLUTION = (960, 540)
DEFAULT_OFFSET = (0.0, 1.2, 0.6)  # world-frame metres, behind-left and above


def make_chase_camera(prim_path: str = "/World/ChaseCamera", resolution=DEFAULT_RESOLUTION):
    """Create and initialize the chase camera on the stage."""
    from isaacsim.sensors.camera.camera import Camera

    camera = Camera(prim_path=prim_path, resolution=resolution)
    camera.initialize()
    return camera


def aim_chase_camera(camera, vehicle_position, world_offset=DEFAULT_OFFSET) -> None:
    """Point ``camera`` at ``vehicle_position`` from a fixed world-frame offset.

    Takes effect on the next render.
    """
    from scipy.spatial.transform import Rotation

    vehicle_position = np.asarray(vehicle_position, dtype=np.float64)
    position = vehicle_position + np.asarray(world_offset, dtype=np.float64)
    forward = vehicle_position - position
    forward /= np.linalg.norm(forward)
    left = np.cross([0.0, 0.0, 1.0], forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    rot_matrix = np.column_stack([forward, left, up])  # local FLU axes in the world frame
    qx, qy, qz, qw = Rotation.from_matrix(rot_matrix).as_quat()
    camera.set_world_pose(position=position, orientation=[qw, qx, qy, qz], camera_axes="world")
