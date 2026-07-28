"""Read the one thing FALCON's mapper needs: a depth frame and where it was taken.

Two pieces, and the second is the one that goes wrong silently.

The **depth frame** comes from Isaac's ``distance_to_image_plane`` annotator:
float32 metres, already the perpendicular distance to the image plane rather
than ray length, which is what a pinhole back-projection wants. It is encoded to
uint16 millimetres here -- see
:mod:`~sparx_agency.tasks.planning.falcon_pegasus.link.depth_codec` for what
happens to ``inf`` and ``NaN``, which is a decision about the map, not a detail.

The **camera pose** must be the pose of the *optical* frame, not the aircraft's.
Getting that wrong does not raise anywhere: FALCON builds a complete,
self-consistent map that is rotated ninety degrees. There are two independent
ways to compute it -- from the vehicle's body pose and the known camera mount, or
by asking Isaac for the camera prim's own world pose in the ROS optical
convention -- and :func:`verify_camera_pose` checks they agree once at start-up,
because that is the only cheap moment they can be compared.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from sparx_agency.robots.PEGASUS.adapters.camera_pose import camera_pose_world
from sparx_agency.tasks.planning.falcon_pegasus.link.depth_codec import encode_depth

POSE_AGREEMENT_M = 0.02
POSE_AGREEMENT_RAD = math.radians(1.0)


def depth_bytes(adapter) -> Tuple[bytes, int, int]:
    """The current depth frame, ready for the wire.

    Args:
        adapter: A ``PegasusIrisVehicle``.

    Returns:
        ``(payload, width, height)`` -- uint16 millimetres, row-major.

    Raises:
        RuntimeError: If the camera has not produced a frame yet. It discards its
            first ~100 render ticks, so this means the warm-up was skipped.
    """
    depth = adapter._camera._camera.get_depth()
    if depth is None:
        raise RuntimeError(
            "the camera has not produced a depth frame yet -- SimLoop.warmup_camera() "
            "must run before streaming to FALCON")
    encoded = encode_depth(np.asarray(depth, dtype=np.float32))
    height, width = encoded.shape
    return encoded.tobytes(), width, height


def camera_pose(adapter):
    """The world pose of the camera optical frame, from the vehicle's body pose.

    Args:
        adapter: A ``PegasusIrisVehicle``.

    Returns:
        ``(translation, quaternion_xyzw)`` -- FALCON's ``T_w_c``.
    """
    state = adapter.vehicle.state
    return camera_pose_world(state.position, state.attitude)


def nav_position(adapter):
    """Where the aircraft is, **as far as FALCON is concerned**: at its sensor.

    This is the aircraft's position reported at the camera's optical centre
    rather than at the airframe's body origin, and it is not a cosmetic choice --
    without it the aircraft cannot plan at all.

    The camera is mounted 20 cm forward of the body origin, and it carves free
    space *outward from itself*. So the body origin sits in the one place the
    camera can never observe: 20 cm directly behind it, in every heading, for the
    whole flight. FALCON's A* validates every 10 cm along each candidate step and
    treats UNKNOWN exactly like OCCUPIED, so with the body origin as the start
    every neighbour is rejected on its first checkpoint and the search returns
    NO_PATH before it has expanded a single node. The symptom is
    "planTrajToView: No path to next viewpoint" forever, against a map that is
    visibly fine, for a viewpoint three metres away in open space.

    Reporting the sensor's position instead is not a fudge: FALCON's own map
    configs give ``T_b_c`` **zero translation** -- upstream's camera is at the
    body origin, and every part of the stack that reasons about what the aircraft
    can see (``PerceptionUtils``' frontier visibility model included) assumes so.
    Putting the reported origin at the sensor is what makes that assumption true
    for an airframe whose camera is really 20 cm out.

    The orientation stays the **body's**: FALCON reads yaw from this quaternion
    to decide where the camera is pointing, and the optical frame's own
    quaternion describes a frame rotated into RDF, whose yaw means nothing to it.

    Returns:
        The world-frame ``(x, y, z)`` of the camera's optical centre.
    """
    translation, _quaternion = camera_pose(adapter)
    return translation


def vehicle_state(adapter):
    """Everything the odometry message carries, in FALCON's sensor-origin frame.

    Position is the camera's (see :func:`nav_position`); orientation, velocity
    and angular rate are the airframe's. The linear velocity of the sensor
    differs from the body's by ``omega x r``, at most 0.2 m/s at the yaw rates
    this flies at, and the planner uses it only to seed the initial velocity of
    the next B-spline -- so the body's is close enough and is the one that is
    measured rather than derived.

    Returns:
        ``(position, quaternion_xyzw, linear_velocity_world, angular_velocity_body)``.
    """
    state = adapter.vehicle.state
    return (nav_position(adapter),
            tuple(float(v) for v in state.attitude),
            tuple(float(v) for v in state.linear_velocity),
            tuple(float(v) for v in state.angular_velocity))


def verify_camera_pose(adapter) -> str:
    """Cross-check the computed camera pose against Isaac's own.

    ``Camera.get_world_pose(camera_axes="ros")`` returns the same optical-frame
    pose by an entirely different route -- through the USD stage rather than
    through the mount constants -- so agreement between the two is real evidence
    that the extrinsics are right. Disagreement is the single failure this whole
    module exists to catch, and it is invisible downstream.

    Args:
        adapter: A ``PegasusIrisVehicle``.

    Returns:
        A one-line description of the agreement, for the log.

    Raises:
        RuntimeError: If the two disagree by more than a couple of centimetres or
            a degree, or if Isaac's accessor is unavailable -- in which case the
            check cannot be made and the run should not start blind.
    """
    translation, quaternion = camera_pose(adapter)
    try:
        isaac_position, isaac_quaternion_wxyz = adapter._camera._camera.get_world_pose(
            camera_axes="ros")
    except Exception as error:                                  # pragma: no cover
        raise RuntimeError(
            "could not read the camera's own world pose from Isaac "
            "(Camera.get_world_pose(camera_axes='ros')): %s. Without it the "
            "camera extrinsics fed to FALCON cannot be checked, and a wrong one "
            "produces a confident, wrong map rather than an error." % (error,))

    isaac_position = np.asarray(isaac_position, dtype=float).reshape(-1)[:3]
    isaac_quaternion = np.asarray(isaac_quaternion_wxyz, dtype=float).reshape(-1)[:4]
    # Isaac is scalar-first here; everything in this repo's vehicle path is
    # scalar-last. Conflating the two is a rotation error that looks like noise.
    isaac_xyzw = np.array([isaac_quaternion[1], isaac_quaternion[2],
                           isaac_quaternion[3], isaac_quaternion[0]])

    offset = float(np.linalg.norm(np.asarray(translation) - isaac_position))
    ours = np.asarray(quaternion, dtype=float)
    # q and -q are the same rotation, so compare through the absolute dot product.
    angle = 2.0 * math.acos(min(1.0, abs(float(np.dot(ours, isaac_xyzw)))))

    if offset > POSE_AGREEMENT_M or angle > POSE_AGREEMENT_RAD:
        raise RuntimeError(
            "the camera pose computed from the airframe's mount constants "
            "disagrees with Isaac's own by %.3f m and %.2f deg. One of them is "
            "wrong, and FALCON would map the building in the wrong place without "
            "reporting anything. Check CAMERA_OFFSET_FLU / BODY_TO_OPTICAL in "
            "robots/PEGASUS/adapters/." % (offset, math.degrees(angle)))
    return ("camera extrinsics agree with Isaac to %.1f mm and %.2f deg"
            % (offset * 1000.0, math.degrees(angle)))
