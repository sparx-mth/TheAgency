"""The agent's ground-truth pose: where it stands, where it faces, where it looks.

Perfect localisation is the premise of this layer, so this is the simulator's
own ground truth -- converted once, at the environment adapter, into the
repo's frames:

* **World ENU**, ``z`` up, right-handed: the frame of ``Pose2D``, A* and every
  map in ``core``. A simulator that is Y-up (Habitat, AI2-THOR) converts at
  its adapter; nothing downstream knows it was ever Y-up.
* **Yaw** in radians, counter-clockwise about ``+z`` from world ``+x``.
* **Camera pitch** in radians, REP-103: the rotation about the body ``+y``
  (left) axis, so **positive looks down**. Zero is a level camera.

``z`` is the height of the floor under the agent, not of its camera. Scenes
have several floors, and the camera sits ``CameraSpec.height_m`` above this.

Python 3.8 syntax; numpy arrives only through ``core.common.types``.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Tuple

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.core.planning.objnav.types.angles import MAX_ANGLE_RAD


@dataclass(frozen=True)
class AgentPose:
    """Where the agent is and where its camera points, in world ENU.

    Attributes:
        x: Base position east, metres.
        y: Base position north, metres.
        z: Height of the floor under the agent, metres.
        yaw: Heading, radians, counter-clockwise from world ``+x``. Not
            normalised: wrap with ``normalize_angle`` before comparing. At
            most :data:`~sparx_agency.core.planning.objnav.types.angles.MAX_ANGLE_RAD`
            in magnitude.
        camera_pitch: Camera pitch, radians, REP-103 -- positive looks down.
            At most ``MAX_ANGLE_RAD`` in magnitude.

    Raises:
        ObjNavError: If any field is not a finite number (a bool included), so
            that ``except ObjNavError`` catches a NaN pose from an adapter, or
            if the yaw or pitch exceeds ``MAX_ANGLE_RAD`` (wrapping it would
            hang the converter; wrap it at the adapter).
    """

    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0
    camera_pitch: float = 0.0

    def __post_init__(self) -> None:
        for name in ("x", "y", "z", "yaw", "camera_pitch"):
            value = getattr(self, name)
            if (not isinstance(value, numbers.Real) or isinstance(value, bool)
                    or not math.isfinite(value)):
                raise ObjNavError(
                    "AgentPose.%s must be a finite number, got %r"
                    % (name, value))
        for name in ("yaw", "camera_pitch"):
            value = getattr(self, name)
            if abs(value) > MAX_ANGLE_RAD:
                raise ObjNavError(
                    "AgentPose.%s=%r rad exceeds %g rad in magnitude; wrap it "
                    "at the adapter -- wrapping an angle this large would hang "
                    "the converter" % (name, value, MAX_ANGLE_RAD))

    def pose2d(self) -> Pose2D:
        """The planar pose ``(x, y, yaw)`` that planners and maps consume."""
        return Pose2D(self.x, self.y, self.yaw)

    def position(self) -> Tuple[float, float, float]:
        """The base position ``(x, y, z)``, metres."""
        return (self.x, self.y, self.z)
