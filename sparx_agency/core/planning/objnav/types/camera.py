"""The agent's RGB-D camera: pinhole model, mount height and depth range.

RGB and depth are registered -- one pinhole model, one optical centre -- as
they are in Habitat and AI2-THOR. The camera is mounted straight above the
agent's base, ``height_m`` above the floor, and tilts about its own left axis
(see :class:`~sparx_agency.core.planning.objnav.types.pose.AgentPose`).

Depth values follow the repo's convention, the same one the Isaac stub and
FALCON use: the **perpendicular distance to the image plane** (optical-frame
``z``), in metres -- not the length of the ray. ``NaN`` means the pixel has no
reading; ``+inf`` means the sensor saw no surface within ``max_depth_m``. An
adapter converts its simulator's encoding (normalised values, a saturated
maximum, zero for "no return") into this, once -- **both clipped ends
included**: a reading clipped to ``min_depth_m`` (something nearer than the
sensor measures) becomes ``NaN``, and one clipped to ``max_depth_m`` becomes
``+inf``. Left as numbers they read as surfaces at exactly the clip distances
-- Habitat's normalised depth un-normalises its 0 to exactly ``min_depth_m``
and its 1 to exactly ``max_depth_m`` -- and a wall 0.3 m away is mapped at
0.5 m. :func:`~sparx_agency.core.planning.objnav.camera_geometry.backproject_depth`
drops a reading equal to either bound, but a method reading the image itself
would not.

Python 3.8 syntax; numpy arrives only through ``core.common.types``.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.planning.objnav.errors import ObjNavError


def _real(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


@dataclass(frozen=True)
class CameraSpec:
    """One registered RGB-D camera.

    Attributes:
        intrinsics: The pinhole model of both images.
        height_m: Height of the optical centre above the agent's base
            (``AgentPose.z``), metres.
        min_depth_m: Nearest distance the depth sensor measures, metres. A
            reading equal to it is a clipped return, not a measurement.
        max_depth_m: Farthest distance the depth sensor measures, metres; a
            reading equal to it is a clipped return too. May be ``inf`` for a
            sensor without a range limit.

    Raises:
        ObjNavError: On a non-positive image size or focal length, a
            non-finite principal point, a negative mount height, or a depth
            range that is negative or empty.
    """

    intrinsics: Intrinsics
    height_m: float
    min_depth_m: float
    max_depth_m: float

    def __post_init__(self) -> None:
        k = self.intrinsics
        if not isinstance(k, Intrinsics):
            raise ObjNavError(
                "CameraSpec.intrinsics must be an Intrinsics, got %r" % (k,))
        for name in ("width", "height"):
            value = getattr(k, name)
            if (not isinstance(value, numbers.Integral)
                    or isinstance(value, bool) or value <= 0):
                raise ObjNavError(
                    "CameraSpec.intrinsics.%s must be a positive integer, got "
                    "%r" % (name, value))
        for name in ("fx", "fy"):
            value = getattr(k, name)
            if not (_real(value) and math.isfinite(value) and value > 0):
                raise ObjNavError(
                    "CameraSpec.intrinsics.%s must be positive and finite, got "
                    "%r" % (name, value))
        for name in ("cx", "cy"):
            value = getattr(k, name)
            if not (_real(value) and math.isfinite(value)):
                raise ObjNavError(
                    "CameraSpec.intrinsics.%s must be finite, got %r"
                    % (name, value))
        if not (_real(self.height_m) and math.isfinite(self.height_m)
                and self.height_m >= 0):
            raise ObjNavError(
                "CameraSpec.height_m must be finite and non-negative -- a "
                "camera below its own base is a sign error -- got %r"
                % (self.height_m,))
        if not (_real(self.min_depth_m) and math.isfinite(self.min_depth_m)
                and self.min_depth_m >= 0):
            raise ObjNavError(
                "CameraSpec.min_depth_m must be finite and non-negative, got "
                "%r" % (self.min_depth_m,))
        if not (_real(self.max_depth_m) and not math.isnan(self.max_depth_m)
                and self.max_depth_m > self.min_depth_m):
            raise ObjNavError(
                "CameraSpec.max_depth_m must exceed min_depth_m=%r, got %r"
                % (self.min_depth_m, self.max_depth_m))
