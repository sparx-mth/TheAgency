"""The pinhole of a simulator's camera, from the one field of view its config states.

Simulators state a camera by a single field of view and square pixels, but
not the same one. Habitat's ``hfov`` is **horizontal**. AI2-THOR's
``fieldOfView`` -- Unity's ``Camera.fieldOfView`` -- is **vertical**: the
RoboTHOR challenge config's 63.453 degrees is 79 degrees horizontal at
640x480. Passed as horizontal it gives focal lengths 4/3 too long, which
``CameraSpec`` cannot tell from right, and every back-projected offset
silently shrinks by a quarter. So each convention has its own function, named
for the axis it takes.

Both give ``fx == fy`` exactly, and the pixel-centre principal point
``((W - 1) / 2, (H - 1) / 2)`` of the frames doctrine. Deriving ``fy`` from a
vertical field of view instead would leave it one ulp from ``fx`` for about
half of all image sizes.

The arithmetic deliberately duplicates
``core.common.spatial_math.fov_to_intrinsics``: that module imports yaml at
module scope, so delegating to it made the one function every adapter calls
load yaml, and fail outright where PyYAML is absent. Six lines of
trigonometry are not worth the dependency; the tests pin the two to agree.

Python 3.8 syntax; numpy arrives only through ``core.common.types``.
"""
from __future__ import annotations

import math
import numbers
from typing import Tuple

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.planning.objnav.errors import ObjNavError


def intrinsics_from_hfov(width: int, height: int, hfov_deg: float) -> Intrinsics:
    """The square-pixel pinhole of a ``width`` x ``height`` image with this horizontal FOV.

    Habitat's ``hfov``. ``fx == fy == (width / 2) / tan(hfov / 2)``; the
    principal point is ``((width - 1) / 2, (height - 1) / 2)``.

    Args:
        width: Image width, pixels.
        height: Image height, pixels.
        hfov_deg: Horizontal field of view, degrees, strictly between 0 and
            180.

    Returns:
        The intrinsics, with ``fx == fy`` exactly.

    Raises:
        ObjNavError: On a size that is not a positive integer, or a field of
            view outside ``(0, 180)`` degrees or too narrow for a finite focal
            length.
    """
    w, h = _image_size("intrinsics_from_hfov", width, height)
    _check_fov("intrinsics_from_hfov", "hfov_deg", hfov_deg, "horizontal")
    half_tan = math.tan(math.radians(float(hfov_deg)) / 2.0)
    if not (half_tan > 0.0 and math.isfinite(w / 2.0 / half_tan)):
        raise ObjNavError("intrinsics_from_hfov: hfov_deg=%r is too narrow "
                          "for a finite focal length" % (hfov_deg,))
    focal = (w / 2.0) / half_tan
    return Intrinsics(width=w, height=h, fx=focal, fy=focal,
                      cx=(w - 1) / 2.0, cy=(h - 1) / 2.0)


def intrinsics_from_vfov(width: int, height: int, vfov_deg: float) -> Intrinsics:
    """The square-pixel pinhole of a ``width`` x ``height`` image with this vertical FOV.

    AI2-THOR's ``fieldOfView`` (Unity's is vertical): RoboTHOR's 63.453
    degrees at 640x480 is 79 degrees horizontal. Converted with
    ``hfov = 2 * atan(tan(vfov / 2) * width / height)``, then
    :func:`intrinsics_from_hfov`.

    Args:
        width: Image width, pixels.
        height: Image height, pixels.
        vfov_deg: Vertical field of view, degrees, strictly between 0 and
            180.

    Returns:
        The intrinsics, with ``fx == fy`` exactly.

    Raises:
        ObjNavError: On a size that is not a positive integer, or a field of
            view outside ``(0, 180)`` degrees or one whose horizontal
            counterpart is not.
    """
    w, h = _image_size("intrinsics_from_vfov", width, height)
    _check_fov("intrinsics_from_vfov", "vfov_deg", vfov_deg, "vertical")
    half_tan = math.tan(math.radians(float(vfov_deg)) / 2.0)
    hfov_deg = math.degrees(2.0 * math.atan(half_tan * w / h))
    if not 0.0 < hfov_deg < 180.0:
        raise ObjNavError(
            "intrinsics_from_vfov: vfov_deg=%r at %dx%d is a horizontal field "
            "of view of %r degrees, outside (0, 180)" % (vfov_deg, w, h,
                                                        hfov_deg))
    return intrinsics_from_hfov(w, h, hfov_deg)


def _image_size(caller: str, width, height) -> Tuple[int, int]:
    """``(width, height)`` as ints, or ObjNavError unless both are positive integers."""
    for name, value in (("width", width), ("height", height)):
        if (not isinstance(value, numbers.Integral) or isinstance(value, bool)
                or value <= 0):
            raise ObjNavError(
                "%s: %s must be a positive integer number of pixels, got %r"
                % (caller, name, value))
    return int(width), int(height)


def _check_fov(caller: str, name: str, value, axis: str) -> None:
    """Raise unless ``value`` is a field of view in degrees strictly between 0 and 180."""
    if (not isinstance(value, numbers.Real) or isinstance(value, bool)
            or not 0.0 < value < 180.0):
        raise ObjNavError(
            "%s: %s must be a %s field of view in degrees, strictly between 0 "
            "and 180, got %r" % (caller, name, axis, value))
