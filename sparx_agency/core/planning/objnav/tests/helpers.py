"""Small valid objects the ObjectNav type tests build on, each with any field replaced.

Shared by the type tests so that the camera every observation and every
episode is checked against has one definition: an 8x6 pinhole with its
principal point at the image centre, at a LoCoBot-like mounting height. A test
module, not a library: nothing outside these tests imports it.

Python 3.8 syntax; numpy arrives only through the types.
"""
from __future__ import annotations

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.planning.objnav.types.camera import CameraSpec

#: Not a number: compares false with everything, so every refusal is tested with it.
NAN = float("nan")
#: Positive infinity: legal for a few fields (an unlimited depth range), refused elsewhere.
INF = float("inf")


def intrinsics(**overrides):
    """An 8x6 pinhole with its principal point at the image centre."""
    fields = dict(width=8, height=6, fx=4.0, fy=4.0, cx=3.5, cy=2.5)
    fields.update(overrides)
    return Intrinsics(**fields)


def camera(**overrides):
    """A valid camera, with any field replaced."""
    fields = dict(intrinsics=intrinsics(), height_m=0.88, min_depth_m=0.1,
                  max_depth_m=5.0)
    fields.update(overrides)
    return CameraSpec(**fields)
