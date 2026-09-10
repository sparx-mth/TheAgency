"""A camera's pinhole from the one field of view its simulator states: horizontal for Habitat, vertical for AI2-THOR.

RoboTHOR's config gives ``fieldOfView: 63.453``, a vertical angle. Read as
horizontal it gives focal lengths 4/3 too long, and every back-projected point
lands a quarter too close to the optical axis with no error anywhere. And the
helper every adapter calls must not load yaml: it used to, through the
``spatial_math`` it delegated to, which imports yaml at module scope.

Python 3.8 syntax; numpy arrives only through the types.
"""
from __future__ import annotations

import math
import pathlib
import subprocess
import sys

import pytest

from sparx_agency.core.planning.objnav.camera_intrinsics import (
    intrinsics_from_hfov,
    intrinsics_from_vfov,
)
from sparx_agency.core.planning.objnav.errors import ObjNavError

#: The directory that holds ``sparx_agency/``: the fresh interpreter's cwd.
REPO_ROOT = pathlib.Path(__import__("sparx_agency").__file__).resolve().parents[1]
#: The RoboTHOR challenge's vertical field of view, degrees (79 horizontal).
ROBOTHOR_VFOV_DEG = 63.453048374758716
#: What calling the helpers in a fresh interpreter must not load.
HEAVY = ("yaml", "scipy", "torch", "tensorrt", "pycuda", "cv2", "requests",
         "PIL", "rclpy", "rospy", "ompl", "networkx", "skimage")


@pytest.mark.parametrize("width, height, hfov_deg", [
    (640, 480, 79.0), (64, 48, 90.0), (1920, 1080, 79.0), (504, 294, 100.0),
    (100, 37, 45.0), (320, 240, 1.0),
])
def test_the_inline_arithmetic_agrees_with_fov_to_intrinsics(width, height,
                                                             hfov_deg):
    """A deliberate duplicate must not drift from the repo's helper it copies."""
    # Imported here only: spatial_math imports yaml at module scope.
    from sparx_agency.core.common.spatial_math import fov_to_intrinsics
    vfov_deg = math.degrees(2.0 * math.atan(
        math.tan(math.radians(hfov_deg) / 2.0) * height / width))
    theirs = fov_to_intrinsics(width, height, hfov_deg, vfov_deg)
    ours = intrinsics_from_hfov(width, height, hfov_deg)
    assert (ours.width, ours.height, ours.fx, ours.cx, ours.cy) == (
        theirs.width, theirs.height, theirs.fx, theirs.cx, theirs.cy)
    assert ours.fy == ours.fx
    assert ours.fy == pytest.approx(theirs.fy, rel=1e-12)


def test_robothors_vertical_63_degrees_is_79_degrees_horizontal_at_640_by_480():
    """Read as horizontal, the challenge's fieldOfView gives fx 517.6 instead of 388.2."""
    k = intrinsics_from_vfov(640, 480, ROBOTHOR_VFOV_DEG)
    assert k.fx == k.fy
    assert k.fx == pytest.approx(intrinsics_from_hfov(640, 480, 79.0).fx,
                                 rel=1e-9)
    assert k.fx == pytest.approx(388.19, abs=0.01)
    assert (k.cx, k.cy) == (319.5, 239.5)
    misread = intrinsics_from_hfov(640, 480, ROBOTHOR_VFOV_DEG)
    assert misread.fx / k.fx == pytest.approx(4.0 / 3.0, rel=1e-9)


@pytest.mark.parametrize("width, height, vfov_deg", [
    (640, 480, 60.0), (300, 300, 90.0), (224, 171, 45.0), (480, 640, 79.0),
])
def test_a_vertical_field_of_view_sets_the_focal_length_from_the_height(
        width, height, vfov_deg):
    """The defining property of a vertical FOV: fy = (H / 2) / tan(vfov / 2), with square pixels."""
    k = intrinsics_from_vfov(width, height, vfov_deg)
    assert k.fy == pytest.approx(
        (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0), rel=1e-12)
    assert k.fx == k.fy
    assert (k.cx, k.cy) == ((width - 1) / 2.0, (height - 1) / 2.0)


@pytest.mark.parametrize("width, height, vfov_deg", [
    (0, 480, 60.0), (640, 480.0, 60.0), (True, 480, 60.0), (640, 480, 0.0),
    (640, 480, 180.0), (640, 480, -10.0), (640, 480, float("nan")),
    (640, 480, "60"), (640, 480, 5e-324),
])
def test_intrinsics_from_vfov_refuses_a_size_or_field_of_view_with_no_pinhole(
        width, height, vfov_deg):
    """No focal length exists for these; a float size is a unit or shape bug upstream."""
    with pytest.raises(ObjNavError):
        intrinsics_from_vfov(width, height, vfov_deg)


def test_calling_either_helper_in_a_fresh_interpreter_loads_no_yaml():
    """Importing was never the leak: the call imported spatial_math, and with it yaml."""
    code = (
        "import sys\n"
        "from sparx_agency.core.planning.objnav.camera_geometry import (\n"
        "    intrinsics_from_hfov, intrinsics_from_vfov)\n"
        "intrinsics_from_hfov(640, 480, 79.0)\n"
        "intrinsics_from_vfov(640, 480, 63.453)\n"
        "print(','.join(sorted({m.split('.')[0] for m in sys.modules}"
        " & set(%r))))\n" % (HEAVY,))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=str(REPO_ROOT))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", (
        "calling the intrinsics helpers loaded %s" % out.stdout.strip())
