"""Camera geometry pinned against points worked out on paper, in every frame the doctrine names.

Each test puts a surface where the answer is obvious -- dead ahead, to one
side, below, with the camera pitched or the agent turned -- because a sign or
frame slip here does not crash anything. It builds a map in which objects sit
behind, above or beside where they really are, and the score quietly drops.
"""
from __future__ import annotations

import math
import pathlib
import subprocess
import sys
import warnings

import numpy as np
import pytest

from sparx_agency.core.common.types import Intrinsics, Pose2D
from sparx_agency.core.planning.objnav.camera_geometry import (
    backproject_depth,
    camera_position_world,
    camera_rotation_world,
    intrinsics_from_hfov,
    project_to_image,
    world_T_camera_optical,
)
from sparx_agency.core.planning.objnav.errors import ObjNavError, ObservationError
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.core.planning.objnav.types.pose import AgentPose

#: The directory that holds ``sparx_agency/``: the fresh interpreter's cwd.
REPO_ROOT = pathlib.Path(__import__("sparx_agency").__file__).resolve().parents[1]

#: Odd, so the principal point ((W - 1) / 2, (H - 1) / 2) is a pixel centre.
WIDTH, HEIGHT = 65, 49
#: That pixel.
CX, CY = 32, 24
#: Mount height of the default camera, metres (Habitat's ObjectNav agent).
MOUNT_M = 0.88
#: A base that is off the origin and on a raised floor, so a dropped term shows.
BASE = dict(x=1.0, y=-2.0, z=0.5)
#: What a fresh ``import`` of the module must not pull in.
HEAVY = ("yaml", "scipy", "torch", "tensorrt", "pycuda", "cv2", "requests",
         "PIL", "rclpy", "rospy", "ompl", "networkx", "skimage")


def make_camera(width=WIDTH, height=HEIGHT, hfov_deg=90.0, min_depth_m=0.1,
                max_depth_m=10.0):
    """A registered RGB-D camera at Habitat's mount height."""
    return CameraSpec(intrinsics_from_hfov(width, height, hfov_deg), MOUNT_M,
                      min_depth_m, max_depth_m)


def point_seen_at(camera, pose, u, v, depth_m):
    """The world point of one reading at pixel ``(u, v)``, every other pixel empty."""
    k = camera.intrinsics
    depth = np.full((k.height, k.width), np.nan)
    depth[v, u] = depth_m
    points = backproject_depth(depth, camera, pose)
    assert points.shape == (1, 3)
    return points[0]


# -- where a pixel lands --------------------------------------------------------

def test_the_centre_pixel_of_a_level_camera_lands_straight_ahead_at_camera_height():
    """The simplest case: a wrong frame or a W/2 principal point moves this point."""
    camera = make_camera()
    point = point_seen_at(camera, AgentPose(**BASE), CX, CY, 3.0)
    np.testing.assert_allclose(point, [4.0, -2.0, 0.5 + MOUNT_M], atol=1e-12)


def test_a_pixel_right_of_centre_lands_on_the_agents_right_which_is_world_minus_y():
    """Optical x is right, FLU y is left: a missed sign mirrors the whole map."""
    camera = make_camera()
    point = point_seen_at(camera, AgentPose(**BASE), CX + 10, CY, 2.0)
    offset = 10.0 / camera.intrinsics.fx * 2.0
    assert point[1] < BASE["y"]
    np.testing.assert_allclose(point, [3.0, -2.0 - offset, 0.5 + MOUNT_M],
                               atol=1e-12)


def test_a_pixel_below_centre_lands_below_the_camera():
    """Optical y is down: the SJTU-style slip puts the floor above the camera."""
    camera = make_camera()
    point = point_seen_at(camera, AgentPose(**BASE), CX, CY + 8, 2.0)
    drop = 8.0 / camera.intrinsics.fy * 2.0
    assert point[2] < BASE["z"] + MOUNT_M
    np.testing.assert_allclose(point, [3.0, -2.0, 0.5 + MOUNT_M - drop],
                               atol=1e-12)


def test_a_camera_pitched_down_30_degrees_sees_its_centre_pixel_ahead_and_below():
    """REP-103 positive pitch looks down; a flipped sign maps the floor onto the ceiling."""
    camera = make_camera()
    pose = AgentPose(camera_pitch=math.radians(30.0), **BASE)
    point = point_seen_at(camera, pose, CX, CY, 3.0)
    expected = [1.0 + 3.0 * math.cos(math.radians(30.0)), -2.0,
                0.5 + MOUNT_M - 3.0 * math.sin(math.radians(30.0))]
    np.testing.assert_allclose(point, expected, atol=1e-12)


def test_turning_left_90_degrees_maps_forward_to_plus_y_and_right_to_plus_x():
    """Yaw is counter-clockwise from +x: facing north, the agent's right is east."""
    camera = make_camera()
    pose = AgentPose(yaw=math.pi / 2.0, **BASE)
    ahead = point_seen_at(camera, pose, CX, CY, 3.0)
    right = point_seen_at(camera, pose, CX + 10, CY, 2.0)
    np.testing.assert_allclose(ahead, [1.0, 1.0, 0.5 + MOUNT_M], atol=1e-12)
    np.testing.assert_allclose(
        right, [1.0 + 10.0 / camera.intrinsics.fx * 2.0, 0.0, 0.5 + MOUNT_M],
        atol=1e-12)


def test_the_camera_pitches_about_its_own_left_axis_after_the_agent_turns():
    """Rz(yaw) @ Ry(pitch), not the reverse: turned north and pitched, it still looks down."""
    camera = make_camera()
    pose = AgentPose(yaw=math.pi / 2.0, camera_pitch=math.radians(30.0),
                     **BASE)
    point = point_seen_at(camera, pose, CX, CY, 3.0)
    expected = [1.0, -2.0 + 3.0 * math.cos(math.radians(30.0)),
                0.5 + MOUNT_M - 3.0 * math.sin(math.radians(30.0))]
    np.testing.assert_allclose(point, expected, atol=1e-12)


def test_projecting_backprojected_depth_recovers_every_pixel_and_its_depth():
    """The two directions must be exact inverses, or a detection lands off its own pixel."""
    camera = make_camera(hfov_deg=79.0)
    rng = np.random.default_rng(7)
    depth = rng.uniform(0.5, 8.0, size=(HEIGHT, WIDTH))
    pose = AgentPose(x=3.0, y=-1.5, z=1.2, yaw=2.1, camera_pitch=0.35)
    pixels, z = project_to_image(backproject_depth(depth, camera, pose),
                                 camera, pose)
    rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH]
    np.testing.assert_allclose(pixels[:, 0], cols.ravel(), atol=1e-9)
    np.testing.assert_allclose(pixels[:, 1], rows.ravel(), atol=1e-9)
    np.testing.assert_allclose(z, depth.ravel(), atol=1e-9)


# -- which pixels become points -------------------------------------------------

def test_nan_inf_out_of_range_and_clipped_readings_are_dropped():
    """No reading, nothing in range, beyond the sensor, or clipped to a bound: never a surface."""
    camera = make_camera(width=10, height=1, min_depth_m=0.1, max_depth_m=10.0)
    depth = np.array([[np.nan, np.inf, -np.inf, 0.05, 10.5, 0.1, 10.0,
                       0.1 + 1e-6, 10.0 - 1e-6, 3.0]])
    points = backproject_depth(depth, camera, AgentPose(**BASE))
    pixels, z = project_to_image(points, camera, AgentPose(**BASE))
    np.testing.assert_allclose(pixels[:, 0], [7.0, 8.0, 9.0], atol=1e-9)
    np.testing.assert_allclose(z, [0.1 + 1e-6, 10.0 - 1e-6, 3.0], atol=1e-9)


def test_a_zero_reading_is_never_a_point_even_with_a_zero_minimum_depth():
    """A "no return" left as 0 would put a surface at the optical centre, where the agent stands."""
    camera = make_camera(width=3, height=1, min_depth_m=0.0, max_depth_m=np.inf)
    depth = np.array([[0.0, -2.0, 2.0]])
    points = backproject_depth(depth, camera, AgentPose(**BASE))
    pixels, _ = project_to_image(points, camera, AgentPose(**BASE))
    np.testing.assert_allclose(pixels[:, 0], [2.0], atol=1e-9)


def test_stride_keeps_every_nth_pixel_in_each_axis_starting_at_zero():
    """A mapper that subsamples must know exactly which pixels it got."""
    camera = make_camera(width=7, height=5)
    depth = np.full((5, 7), 2.0)
    points = backproject_depth(depth, camera, AgentPose(**BASE), stride=3)
    pixels, _ = project_to_image(points, camera, AgentPose(**BASE))
    np.testing.assert_allclose(
        pixels, [[0, 0], [3, 0], [6, 0], [0, 3], [3, 3], [6, 3]], atol=1e-9)


def test_an_image_with_no_valid_reading_backprojects_to_an_empty_point_array():
    """A blank frame is legal (facing open space) and must not break the caller's stacking."""
    camera = make_camera()
    points = backproject_depth(np.full((HEIGHT, WIDTH), np.inf), camera,
                               AgentPose(**BASE))
    assert points.shape == (0, 3)
    assert points.dtype == np.float64


def test_float32_depth_gives_float64_points():
    """Simulators hand over float32; the world coordinates must not inherit its precision."""
    camera = make_camera()
    depth = np.full((HEIGHT, WIDTH), 2.0, dtype=np.float32)
    points = backproject_depth(depth, camera, AgentPose(**BASE))
    assert points.dtype == np.float64
    assert points.shape == (HEIGHT * WIDTH, 3)


# -- intrinsics -----------------------------------------------------------------

def test_intrinsics_from_hfov_reproduces_habitats_640_by_480_79_degree_camera():
    """The HM3D ObjectNav camera: square pixels and a pixel-centre principal point."""
    k = intrinsics_from_hfov(640, 480, 79)
    assert isinstance(k, Intrinsics)
    assert (k.width, k.height) == (640, 480)
    assert k.fx == k.fy
    assert k.fx == pytest.approx(320.0 / math.tan(math.radians(39.5)), rel=1e-12)
    assert k.fx == pytest.approx(388.19, abs=0.01)
    assert (k.cx, k.cy) == (319.5, 239.5)


@pytest.mark.parametrize("width, height, hfov_deg", [
    (640, 480, 45.0), (64, 48, 90.0), (1920, 1080, 79.0), (504, 294, 100.0),
    (100, 37, 45.0), (320, 240, 1.0),
])
def test_square_pixels_are_exact_where_the_vertical_fov_round_trip_is_not(
        width, height, hfov_deg):
    """These sizes leave fov_to_intrinsics' fx and fy an ulp apart; ours must not be."""
    k = intrinsics_from_hfov(width, height, hfov_deg)
    assert k.fx == k.fy


def test_fov_to_intrinsics_itself_describes_square_pixels_so_snapping_fy_is_round_off():
    """Pins the convention of the helper ours duplicates: if it changed, setting fy = fx would hide it."""
    from sparx_agency.core.common.spatial_math import fov_to_intrinsics
    vfov = math.degrees(2.0 * math.atan(
        math.tan(math.radians(79.0) / 2.0) * 480 / 640))
    k = fov_to_intrinsics(640, 480, 79.0, vfov)
    assert k.fy == pytest.approx(k.fx, rel=1e-12)
    assert (k.cx, k.cy) == (319.5, 239.5)


def test_numpy_integer_sizes_are_accepted_because_adapters_read_them_off_arrays():
    """``depth.shape`` entries and config arrays are numpy scalars, not ints."""
    k = intrinsics_from_hfov(np.int64(64), np.int32(48), np.float64(79.0))
    assert (k.width, k.height) == (64, 48)
    assert type(k.width) is int and type(k.height) is int


@pytest.mark.parametrize("width, height, hfov_deg", [
    (0, 480, 79.0), (640, -1, 79.0), (640.0, 480, 79.0), (True, 480, 79.0),
    (640, 480, 0.0), (640, 480, 180.0), (640, 480, -30.0),
    (640, 480, float("nan")), (640, 480, "79"), (640, 480, 1e-320),
    (640, 480, 5e-324),
])
def test_intrinsics_from_hfov_refuses_a_size_or_field_of_view_with_no_pinhole(
        width, height, hfov_deg):
    """No focal length exists for these; a float size is a unit or shape bug upstream."""
    with pytest.raises(ObjNavError):
        intrinsics_from_hfov(width, height, hfov_deg)


# -- transforms -----------------------------------------------------------------

def test_the_optical_transform_is_a_proper_rotation_with_optical_z_along_the_view():
    """A ray caster renders through this block; a reflection would mirror its images."""
    camera = make_camera()
    pose = AgentPose(x=1.0, y=2.0, z=3.0, yaw=0.7, camera_pitch=-0.4)
    transform = world_T_camera_optical(pose, camera)
    rotation = transform[:3, :3]
    flu = camera_rotation_world(pose)
    assert transform.shape == (4, 4) and transform.dtype == np.float64
    np.testing.assert_allclose(rotation.T.dot(rotation), np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_array_equal(transform[3], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(rotation[:, 2], flu[:, 0], atol=1e-12)
    np.testing.assert_allclose(rotation[:, 0], -flu[:, 1], atol=1e-12)
    np.testing.assert_allclose(rotation[:, 1], -flu[:, 2], atol=1e-12)
    np.testing.assert_array_equal(transform[:3, 3],
                                  camera_position_world(pose, camera))


def test_the_camera_sits_at_mount_height_above_the_floor_under_the_agent():
    """``AgentPose.z`` is the floor, not the camera: a second storey must add, not replace."""
    position = camera_position_world(AgentPose(x=1.0, y=2.0, z=3.0),
                                     make_camera())
    assert position.shape == (3,) and position.dtype == np.float64
    np.testing.assert_allclose(position, [1.0, 2.0, 3.0 + MOUNT_M], atol=1e-12)


# -- projection -----------------------------------------------------------------

def test_a_point_behind_the_camera_or_in_its_plane_projects_with_non_positive_depth():
    """The caller filters on z; it needs a sign it can trust and no warning noise."""
    camera = make_camera()
    pose = AgentPose(x=0.0, y=0.0)
    points = np.array([[-1.0, 0.0, MOUNT_M], [0.0, 1.0, MOUNT_M]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pixels, z = project_to_image(points, camera, pose)
    assert pixels.shape == (2, 2)
    np.testing.assert_allclose(z, [-1.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("points", [
    np.zeros(3), np.zeros((2, 2)), np.array([[0.0, 0.0, np.nan]]),
    np.array([[np.inf, 0.0, 0.0]]), [["a", "b", "c"]],
], ids=["single-point", "two-columns", "nan", "inf", "strings"])
def test_project_to_image_refuses_points_that_are_not_finite_n_by_3(points):
    """A NaN point would project to a NaN pixel that indexes an image silently."""
    with pytest.raises(ObjNavError):
        project_to_image(points, make_camera(), AgentPose(**BASE))


# -- refusals -------------------------------------------------------------------

@pytest.mark.parametrize("depth", [
    np.full((WIDTH, HEIGHT), 1.0),
    np.full((HEIGHT, WIDTH), 1000, dtype=np.uint16),
    [[1.0] * WIDTH] * HEIGHT,
    np.full((HEIGHT, WIDTH, 1), 1.0),
], ids=["width-height-swapped", "integer-millimetres", "list", "trailing-axis"])
def test_a_depth_image_that_disagrees_with_the_camera_is_refused(depth):
    """Swapped axes or millimetre integers would back-project to a wrong but plausible cloud."""
    with pytest.raises(ObservationError):
        backproject_depth(depth, make_camera(), AgentPose(**BASE))


@pytest.mark.parametrize("stride", [0, -2, True, 1.5, "2"])
def test_a_stride_that_is_not_a_positive_integer_is_refused(stride):
    """Zero would divide the image into nothing; a float stride is a caller bug."""
    with pytest.raises(ObjNavError):
        backproject_depth(np.full((HEIGHT, WIDTH), 1.0), make_camera(),
                          AgentPose(**BASE), stride=stride)


def test_a_planar_pose_or_a_bare_tuple_is_a_type_error():
    """A Pose2D has no floor height or camera pitch, so it cannot place the camera."""
    camera = make_camera()
    with pytest.raises(TypeError):
        camera_rotation_world(Pose2D(0.0, 0.0, 0.0))
    with pytest.raises(TypeError):
        camera_position_world(AgentPose(**BASE), camera.intrinsics)
    with pytest.raises(TypeError):
        project_to_image(np.zeros((1, 3)), camera, (0.0, 0.0, 0.0))


# -- import weight --------------------------------------------------------------

def test_importing_the_module_pulls_in_neither_yaml_nor_any_heavy_dependency():
    """spatial_math imports yaml at module scope; camera geometry must never reach it."""
    # A FRESH interpreter: this process has long since imported whatever the
    # other tests pulled in.
    code = (
        "import sys, importlib\n"
        "importlib.import_module("
        "'sparx_agency.core.planning.objnav.camera_geometry')\n"
        "print(','.join(sorted({m.split('.')[0] for m in sys.modules}"
        " & set(%r))))\n" % (HEAVY,))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, cwd=str(REPO_ROOT))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "", (
        "importing camera_geometry pulled in %s" % out.stdout.strip())
