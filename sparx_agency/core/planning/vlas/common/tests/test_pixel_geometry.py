"""A pixel goal is a place. The two directions have to be each other's inverse."""
import math

import numpy as np
import pytest

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.planning.vlas.common.pixel_geometry import (
    bearing_to,
    body_to_pixel,
    body_to_world,
    patch_median_depth,
    pixel_to_body,
    world_to_body,
)

INTR = Intrinsics(width=600, height=600, fx=390.642735, fy=390.642735,
                  cx=300.0, cy=300.0)


class TestRoundTrips:
    @pytest.mark.parametrize("u,v,d", [(300, 300, 3.0), (100, 250, 2.0),
                                       (500, 400, 4.5), (0, 599, 1.2)])
    def test_pixel_to_body_and_back(self, u, v, d):
        back = body_to_pixel(*pixel_to_body(u, v, d, INTR), INTR)
        assert back == pytest.approx((u, v), abs=1e-6)

    @pytest.mark.parametrize("yaw_deg", [0.0, 35.0, -120.0, 179.0])
    def test_body_to_world_and_back(self, yaw_deg):
        pose = (2.0, -1.0, 1.2, math.radians(yaw_deg))
        body = pixel_to_body(420, 280, 3.5, INTR)
        assert world_to_body(body_to_world(*body, pose), pose) == pytest.approx(body, abs=1e-9)

    def test_the_goal_moves_across_the_image_as_the_aircraft_turns(self):
        # THE WHOLE POINT. Seen from a pose, re-projected from a later one, the
        # marker has to move -- a pixel redrawn at its original coordinate is
        # the bug this module exists to fix.
        seen_from = (0.0, 0.0, 1.2, 0.0)
        goal = body_to_world(*pixel_to_body(300, 300, 4.0, INTR), seen_from)
        straight = body_to_pixel(*world_to_body(goal, seen_from), INTR)
        turned = body_to_pixel(*world_to_body(goal, (0.0, 0.0, 1.2, math.radians(15.0))), INTR)
        assert straight[0] == pytest.approx(300.0, abs=1e-6)
        assert turned[0] > straight[0] + 50      # turning left pushes it right

    def test_flying_toward_the_goal_does_not_move_a_centred_one(self):
        seen_from = (0.0, 0.0, 1.2, 0.0)
        goal = body_to_world(*pixel_to_body(300, 300, 4.0, INTR), seen_from)
        closer = body_to_pixel(*world_to_body(goal, (2.0, 0.0, 1.2, 0.0)), INTR)
        assert closer == pytest.approx((300.0, 300.0), abs=1e-6)


class TestBehindTheCamera:
    def test_a_point_behind_the_aircraft_has_no_pixel(self):
        # Unguarded, the projection puts it confidently on the far side of the
        # image -- a marker pointing exactly the wrong way.
        assert body_to_pixel(-1.0, 0.2, 0.0, INTR) is None
        assert body_to_pixel(0.0, 0.0, 0.0, INTR) is None

    def test_a_point_just_in_front_still_projects(self):
        assert body_to_pixel(0.06, 0.0, 0.0, INTR) is not None


class TestPatchMedianDepth:
    def test_it_ignores_misses_and_the_sky(self):
        depth = np.full((100, 100), np.nan, np.float32)
        depth[45:55, 45:55] = 2.5
        assert patch_median_depth(depth, 50, 50, half=8) == pytest.approx(2.5)

    def test_a_patch_with_nothing_valid_returns_none(self):
        # Dropped, not guessed: an invented range puts the marker somewhere the
        # model never meant.
        assert patch_median_depth(np.full((50, 50), np.nan, np.float32), 25, 25) is None
        assert patch_median_depth(np.zeros((50, 50), np.float32), 25, 25) is None

    def test_it_survives_a_pixel_at_the_edge(self):
        depth = np.full((60, 60), 1.5, np.float32)
        assert patch_median_depth(depth, 0, 0) == pytest.approx(1.5)
        assert patch_median_depth(depth, 59, 59) == pytest.approx(1.5)

    def test_a_non_2d_array_is_an_error_not_a_guess(self):
        with pytest.raises(ValueError):
            patch_median_depth(np.zeros((10, 10, 1), np.float32), 5, 5)


class TestBearing:
    def test_left_is_positive(self):
        assert bearing_to(1.0, 1.0) == pytest.approx(math.pi / 4)
        assert bearing_to(1.0, -1.0) == pytest.approx(-math.pi / 4)
        assert bearing_to(1.0, 0.0) == 0.0


def test_navdp_still_gets_the_same_back_projection():
    """The shared helper replaced NavDP's inline copy; it must not have moved."""
    from sparx_agency.core.planning.vlas.navdp.geometry import pixel_to_pointgoal
    depth = np.full((600, 600), 3.0, np.float32)
    gx, gy, d, bz = pixel_to_pointgoal(200, 250, depth, INTR)
    fwd, left, up = pixel_to_body(200, 250, 3.0, INTR)
    assert (gx, gy, d, bz) == pytest.approx((fwd, left, 3.0, up))
