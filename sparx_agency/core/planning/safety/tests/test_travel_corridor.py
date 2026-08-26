"""The brake protects where the aircraft is GOING, not where its nose points."""
import math

import numpy as np
import pytest

from sparx_agency.core.planning.safety.depth_proximity_brake import (
    DepthProximityBrake,
    DepthProximityBrakeConfig,
)

CFG = DepthProximityBrakeConfig(fx=390.642735, fy=390.642735, cx=300.0, cy=300.0,
                                corridor_halfwidth_m=0.35, corridor_halfheight_m=0.35,
                                hard_block_d_m=0.70, margin_m=0.45,
                                nose_offset_m=0.10, min_valid_m=0.15, stride=4)


def doorway(x0=0.55, half_door=0.465, yaw_deg=0.0, room=4.0, size=600):
    """A wall at world ``x0`` with a ``2*half_door`` opening, seen from a yawed camera.

    The hospital's doorways are 0.93 m clear and the corridor is 0.70 m wide, so
    the opening is passable with 0.115 m to spare -- if the corridor is pointed
    at it.
    """
    u = np.arange(size, dtype=np.float32)[None, :].repeat(size, 0)
    psi = math.radians(yaw_deg)
    right = np.arctan((u - CFG.cx) / CFG.fx)      # ray angle right of the optical axis
    world_right = right - psi                      # ...relative to the door normal
    hit = x0 * np.tan(world_right)                 # where the ray crosses the wall
    return np.where(np.abs(hit) < half_door, room,
                    x0 / np.maximum(0.05, np.cos(world_right))).astype(np.float32)


class TestTheDoorwayItUsedToRefuse:
    """A route through the middle of an opening, flown with the nose off-axis."""

    @pytest.mark.parametrize("yaw_deg", [15.0, 25.0, 30.0])
    def test_the_nose_corridor_refuses_a_clear_path(self, yaw_deg):
        # The failure as measured in the hospital: the depth corridor swings onto
        # the jamb because the NOSE is off, and the aircraft stops for a doorway
        # its own route goes straight through.
        brake = DepthProximityBrake(CFG)
        v_nose, d_nose = brake.allowed_forward_speed(doorway(yaw_deg=yaw_deg))
        assert d_nose is not None and d_nose < 1.0
        assert v_nose < 0.4

    @pytest.mark.parametrize("yaw_deg", [15.0, 20.0, 25.0])
    def test_the_travel_corridor_allows_it(self, yaw_deg):
        brake = DepthProximityBrake(CFG)
        # Travelling straight through the door is -yaw in the body frame.
        v, d, certified = brake.allowed_speed_along(doorway(yaw_deg=yaw_deg),
                                                    math.radians(-yaw_deg))
        assert certified
        assert d > 2.0          # it sees through the opening into the room
        assert v > 1.0


class TestItStillStopsForThingsInTheWay:
    def test_a_wall_across_the_travel_direction_is_a_hard_block(self):
        brake = DepthProximityBrake(CFG)
        wall = np.full((600, 600), 0.5, np.float32)
        for deg in (-20.0, 0.0, 20.0):
            v, d, certified = brake.allowed_speed_along(wall, math.radians(deg))
            assert certified and v == 0.0 and d is not None

    def test_travelling_INTO_the_jamb_is_refused_even_with_the_door_in_view(self):
        # The mirror image of the bug, and the reason this is a fix and not just
        # a loosening: nose on the opening, travel at the wall beside it. A
        # corridor that followed the nose calls this clear at 4 m.
        brake = DepthProximityBrake(CFG)
        frame = doorway(x0=1.5, yaw_deg=0.0)
        clear_ahead, _ = brake.allowed_forward_speed(frame)
        assert clear_ahead > 1.0                      # the nose sees through the door
        v, d, certified = brake.allowed_speed_along(frame, math.radians(-25.0))
        assert certified
        assert d is not None and d < 2.0              # ...the travel ray does not

    def test_zero_bearing_is_exactly_the_old_behaviour(self):
        brake = DepthProximityBrake(CFG)
        frame = doorway(yaw_deg=12.0)
        old_v, old_d = brake.allowed_forward_speed(frame)
        new_v, new_d, _ = brake.allowed_speed_along(frame, 0.0)
        assert new_v == pytest.approx(old_v)
        assert new_d == pytest.approx(old_d)


class TestUncertifiedBearings:
    def test_a_bearing_outside_the_field_of_view_is_not_certified(self):
        # 37.5 deg of half-FOV less a 10 deg guard leaves +-27.5 deg certified,
        # which covers every travel bearing this follower actually produces:
        # measured over five hospital runs, the median was 1-7 deg off the nose
        # and the follower rotates rather than crab past `stop_turn_rad`.
        brake = DepthProximityBrake(CFG)
        assert math.degrees(brake.horizontal_half_fov()) == pytest.approx(37.5, abs=0.5)
        assert brake.sees_bearing(math.radians(25.0))
        assert not brake.sees_bearing(math.radians(30.0))
        _, _, certified = brake.allowed_speed_along(doorway(), math.radians(35.0))
        assert not certified

    def test_an_uncertified_bearing_answers_for_the_nose_instead_of_clear(self):
        # A corridor the camera cannot observe holds no returns, and "no returns"
        # means "clear" -- which is the one answer that must never come back for
        # a direction nobody has looked at.
        brake = DepthProximityBrake(CFG)
        wall = np.full((600, 600), 0.4, np.float32)
        v, d, certified = brake.allowed_speed_along(wall, math.radians(60.0))
        assert not certified
        assert v == 0.0 and d is not None
