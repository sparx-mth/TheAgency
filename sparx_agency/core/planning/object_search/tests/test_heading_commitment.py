"""HeadingCommitment: stair steering decides once per standing position and never turns back to a ruled-out heading."""
import math

import pytest

from sparx_agency.core.planning.object_search.multifloor.heading_commitment import HeadingCommitment
from sparx_agency.core.planning.objnav.types.pose import AgentPose

TURN = math.radians(30.0)


def option(yaw_deg, distance_m, turn):
    yaw = math.radians(yaw_deg)
    return (-distance_m - 0.025 * abs(turn), -abs(turn), yaw, (math.cos(yaw), math.sin(yaw), 0.0))


def flip_options(facing_deg):
    """00802/000008: facing -103 only -133 is safe, facing -133 the shorter -103 is safe too."""
    if round(facing_deg) == -103:
        return [option(-133, 1.483, -1), option(-43, 1.524, 2)]
    return [option(-133, 1.483, 0), option(-103, 1.300, 1), option(-73, 1.341, 2)]


def turn_to(pose, chosen):
    """The converter's one action: a 30-degree turn toward the chosen heading, or a step along it."""
    error = math.atan2(math.sin(chosen[2] - pose.yaw), math.cos(chosen[2] - pose.yaw))
    if abs(error) < 0.5 * TURN:
        return None
    return AgentPose(pose.x, pose.y, pose.z, pose.yaw + math.copysign(TURN, error))


def test_without_the_commitment_the_view_flip_turns_the_agent_back_and_forth_forever():
    pose = AgentPose(12.29, 1.39, 0.87, math.radians(-103))
    yaws = []
    for _ in range(8):
        best = max(flip_options(math.degrees(pose.yaw)), key=lambda o: o[:2])
        pose = turn_to(pose, best)
        assert pose is not None
        yaws.append(round(math.degrees(pose.yaw)))
    assert yaws == [-133, -103] * 4


def test_the_heading_faced_unsafe_is_ruled_out_and_the_safe_one_is_walked():
    commitment = HeadingCommitment(TURN)
    choose = commitment.chooser((12.25, -0.25))
    pose = AgentPose(12.29, 1.39, 0.87, math.radians(-103))
    chosen = choose(pose, flip_options(-103))
    assert round(math.degrees(chosen[2])) == -133
    pose = turn_to(pose, chosen)
    chosen = choose(pose, flip_options(-133))          # -103 now looks shorter, but it was faced and unsafe
    assert round(math.degrees(chosen[2])) == -133 and turn_to(pose, chosen) is None
    assert commitment.diagnostics() == {"held_deg": -133, "ruled_out": 1}


def test_a_held_heading_is_turned_to_whatever_the_views_on_the_way_say():
    commitment = HeadingCommitment(TURN)
    choose = commitment.chooser((0.0, -5.0))
    pose = AgentPose(0.0, 0.0, 0.0, math.radians(90))
    first = choose(pose, [option(-90, 4.0, -6), option(60, 6.0, -1)])
    assert round(math.degrees(first[2])) == -90
    pose = AgentPose(0.0, 0.0, 0.0, math.radians(60))
    assert choose(pose, [option(30, 3.0, -1)]) is first         # an intermediate view does not re-decide
    assert choose(pose, []) is first                            # nor does one with nothing safe in it
    pose = AgentPose(0.0, 0.0, 0.0, math.radians(-90))
    assert choose(pose, []) is None                             # faced and unsafe: ruled out, nothing left
    assert [round(math.degrees(r)) for r in commitment.ruled_out] == [90, 60, -90]   # each faced while unsafe


def test_moving_or_a_new_route_end_decides_afresh():
    commitment = HeadingCommitment(TURN)
    choose = commitment.chooser((0.0, -5.0))
    pose = AgentPose(0.0, 0.0, 0.0, math.radians(-90))
    assert choose(pose, []) is None
    assert choose(pose, [option(-90, 3.0, 0)]) is None          # ruled out here
    moved = AgentPose(0.0, -0.25, 0.0, math.radians(-90))
    assert choose(moved, [option(-90, 3.0, 0)]) is not None     # a new standing position
    commitment.chooser((4.0, -5.0))
    assert commitment.held is None and commitment.ruled_out == []


def test_headings_match_by_angle_across_the_wrap_and_off_the_lattice():
    commitment = HeadingCommitment(TURN)
    assert commitment.same(math.radians(179.0), math.radians(-179.0))
    assert commitment.same(math.radians(15.0 - 1e-9), 0.0) and not commitment.same(math.radians(16.0), 0.0)
    with pytest.raises(ValueError):
        HeadingCommitment(0.0)
