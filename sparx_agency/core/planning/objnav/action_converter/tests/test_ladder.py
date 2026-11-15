"""The decision ladder refuses what only the converter may answer, and decides from its inputs alone.

The ladder is stateless and public, so it is checked on its own as well as
through the converter: a stop command handed to it would silently idle instead
of stopping, and a pitch the benchmark cannot execute must be refused by the
ladder exactly as by the converter. The reach floor it arrives within is pinned
here against the geometry it comes from.
"""
from __future__ import annotations

import math

import pytest

from sparx_agency.core.planning.objnav.action_converter.ladder import (
    check_executable,
    choose_action,
    reach_floor,
)
from sparx_agency.core.planning.objnav.action_converter.params import (
    ActionConverterParams,
)
from sparx_agency.core.planning.objnav.action_converter.path_progress import (
    PathProgress,
    PathSample,
)
from sparx_agency.core.planning.objnav.action_converter.tests.helpers import (
    FORWARD,
    HABITAT,
    NO_TILT,
    ORIGIN,
)
from sparx_agency.core.planning.objnav.action_converter.transition import (
    apply_action,
)
from sparx_agency.core.planning.objnav.action_converter.types import (
    STATUS_FORWARD,
)
from sparx_agency.core.planning.objnav.errors import CommandError, ObjNavError
from sparx_agency.core.planning.objnav.types.actions import DiscreteActionSpec
from sparx_agency.core.planning.objnav.types.command import NavigationCommand

PARAMS = ActionConverterParams()


def test_the_ladder_refuses_a_stop_command():
    """STOP is answered before the ladder; a ladder that saw one would idle instead of stopping."""
    with pytest.raises(ObjNavError, match="STOP"):
        choose_action(ORIGIN, NavigationCommand.stop_here(), None, False,
                      HABITAT, PARAMS)


def test_a_pitch_the_benchmark_cannot_execute_is_refused_by_both_entries():
    """The converter checks before it changes state, the ladder on every call; both refuse alike."""
    command = NavigationCommand.hold(camera_pitch=0.3)
    with pytest.raises(CommandError):
        check_executable(command, NO_TILT)
    with pytest.raises(CommandError):
        choose_action(ORIGIN, command, None, False, NO_TILT, PARAMS)
    check_executable(command, HABITAT)


def test_the_ladder_decides_from_its_inputs_alone_and_only_reports_blocked():
    """The same sample and command give the same action; the blocked flag rides along untouched."""
    progress = PathProgress()
    progress.adopt(((0.0, 0.0), (3.0, 0.0)))
    sample = progress.advance(ORIGIN, 0.5, 0.125)
    command = NavigationCommand.follow([(0.0, 0.0), (3.0, 0.0)])
    clear = choose_action(ORIGIN, command, sample, False, HABITAT, PARAMS)
    blocked = choose_action(ORIGIN, command, sample, True, HABITAT, PARAMS)
    assert clear.action == blocked.action == FORWARD
    assert (clear.forward_blocked, blocked.forward_blocked) == (False, True)
    assert clear.target_xy == sample.target_xy


def test_an_aim_a_step_cannot_close_in_on_still_draws_a_forward_step_when_the_path_goes_on():
    """Only the goal is arrived at; idling short of it would strand the agent mid-path."""
    bearing = math.radians(14.0)
    aim = (0.126 * math.cos(bearing), 0.126 * math.sin(bearing))
    sample = PathSample(segment_index=0, cross_track_m=0.0,
                        distance_to_goal_m=3.0, remaining_path_m=3.0,
                        target_xy=aim, aims_at_goal=False,
                        heading_error_rad=bearing)
    after = apply_action(ORIGIN, FORWARD, HABITAT)
    assert math.hypot(aim[0] - after.x, aim[1] - after.y) > 0.126
    command = NavigationCommand.follow([(0.0, 0.0), aim, (3.0, 0.0)])
    result = choose_action(ORIGIN, command, sample, False, HABITAT, PARAMS)
    assert (result.action, result.status) == (FORWARD, STATUS_FORWARD)


def test_the_reach_floor_is_where_a_step_aligned_within_half_a_turn_stops_closing_in():
    """Beyond it every aligned step closes in; inside it one half a turn off does not."""
    floor = reach_floor(HABITAT)
    assert floor == pytest.approx(0.25 / (2.0 * math.cos(math.radians(15.0))),
                                  rel=1e-7)
    assert floor == pytest.approx(0.1294, abs=1e-4)
    after = apply_action(ORIGIN, FORWARD, HABITAT)
    for scale, closes_in in ((1.001, True), (0.999, False)):
        d = scale * floor
        goal = (d * math.cos(math.radians(15.0)), d * math.sin(math.radians(15.0)))
        assert (math.hypot(goal[0] - after.x, goal[1] - after.y) < d) is closes_in


def test_a_180_degree_turn_keeps_the_reach_floor_finite():
    """Its dead band is a quarter turn, whose cosine is 0; an infinite floor would arrive from anywhere."""
    spec = DiscreteActionSpec(turn_angle_deg=180.0)
    assert reach_floor(spec) == pytest.approx(
        0.25 / (2.0 * math.cos(math.radians(89.0))))
