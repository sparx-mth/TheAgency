"""``NavigationCommand`` accepts every path shape the repo's planners produce and refuses every command the converter could misread.

A stop command that also carries a path is how an agent calls STOP a metre
short, a NaN waypoint turns every heading error into NaN, and ``bool("false")``
is True. None of those crash anything downstream -- they move SR and SPL -- so
each documented refusal is exercised here, with the error class the docstring
names, and each documented acceptance too (``Pose2D`` lists, a ``Path2D``, an
``(N, 3)`` array), because a command that refuses a legal path breaks every
policy that plans one.

Python 3.8 syntax, numpy.
"""
from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from sparx_agency.core.common.types import Path2D, Pose2D
from sparx_agency.core.planning.objnav.errors import CommandError
from sparx_agency.core.planning.objnav.tests.helpers import INF, NAN
from sparx_agency.core.planning.objnav.types.angles import MAX_ANGLE_RAD
from sparx_agency.core.planning.objnav.types.command import NavigationCommand

PATH = ((0.0, 0.0), (1.0, 0.5), (2.0, 0.0))


def test_follow_coerces_any_pair_sequence_to_float_tuples():
    """The same path must compare equal however it was built, so the converter keeps its cursor."""
    from_tuples = NavigationCommand.follow(PATH)
    from_lists = NavigationCommand.follow([[0, 0], [1, 0.5], [2, 0]])
    assert from_tuples.waypoints == PATH
    assert all(type(v) is float for p in from_lists.waypoints for v in p)
    assert from_tuples == from_lists
    assert not from_tuples.stop
    assert from_tuples.camera_pitch is None and from_tuples.final_yaw is None


def test_follow_accepts_pose2d_points_and_a_path2d():
    """The repo's planners return ``Pose2D`` lists and ``Path2D``; the heading is not a waypoint."""
    poses = [Pose2D(x, y, 1.0) for x, y in PATH]
    assert NavigationCommand.follow(poses).waypoints == PATH
    assert NavigationCommand.follow(Path2D(tuple(poses))).waypoints == PATH


def test_follow_takes_the_first_two_columns_of_an_n_by_3_array():
    """A ``(N, 3)`` path with a yaw column is common here; its third column is not a coordinate."""
    array = np.array([[x, y, 0.7] for x, y in PATH])
    command = NavigationCommand.follow(array)
    assert command.waypoints == PATH
    assert all(type(v) is float for p in command.waypoints for v in p)


def test_follow_needs_at_least_one_waypoint():
    """An empty path is a hold, and saying so keeps the policy's intent readable."""
    with pytest.raises(CommandError, match="hold"):
        NavigationCommand.follow([])


@pytest.mark.parametrize("points", [
    [3.0], [(0.0,)], [(0.0, NAN)], [(INF, 0.0)], [(True, 0.0)], [("1", "2")],
], ids=["scalar", "one-coordinate", "nan", "inf", "bool", "strings"])
def test_follow_refuses_a_malformed_or_non_finite_point(points):
    """A NaN waypoint turns every heading error into NaN and every comparison into False."""
    with pytest.raises(CommandError, match="waypoint 0"):
        NavigationCommand.follow(points)


def test_the_constructor_refuses_a_waypoint_that_is_not_a_pair():
    """Only ``follow`` strips a yaw column; the validating form takes pairs only."""
    with pytest.raises(CommandError, match="pair"):
        NavigationCommand(waypoints=((0.0, 0.0, 0.0),))


@pytest.mark.parametrize("extra", [
    dict(waypoints=PATH), dict(camera_pitch=0.0), dict(final_yaw=0.0),
], ids=["waypoints", "pitch", "heading"])
def test_a_stop_that_carries_anything_else_is_refused(extra):
    """'Stop, but first walk there' is how an agent calls STOP a metre short."""
    with pytest.raises(CommandError, match="carries nothing else"):
        NavigationCommand(stop=True, **extra)


@pytest.mark.parametrize("stop", ["false", "true", 1, 0, None])
def test_a_stop_flag_that_is_not_a_real_bool_is_refused(stop):
    """``bool("false")`` is True, so a string from a config would end the episode."""
    with pytest.raises(CommandError, match="stop must be a bool"):
        NavigationCommand(stop=stop)


@pytest.mark.parametrize("field", ["camera_pitch", "final_yaw"])
@pytest.mark.parametrize("value", [NAN, INF, True, "0.5"])
def test_a_non_finite_or_non_numeric_angle_is_refused(field, value):
    """A NaN heading never compares as reached, so the agent would face it forever."""
    with pytest.raises(CommandError, match=field):
        NavigationCommand.hold(**{field: value})


def test_hold_without_arguments_is_the_empty_command():
    """A policy may have nothing to ask; the converter then reports status empty."""
    command = NavigationCommand.hold()
    assert command == NavigationCommand()
    assert (command.waypoints, command.stop, command.camera_pitch,
            command.final_yaw) == ((), False, None, None)


def test_hold_carries_a_pitch_a_heading_and_its_info():
    """Re-aiming the camera in place is a hold, not a one-point path."""
    command = NavigationCommand.hold(camera_pitch=0.5, final_yaw=-1.0,
                                     info={"why": "look"})
    assert (command.waypoints, command.camera_pitch, command.final_yaw) == (
        (), 0.5, -1.0)
    assert command.info == {"why": "look"}


def test_stop_here_ends_the_episode_and_copies_its_info():
    """A policy may reuse its info dict next step; the logged command must not change with it."""
    info = {"reason": "target seen"}
    command = NavigationCommand.stop_here(info)
    info["reason"] = "changed"
    assert command.stop and command.waypoints == ()
    assert command.info == {"reason": "target seen"}


# -- stop_on_arrival ------------------------------------------------------------

def test_follow_and_hold_carry_stop_on_arrival_and_default_it_to_false():
    """A policy that faces before stopping needs the STOP on the very step the facing completes."""
    assert NavigationCommand.follow(PATH).stop_on_arrival is False
    assert NavigationCommand.follow(PATH, stop_on_arrival=True).stop_on_arrival
    assert NavigationCommand.hold(final_yaw=1.0,
                                  stop_on_arrival=True).stop_on_arrival


@pytest.mark.parametrize("build", [
    lambda: NavigationCommand(stop=True, stop_on_arrival=True),
    lambda: NavigationCommand(stop_on_arrival=True),
    lambda: NavigationCommand.hold(stop_on_arrival=True),
], ids=["with-a-stop", "empty-constructor", "empty-hold"])
def test_stop_on_arrival_with_a_stop_or_on_an_otherwise_empty_command_is_refused(build):
    """With a stop it is redundant; alone it is stop_here() by another name -- say which is meant."""
    with pytest.raises(CommandError, match="stop_on_arrival"):
        build()


@pytest.mark.parametrize("flag", ["true", 1, None])
def test_a_stop_on_arrival_flag_that_is_not_a_real_bool_is_refused(flag):
    """``bool("false")`` is True: a config string would stop an episode the policy meant to go on."""
    with pytest.raises(CommandError, match="stop_on_arrival must be a bool"):
        NavigationCommand.hold(final_yaw=0.0, stop_on_arrival=flag)


# -- angles too large to wrap ------------------------------------------------------

@pytest.mark.parametrize("field", ["camera_pitch", "final_yaw"])
@pytest.mark.parametrize("value", [1e17, -2.0 * MAX_ANGLE_RAD])
def test_an_angle_too_large_to_wrap_is_refused(field, value):
    """normalize_angle subtracts one turn per loop: at 1e17 rad it never ends, and the sweep hangs."""
    with pytest.raises(CommandError, match="wrap it"):
        NavigationCommand.hold(**{field: value})


def test_an_angle_at_the_bound_is_accepted():
    """The bound refuses garbage, not a heading accumulated over a long episode."""
    assert NavigationCommand.hold(final_yaw=-MAX_ANGLE_RAD).final_yaw == \
        -MAX_ANGLE_RAD


# -- info ------------------------------------------------------------------------

@pytest.mark.parametrize("info", [None, [("room", "kitchen")], "kitchen"],
                         ids=["none", "pairs", "string"])
def test_info_that_is_not_a_mapping_is_refused_by_the_constructor(info):
    """None used to pass here and crash inside the agent after the step was counted, never naming info."""
    with pytest.raises(CommandError, match="info"):
        NavigationCommand(waypoints=PATH, info=info)


@pytest.mark.parametrize("info", [[("room", "kitchen")], "kitchen"],
                         ids=["pairs", "string"])
def test_follow_hold_and_stop_here_refuse_info_that_is_not_a_mapping(info):
    """Pairs used to become a dict silently, and a string raised a cryptic ValueError."""
    for build in (lambda: NavigationCommand.follow(PATH, info=info),
                  lambda: NavigationCommand.hold(info=info),
                  lambda: NavigationCommand.stop_here(info)):
        with pytest.raises(CommandError, match="info"):
            build()


def test_info_is_stored_as_a_dict_copy_of_any_mapping():
    """A read-only mapping is fine; the command keeps its own dict, so the policy may reuse its own."""
    source = {"room": "kitchen"}
    command = NavigationCommand.follow(PATH, info=MappingProxyType(source))
    source["room"] = "hall"
    assert type(command.info) is dict and command.info == {"room": "kitchen"}
