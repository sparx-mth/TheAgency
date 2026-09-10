"""The converter's tuning and result types accept what their docstrings promise and refuse what they forbid.

A zero goal tolerance is never met, so the agent would circle its last waypoint
forever; a result that carried STOP on anything but a stop command would end
episodes a metre short; a rollout whose poses are off by one shifts every pose
against the action that produced it. So each documented refusal of
``ActionConverterParams``, ``ConversionResult`` and ``Rollout`` is exercised
here, with the error class the docstring names.

Python 3.8 syntax; numpy arrives only through the types.
"""
from __future__ import annotations

import pytest

from sparx_agency.core.planning.objnav.action_converter.params import (
    ActionConverterParams,
)
from sparx_agency.core.planning.objnav.action_converter.types import (
    ACTING_STATUSES,
    IDLE_STATUSES,
    ROLLOUT_ENDINGS,
    ROLLOUT_LIMIT,
    STATUS_ARRIVED,
    STATUS_EMPTY,
    STATUS_FORWARD,
    STATUS_STOP,
    STATUS_TURN,
    STATUSES,
    ConversionResult,
    Rollout,
)
from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.core.planning.objnav.tests.helpers import INF, NAN
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.pose import AgentPose


# -- ActionConverterParams -----------------------------------------------------------------

def test_converter_params_default_to_two_steps_of_lookahead_and_one_of_tolerance():
    """The defaults are the documented ones, which the converter's tests assume."""
    params = ActionConverterParams()
    assert (params.lookahead_m, params.goal_tolerance_m,
            params.blocked_epsilon_m) == (0.5, 0.25, 0.01)


@pytest.mark.parametrize("field", ["lookahead_m", "goal_tolerance_m",
                                   "blocked_epsilon_m"])
@pytest.mark.parametrize("value", [0.0, -0.5, NAN, INF, True])
def test_a_non_positive_or_non_finite_converter_parameter_is_refused(field, value):
    """A zero tolerance is never met, so the agent would circle its last waypoint forever."""
    with pytest.raises(ObjNavError, match=field):
        ActionConverterParams(**{field: value})


# -- ConversionResult and Rollout --------------------------------------------------------------

def test_acting_and_idle_statuses_partition_the_statuses():
    """A status in neither set would carry an action or not by accident."""
    assert set(ACTING_STATUSES).isdisjoint(IDLE_STATUSES)
    assert set(ACTING_STATUSES).union(IDLE_STATUSES) == set(STATUSES)
    assert set(ROLLOUT_ENDINGS) == set(IDLE_STATUSES).union(
        {STATUS_STOP, ROLLOUT_LIMIT})


def test_an_acting_result_carries_its_action_and_an_idle_one_none():
    """Idle means no action, so the agent -- not the converter -- decides the step."""
    forward = ConversionResult(DiscreteAction.MOVE_FORWARD, STATUS_FORWARD,
                               segment_index=1, target_xy=(1.0, 0.0))
    assert not forward.idle and forward.target_xy == (1.0, 0.0)
    arrived = ConversionResult(None, STATUS_ARRIVED)
    assert arrived.idle and arrived.segment_index == 0
    assert arrived.target_xy is None and not arrived.forward_blocked
    assert ConversionResult(DiscreteAction.STOP, STATUS_STOP).action == (
        DiscreteAction.STOP)


@pytest.mark.parametrize("action,status,message", [
    (DiscreteAction.MOVE_FORWARD, "walking", "status must be one of"),
    (DiscreteAction.TURN_LEFT, STATUS_ARRIVED, "exactly when"),
    (DiscreteAction.TURN_LEFT, STATUS_EMPTY, "exactly when"),
    (None, STATUS_TURN, "exactly when"),
    (DiscreteAction.STOP, STATUS_TURN, "STOP only for a stop"),
    (DiscreteAction.MOVE_FORWARD, STATUS_STOP, "STOP only for a stop"),
    (1, STATUS_FORWARD, "must be a DiscreteAction"),
], ids=["unknown-status", "action-when-arrived", "action-when-empty",
        "no-action-when-turning", "stop-while-turning", "stop-status-no-stop",
        "bare-int"])
def test_a_result_that_breaks_its_invariants_is_refused(action, status, message):
    """A converter that defaulted to STOP would end episodes a metre short; the type forbids it."""
    with pytest.raises(ObjNavError, match=message):
        ConversionResult(action, status)


def test_a_rollout_holds_the_start_pose_plus_one_per_action():
    """``final_pose`` is where perfect execution leaves the agent, even with no action."""
    poses = [AgentPose(0.0, 0.0), AgentPose(0.25, 0.0)]
    rollout = Rollout([DiscreteAction.MOVE_FORWARD], poses, STATUS_ARRIVED)
    assert rollout.actions == (DiscreteAction.MOVE_FORWARD,)
    assert isinstance(rollout.poses, tuple)
    assert rollout.final_pose == AgentPose(0.25, 0.0)
    empty = Rollout((), (AgentPose(1.0, 2.0),), STATUS_EMPTY)
    assert empty.final_pose == AgentPose(1.0, 2.0)


@pytest.mark.parametrize("n_poses", [0, 1, 3])
def test_a_rollout_with_the_wrong_number_of_poses_is_refused(n_poses):
    """An off-by-one here shifts every pose against the action that produced it."""
    poses = [AgentPose(0.0, 0.0)] * n_poses
    with pytest.raises(ObjNavError, match="start pose plus one pose per action"):
        Rollout((DiscreteAction.MOVE_FORWARD,), poses, ROLLOUT_LIMIT)


@pytest.mark.parametrize("status", [STATUS_FORWARD, STATUS_TURN, "done"])
def test_a_rollout_ending_that_is_not_an_ending_is_refused(status):
    """A rollout ends idle, on STOP, or at its budget -- never mid-turn."""
    with pytest.raises(ObjNavError, match="Rollout.status"):
        Rollout((), (AgentPose(0.0, 0.0),), status)
