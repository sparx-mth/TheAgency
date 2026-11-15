"""Progress belongs to one path and only runs forward; the aim sits a lookahead past it, never on the agent's own foot.

``PathProgress`` is the converter's memory of where the agent is on its path.
Each test pins one of its rules directly: the same path keeps its progress and
any other starts over, advancing without a path is refused, the aim is
exactly the goal once the lookahead reaches past it -- the one way the ladder
tells that it is aiming at the goal -- and an aim that a fold puts at the
agent's feet is moved on along the path.
"""
from __future__ import annotations

import pytest

from sparx_agency.core.planning.objnav.action_converter.path_progress import (
    PathProgress,
)
from sparx_agency.core.planning.objnav.action_converter.tests.helpers import (
    HAIRPIN,
)
from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.core.planning.objnav.types.pose import AgentPose

#: Half of Habitat's forward step: the nearest the aim may sit, short of the goal.
HALF_STEP = 0.125


def test_the_same_path_keeps_its_progress_and_any_other_starts_over():
    """Re-sent every step, the same route must not lose progress; a new one must."""
    progress = PathProgress()
    progress.adopt(HAIRPIN)
    assert progress.advance(AgentPose(2.0, 1.0), 0.5,
                            HALF_STEP).segment_index == 2
    progress.adopt(tuple(HAIRPIN))
    assert progress.advance(AgentPose(2.0, 0.4), 0.5,
                            HALF_STEP).segment_index == 2
    progress.adopt(HAIRPIN + ((-1.0, 1.0),))
    assert progress.advance(AgentPose(2.0, 0.4), 0.5,
                            HALF_STEP).segment_index == 0


def test_advancing_without_a_path_is_refused():
    """There is nothing to project onto; a silent sample would aim at nowhere."""
    progress = PathProgress()
    assert progress.path == ()
    with pytest.raises(ObjNavError):
        progress.advance(AgentPose(0.0, 0.0), 0.5, HALF_STEP)


def test_the_aim_sits_the_lookahead_past_the_progress_and_ends_exactly_at_the_goal():
    """The ladder knows it aims at the goal only if the aim is exactly the last waypoint."""
    progress = PathProgress()
    progress.adopt(((0.0, 0.0), (1.0, 0.0)))
    near = progress.advance(AgentPose(0.0, 0.0), 0.25, HALF_STEP)
    assert near.target_xy == pytest.approx((0.25, 0.0))
    assert not near.aims_at_goal
    far = progress.advance(AgentPose(0.5, 0.0), 0.5, HALF_STEP)
    assert far.target_xy == (1.0, 0.0) and far.aims_at_goal
    assert far.remaining_path_m == pytest.approx(0.5)


def test_an_aim_on_the_agents_own_foot_is_moved_on_to_half_a_step_away():
    """At an exact fold the lookahead lands where the agent stands, and the heading to it is noise."""
    progress = PathProgress()
    progress.adopt(((0.0, 0.0), (0.0, 2.0), (0.0, -1.0)))
    progress.advance(AgentPose(0.0, 1.5), 0.5, HALF_STEP)
    sample = progress.advance(AgentPose(0.0, 1.75), 0.5, HALF_STEP)
    assert sample.target_xy == pytest.approx((0.0, 1.625))
    assert not sample.aims_at_goal


def test_the_goal_is_never_moved_however_near_the_agent_is():
    """Near the goal the ladder arrives; an aim moved past the end would mean nothing."""
    progress = PathProgress()
    progress.adopt(((0.0, 0.0), (1.0, 0.0)))
    sample = progress.advance(AgentPose(0.95, 0.0), 0.5, HALF_STEP)
    assert sample.target_xy == (1.0, 0.0) and sample.aims_at_goal


@pytest.mark.parametrize("min_aim_m", [0.0, -0.1, float("nan")])
def test_a_minimum_aim_distance_that_is_not_positive_is_refused(min_aim_m):
    """Zero would switch the fold guard off without a word."""
    progress = PathProgress()
    progress.adopt(((0.0, 0.0), (1.0, 0.0)))
    with pytest.raises(ObjNavError):
        progress.advance(AgentPose(0.0, 0.0), 0.5, min_aim_m)
