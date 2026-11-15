"""The fake serves the episodes it was given, publicly, and refuses at construction any it could not score honestly.

A goal position in the episode would let an agent score without searching, and
an episode the fake cannot score -- a start in the wall band, an object the
building lacks, a goal walled off -- fails quietly at the step budget, where it
reads as a method's miss. So what reset hands over, and every episode refused
instead, is pinned here.
"""
from __future__ import annotations

import pytest

from sparx_agency.core.planning.objnav.errors import ObjNavError
from sparx_agency.core.planning.objnav.types.actions import DiscreteActionSpec
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.env import (
    FakeEpisodeSpec,
    FakeObjNavEnv,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.world import GridWorld
from sparx_agency.tasks.planning.objnav_benchmark.tests.fake_rooms import (
    INSIDE,
    OUTSIDE,
    ROOM,
    SEALED,
    pose_at,
    room_env,
    small_camera,
)


def test_reset_hands_over_the_public_episode_and_a_step_zero_frame_with_nothing_privileged():
    """A goal position in the episode would let an agent score without searching."""
    env = room_env(max_steps=40, benchmark="fake", split="val")
    episode, observation = env.reset("out")
    assert episode.metadata == {}
    assert (episode.episode_id, episode.scene_id, episode.benchmark, episode.split) == (
        "out", "room", "fake", "val")
    assert (episode.target_category, episode.max_steps) == ("chair", 40)
    assert episode.camera == env.camera and episode.action_spec == DiscreteActionSpec()
    assert observation.step == 0 and observation.target_category == "chair"
    assert observation.pose == pose_at(env.privileged_world(), OUTSIDE)
    assert not env.episode_over


def test_an_unknown_episode_id_is_a_key_error():
    """The ObjNavEnv contract names KeyError; the runner relies on it."""
    with pytest.raises(KeyError, match="serves no episode 'nope'"):
        room_env().reset("nope")


@pytest.mark.parametrize("starts,options,message", [
    ((("a", OUTSIDE, 0.0), ("a", INSIDE, 0.0)), {}, "appears twice"),
    ((("a", (4, 1), 0.0),), {}, "not navigable"),
    ((("a", (4, 3), 0.0),), {"rows": SEALED}, "cannot reach any 'chair'"),
    ((("a", OUTSIDE, 0.0),), {"max_steps": 0}, "max_steps"),
    ((("a", OUTSIDE, 0.0),), {"camera": small_camera(max_depth_m=float("inf"))}, "finite"),
    ((), {}, "at least one episode"),
], ids=["duplicate_id", "start_in_wall_band", "goal_unreachable", "no_budget",
        "infinite_depth", "no_episodes"])
def test_an_environment_that_could_not_run_its_episodes_is_refused_at_construction(
        starts, options, message):
    """Found at construction it costs nothing; found at episode 300 it costs the night."""
    with pytest.raises(ObjNavError, match=message):
        room_env(starts=starts, **options)


def test_a_missing_category_or_a_start_off_the_floor_is_refused():
    """An episode for an object the building lacks can only fail, and quietly."""
    world = GridWorld.from_ascii(ROOM, {"c": "chair"})
    bed = FakeEpisodeSpec("a", "room", "bed", pose_at(world, OUTSIDE))
    with pytest.raises(ObjNavError, match="has only 'chair'"):
        FakeObjNavEnv(world, [bed], camera=small_camera())
    x, y = world.cell_center(*OUTSIDE)
    raised = FakeEpisodeSpec("a", "room", "chair", AgentPose(x, y, 1.0))
    with pytest.raises(ObjNavError, match="one floor"):
        FakeObjNavEnv(world, [raised], camera=small_camera())
