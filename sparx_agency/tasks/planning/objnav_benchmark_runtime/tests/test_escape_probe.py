"""The escape probe: boxed in on the inflated map, the fallback steps out instead of turning in place for ever.

Collierville/000036 (``6bc7ace42dcf``, 2026-10-13): the agent spawned in the
strip behind a sofa, the depth camera ringed it with occupied cells, no
inflated route left the pocket, and the fallback's idle turn ran 488 actions.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams
from sparx_agency.core.planning.object_search.exploration.escape_probe import (
    EscapeProbe, EscapeSettings, ray_profile)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_exploration_fallback import no_rooms
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import (
    IN_A, RES, VALUES, loop_policy, obs_at)


def ringed():
    """The agent's 5x5 footprint pocket at IN_A, a one-cell occupied ring round it, unknown beyond; the west side five cells thick."""
    g = np.full((60, 120), -1, np.int8)
    g[27:34, 27:34] = 100
    g[28:33, 28:33] = 0
    g[27:34, 23:27] = 100
    return OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)


def test_the_ray_profile_reads_pocket_band_and_the_floor_beyond_without_inflation():
    world = ringed()
    pocket, barrier, beyond = ray_profile(world, IN_A, 0.0, 0.18, 3.0)
    assert barrier == pytest.approx(0.1, abs=0.051), "one ring cell to the east"
    assert beyond > 2.0, "unknown floor beyond it to the range"
    assert pocket == pytest.approx(0.1, abs=0.051)
    _, west_barrier, west_beyond = ray_profile(world, IN_A, math.pi, 0.18, 3.0)
    assert west_barrier == pytest.approx(0.5, abs=0.051), "five cells to the west: thicker than a probe pushes against"
    assert west_beyond > 2.0
    # A heading with no occupied cell in range is open: the pocket is the floor beyond.
    free = OccupancyGrid2D(np.zeros((60, 120), np.int8), OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)
    pocket, barrier, beyond = ray_profile(free, IN_A, 0.0, 0.18, 3.0)
    assert barrier == 0.0 and beyond == pytest.approx(pocket) and beyond > 2.5
    # A wall that fills the range is a band with nothing beyond.
    wall = OccupancyGrid2D(np.full((60, 120), 100, np.int8), OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)
    _, barrier, beyond = ray_profile(wall, IN_A, 0.0, 0.18, 3.0)
    assert barrier == pytest.approx(3.0, abs=0.2) and beyond == 0.0


def test_boxed_in_by_a_thin_ring_the_fallback_probes_the_thinnest_side_instead_of_holding():
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    no_rooms(policy)
    command = policy.loop.plan(obs_at(episode, 0, IN_A, yaw=0.0), ringed())
    assert command.info["kind"] == "escape" and command.info["fallback_stage"] == "escape"
    assert command.waypoints, "facing the one-cell side already: a forward step, not a turn"
    assert command.info["escape"]["barrier_m"] == pytest.approx(0.1, abs=0.051)
    assert policy.fallback.stats["escape"] == 1 and policy.fallback.stats["hold"] == 0
    assert policy.escape.stats == {"probes": 1, "turns": 0, "steps": 1, "blocked": 0}
    # Facing the thick west side, the probe turns toward a thin side first -- the nearest one, not straight ahead.
    command = policy.loop.plan(obs_at(episode, 1, IN_A, yaw=math.pi), ringed())
    assert command.info["kind"] == "escape" and not command.waypoints
    assert abs(normalize_angle(command.final_yaw - math.pi)) > episode.action_spec.turn_angle_rad / 2, "not into the five-cell band"
    assert command.info["escape"]["barrier_m"] <= 0.15 and command.info["escape"]["turns"] != 0
    assert all(c["turns"] != 0 for c in policy.escape.candidates(obs_at(episode, 1, IN_A, yaw=math.pi), ringed()))
    assert policy.escape.stats["turns"] == 1


def test_a_heading_the_simulator_blocked_is_not_probed_again_from_that_cell():
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    no_rooms(policy)
    boxed = ringed()
    obs = obs_at(episode, 0, IN_A, yaw=0.0)
    policy.mapping.worlds[policy.mapping.floor_id] = boxed
    assert policy.escape.candidates(obs, boxed)[0]["turns"] == 0
    policy.escape.note_blocked(obs)
    best = policy.escape.candidates(obs_at(episode, 1, IN_A, yaw=0.0), boxed)[0]
    assert best["turns"] != 0 and policy.escape.stats["blocked"] == 1
    # The memory expires, and the heading is worth a second try from the same cell.
    later = policy.escape.candidates(obs_at(episode, 1 + policy.escape.settings.memory_actions, IN_A, yaw=0.0), boxed)[0]
    assert later["turns"] == 0


def test_walled_in_by_a_band_thicker_than_the_probe_pushes_against_the_hold_is_still_the_last_move():
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    no_rooms(policy)
    g = np.full((60, 120), 100, np.int8)
    g[29:32, 29:32] = 0
    solid = OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)
    command = policy.loop.plan(obs_at(episode, 0, IN_A), solid)
    assert command.info["kind"] == "fallback_hold" and policy.fallback.stats["escape"] == 0
    assert policy.escape.last is None


def test_the_probe_can_be_switched_off_and_its_settings_are_validated():
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    no_rooms(policy)
    policy.escape = EscapeProbe(policy, EscapeSettings(enabled=False))
    command = policy.loop.plan(obs_at(episode, 0, IN_A), ringed())
    assert command.info["kind"] == "fallback_hold"
    for kwargs in ({"range_m": 0.0}, {"max_barrier_m": 3.0}, {"min_beyond_m": 5.0}, {"memory_actions": 0}):
        with pytest.raises(ValueError):
            EscapeSettings(**kwargs)


def test_the_episode_record_carries_the_probe():
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    no_rooms(policy)
    policy.loop.plan(obs_at(episode, 0, IN_A, yaw=0.0), ringed())
    record = policy.episode_info()["escape_probe"]
    assert record["stats"]["probes"] == 1 and record["last"]["step"] == 0
    assert record["settings"]["max_barrier_m"] == 0.4


def wedged():
    """IN_A in a 0.5 m wide pocket running 0.8 m to the north, closed by a thick band everywhere else (00824/000066).

    Thin barriers at the pocket's south and east were already refused at this
    cell; the open run north ends at a bed too thick to probe. Until 2026-10-25
    no heading qualified and the fallback held for 450 actions.
    """
    g = np.full((60, 120), 100, np.int8)
    g[28:41, 28:33] = 0           # a five-cell-wide pocket from the agent's row (y 2.8) north to y 4.1
    g[27, 28:33] = -1             # one unknown cell south: a thin band with nothing worth reaching behind it
    g[26, 28:33] = 100
    return OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)


def test_a_long_free_run_is_probed_even_when_a_thick_band_closes_it():
    policy, episode, world, rooms, _ = loop_policy(order=(1, 0))
    no_rooms(policy)
    pocket = wedged()
    obs = obs_at(episode, 0, IN_A, yaw=0.0)
    north = [c for c in policy.escape.candidates(obs, pocket) if abs(normalize_angle(c["heading"] - math.pi / 2)) < 0.01]
    assert north and north[0]["rule"] == "open_run" and north[0]["pocket_m"] >= 0.5, "0.8 m of free floor north is a step worth taking"
    assert north[0]["barrier_m"] > policy.escape.settings.max_barrier_m, "... although the band beyond it is no probe's"
    command = policy.loop.plan(obs, pocket)
    assert command.info["kind"] == "escape" and command.info["escape"]["rule"] == "open_run"
    assert abs(normalize_angle(command.final_yaw - math.pi / 2)) < 0.01, "turning toward the open run"
    # With the run under two steps the heading is no candidate, and nothing else here is: the hold.
    short = wedged()
    short.grid[32:41, 28:33] = 100
    assert not [c for c in policy.escape.candidates(obs_at(episode, 1, IN_A, yaw=0.0), short)
                if abs(normalize_angle(c["heading"] - math.pi / 2)) < 0.01]
    with pytest.raises(ValueError):
        EscapeSettings(min_pocket_m=0.0)
