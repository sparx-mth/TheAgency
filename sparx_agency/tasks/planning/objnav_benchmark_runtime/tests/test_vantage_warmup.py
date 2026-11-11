"""The vantage warm-up (2026-11-11): the greedy look-around that may walk to where the camera learns more.

CPU-only, like ``test_discovery.py``: real geometry and planning on hand-built
maps, scripted perception, the body moved by the plan's own actions.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.core.planning.object_search.exploration.vantage_warmup import SPIN, VANTAGE, WarmupSettings
from sparx_agency.core.planning.object_search.settings import RPTSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import setup_policy, observation

RES = 0.1
FREE, OCC, UNK = 0, 100, -1
VALUES = OccupancyValues(free=FREE, occupied=OCC, unknown=UNK)
TURN = math.radians(30)


def cells(x0, x1, y0, y1):
    """Grid slices of the world rectangle ``[x0, x1) x [y0, y1)`` on the 20 m map about the origin."""
    return (slice(int(round((y0 + 10) / RES)), int(round((y1 + 10) / RES))),
            slice(int(round((x0 + 10) / RES)), int(round((x1 + 10) / RES))))


def grid_world(g):
    return OccupancyGrid2D(g, OccupancyGrid2DParams(RES, -10.0, -10.0, "world"), values=VALUES)


def alcove_world():
    """A dead-end alcove 1 m wide and 1.5 m deep about the spawn, open to the east onto a big unknown room.

    The first frame (facing east) saw the mouth and a little floor beyond it. A spin here sees three walls
    within 0.6 m; a step or two east shows the room.
    """
    g = np.full((200, 200), UNK, np.int8)
    g[cells(-0.5, 1.0, -0.5, 0.5)] = FREE
    g[cells(-0.6, -0.5, -0.6, 0.6)] = OCC
    g[cells(-0.6, 1.0, 0.5, 0.6)] = OCC
    g[cells(-0.6, 1.0, -0.6, -0.5)] = OCC
    g[cells(1.0, 2.5, -1.0, 1.0)] = FREE
    return grid_world(g)


def cone_world():
    """An open room of which only the first frame's cone is known (79 degrees to 5 m, facing +x), the rest unknown."""
    g = np.full((200, 200), UNK, np.int8)
    ys, xs = np.mgrid[0:200, 0:200]
    wx, wy = (xs + 0.5) * RES - 10.0, (ys + 0.5) * RES - 10.0
    r, b = np.hypot(wx, wy), np.arctan2(wy, wx)
    g[(r <= 5.0) & (np.abs(b) <= math.radians(39.5))] = FREE
    return grid_world(g)


def known_room():
    """A 10 m hall, every cell known: nothing anywhere for a look to reveal."""
    g = np.full((200, 200), UNK, np.int8)
    g[cells(-5.0, 5.0, -5.0, 5.0)] = FREE
    g[cells(-5.1, 5.1, 5.0, 5.1)] = OCC
    g[cells(-5.1, 5.1, -5.1, -5.0)] = OCC
    g[cells(-5.1, -5.0, -5.1, 5.1)] = OCC
    g[cells(5.0, 5.1, -5.1, 5.1)] = OCC
    return grid_world(g)


def vantage_policy(world, **warmup):
    """A discovery-phase policy whose map is the hand-built ``world`` every action; the detector sees a table."""
    settings = dict({"mode": VANTAGE, "budget_steps": 20, "max_points": 1, "detection_range_m": 5.0}, **warmup)
    p, episode = setup_policy("table", discovery=True, doorway_peek={"enabled": False}, warmup=settings)
    p.mapping.update = lambda observation, **kwargs: world
    return p, episode


def drive(p, episode, pose, steps=60):
    """Run the warm-up to its end: every action moves the body as the command says. Returns the log and the pose."""
    log = []
    for step in range(steps):
        obs = replace(observation(episode, step, depth=3.0), pose=pose)
        command = p.plan(obs)
        info = command.info
        if info.get("kind") != "warmup":
            log.append(dict(step=step, kind=info.get("kind"), phase=info.get("phase")))
            return log, pose
        if command.actions:
            action = command.actions[0]
            if action == DiscreteAction.MOVE_FORWARD:
                pose = AgentPose(pose.x + 0.25 * math.cos(pose.yaw), pose.y + 0.25 * math.sin(pose.yaw), 0.0, pose.yaw)
            else:
                pose = AgentPose(pose.x, pose.y, 0.0, pose.yaw + (TURN if action == DiscreteAction.TURN_LEFT else -TURN))
        else:
            assert command.final_yaw is not None and not command.waypoints, "a warm-up turn is a turn in place"
            delta = (command.final_yaw - pose.yaw + math.pi) % (2 * math.pi) - math.pi
            action = DiscreteAction.TURN_LEFT if delta > 0 else DiscreteAction.TURN_RIGHT
            pose = AgentPose(pose.x, pose.y, 0.0, command.final_yaw)
        p.notify_action(obs, action)
        log.append(dict(step=step, kind="warmup", choice=info.get("choice"), action=action.name,
                        xy=(round(pose.x, 2), round(pose.y, 2)), yaw_deg=round(math.degrees(pose.yaw))))
    raise AssertionError("the warm-up did not end within %d actions" % steps)


def test_the_default_is_the_spin_and_the_vantage_settings_validate():
    s = RPTSettings()
    assert s.warmup.mode == SPIN and s.warmup_steps == 12
    p, _ = setup_policy(discovery=True)
    assert p.warmup_turns == 10 and p.warmup_active() and p.settings.warmup.mode == SPIN
    assert RPTSettings(warmup={"mode": "vantage", "budget_steps": 20}).warmup.budget_steps == 20
    for bad in (dict(mode="look"), dict(budget_steps=0), dict(max_points=-1), dict(min_rate_m2_per_action=-0.1),
                dict(detection_range_m=0.0), dict(unknown_free_run_m=0.0), dict(candidate_spacing_m=-1.0),
                dict(plan_attempts=0)):
        with pytest.raises(ValueError):
            WarmupSettings(**bad)
    with pytest.raises(ValueError):
        RPTSettings(warmup="vantage")


def test_a_confined_spawn_walks_to_the_mouth_looks_from_there_and_gives_the_unspent_budget_back():
    p, episode = vantage_policy(alcove_world(), budget_steps=20, max_points=1)
    assert p.warmup_turns == 20, "the vantage warm-up's bound is its action budget"
    log, pose = drive(p, episode, AgentPose(0.0, 0.0, 0.0, 0.0))
    warm = [row for row in log if row["kind"] == "warmup"]
    assert warm[0]["choice"] == "move", "three walls within 0.6 m: no turn here is worth an action, the mouth is"
    moves = [row for row in warm if row["choice"] == "move"]
    turns = [row for row in warm if row["choice"] in ("turn_left", "turn_right")]
    assert 3 <= len(moves) <= 8 and turns, "a few steps east, then the look from the mouth"
    assert all(row["xy"][0] >= 1.0 for row in turns), "every turn is taken at the mouth, none in the alcove"
    assert p.warmup_actions == len(warm) < 20 and p._warmup_done, "ended before the budget: the rest is the search's"
    assert not p.warmup_active() and log[-1]["kind"] != "warmup"
    d = p.vantage_warmup.diagnostics()
    assert d["stats"]["moves"] == 1 and d["stats"]["arrivals"] == 1 and d["stats"]["early_end"] == 1
    assert len(d["spots"]) == 2 and d["spots"][0]["xy"] == [0.0, 0.0]
    assert [e["event"] for e in d["events"]][:3] == ["look_spot", "look_walk", "look_arrived"]
    walk = next(e for e in d["events"] if e["event"] == "look_walk")
    assert walk["rate"] > 0.25 and walk["geodesic_m"] >= 1.0
    assert [r["source"] for r in p.scans.records] == ["warmup", "warmup"], (
        "both spots are completed look-arounds: the alcove's walls leave nothing to turn toward, the mouth's circle was swept")
    assert p.episode_info()["warmup"]["mode"] == VANTAGE and p.episode_info()["warmup"]["vantage"]["moves"] == 1


def test_a_spawn_that_sees_only_its_first_cone_spins_in_place_as_the_spin_does():
    p, episode = vantage_policy(cone_world(), budget_steps=20, max_points=1)
    log, pose = drive(p, episode, AgentPose(0.0, 0.0, 0.0, 0.0))
    warm = [row for row in log if row["kind"] == "warmup"]
    assert len(warm) == 10 and all(row["choice"] == "turn_left" for row in warm), (
        "every sliver of unknown is worth about the same, and a turn is the cheapest look: ten turns, one direction")
    assert all(row["xy"] == (0.0, 0.0) for row in warm)
    assert p.warmup_actions == 10 and p._warmup_done
    assert len(p.scans.records) == 1 and p.scans.records[0]["source"] == "warmup"
    assert p.vantage_warmup.diagnostics()["spots"][0]["uncovered_deg"] == 0.0


def test_a_fully_known_room_ends_the_warmup_at_once_and_records_the_spot_as_looked_around():
    p, episode = vantage_policy(known_room(), budget_steps=20, max_points=2)
    log, pose = drive(p, episode, AgentPose(0.0, 0.0, 0.0, 0.0))
    assert log[0]["kind"] != "warmup" and p.warmup_actions == 0 and p._warmup_done
    assert len(p.scans.records) == 1, "nothing a look from here could add: the look-around is complete without a turn"
    assert p.vantage_warmup.diagnostics()["decisions"][0]["choice"] == "end"


def test_max_points_zero_never_walks_and_the_budget_bounds_the_warmup():
    p, episode = vantage_policy(alcove_world(), budget_steps=20, max_points=0)
    log, pose = drive(p, episode, AgentPose(0.0, 0.0, 0.0, 0.0))
    assert all(row["kind"] != "warmup" or row["choice"] != "move" for row in log)
    assert p.warmup_actions == 0, "walls all round and no walk allowed: nothing worth a turn"
    p, episode = vantage_policy(cone_world(), budget_steps=3, max_points=1)
    log, pose = drive(p, episode, AgentPose(0.0, 0.0, 0.0, 0.0))
    assert p.warmup_actions == 3 and [row["choice"] for row in log[:3]] == ["turn_left"] * 3
    assert log[3]["kind"] != "warmup" and p.vantage_warmup.diagnostics()["stats"]["budget_end"] == 1


def test_the_vantage_warmup_counts_only_its_own_actions_and_the_glance_gate_reads_it():
    p, episode = vantage_policy(cone_world(), budget_steps=20)
    obs = replace(observation(episode, 0, depth=3.0), pose=AgentPose(0.0, 0.0, 0.0, 0.0))
    p.plan(obs)
    assert p._action_owner == "warmup" and not p.glances._may_glance(obs)
    p.notify_action(obs, DiscreteAction.TURN_LEFT)
    p.notify_action(obs, DiscreteAction.TURN_LEFT)
    assert p.warmup_actions == 1, "one action per step, however often the ledger is told"
    p._warmup_done = True
    assert not p.warmup_active()
    p._action_owner = "glance"
    p.notify_action(replace(obs, step=1), DiscreteAction.TURN_LEFT)
    assert p.warmup_actions == 1, "a suspended phase after the warm-up ended is not a warm-up action"


def test_the_configuration_describes_the_warmup_in_force():
    p, _ = setup_policy(discovery=True)
    assert p.configuration()["warmup"]["mode"] == SPIN and "in place" in p.configuration()["warmup"]["rule"]
    p, _ = setup_policy(discovery=True, warmup={"mode": "vantage", "budget_steps": 15, "max_points": 2})
    conf = p.configuration()["warmup"]
    assert conf["mode"] == VANTAGE and conf["budget_steps"] == 15 and conf["max_points"] == 2
    assert "useful area per action" in conf["rule"]
    assert p.configuration()["adaptation"]["warmup"]["budget_steps"] == 15


def test_the_first_sighting_of_the_target_is_recorded_whatever_the_closing_does_with_it():
    p, episode = setup_policy("chair")
    assert p.perception.target_first_seen_step is None
    p.plan(observation(episode, 0))
    diag = p.episode_info()["perception"]
    assert diag["target_first_seen_step"] == 0 and diag["target_first_placed_step"] == 0
    p.plan(observation(episode, 1))
    assert p.perception.target_first_seen_step == 0, "the first, not the latest"
    other, episode = setup_policy("table")
    other.plan(observation(episode, 0))
    assert other.perception.target_first_seen_step is None and other.perception.target_first_placed_step is None
