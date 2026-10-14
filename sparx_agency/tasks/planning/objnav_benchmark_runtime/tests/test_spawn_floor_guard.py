"""``allow_stair_traversal=False`` (the default): stairs are known, masked and never taken."""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.peek_stairs import stair_peek_mask
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_settings import RPTSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import observation, setup_policy
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import (
    FLIGHT, IN_A, STAIRS, STRUCTURE, loop_policy, obs_at, stair_policy)


def forbidden_stairs(order=(STAIRS, 1, 0), stair_prob=0.9, metadata=STRUCTURE, **loop_overrides):
    """``stair_policy`` under the benchmark default: the staircase is seen and placed, traversal forbidden."""
    policy, episode, world, rooms, _ = loop_policy(order=order, stair_prob=stair_prob, metadata=metadata, **loop_overrides)
    obs = obs_at(episode, 0, IN_A)
    policy.mapping.atlas.update(obs.pose)
    for connector in policy.building.ground_truth.connectors:
        policy.building.sightings.mark_seen(connector.id, 0, connector.polyline)
    policy.building.observe(obs)
    assert len(policy.building.portals) == 1, "the seen staircase is still placed: knowledge, not a destination"
    return policy, episode, world, rooms


def at(episode, step, x, y, z=0.0, yaw=0.0, depth=0.7):
    return replace(observation(episode, step, depth=depth), pose=AgentPose(x, y, z, yaw))


def test_the_default_forbids_stair_traversal_and_the_frozen_configuration_records_it():
    settings = RPTSettings()
    assert settings.allow_stair_traversal is False and settings.floor_plane_bound_m == 0.5
    with pytest.raises(ValueError, match="allow_stair_traversal"):
        RPTSettings(allow_stair_traversal="no")
    with pytest.raises(ValueError, match="floor_plane_bound_m"):
        RPTSettings(floor_plane_bound_m=0.0)
    policy, _ = setup_policy("bed")
    configuration = policy.configuration()
    assert configuration["allow_stair_traversal"] is False and configuration["floor_plane_bound_m"] == 0.5
    assert policy.building.allowed is False and policy.floor_guard.enabled
    allowed, _ = setup_policy("bed", allow_stair_traversal=True)
    assert allowed.configuration()["allow_stair_traversal"] is True
    assert allowed.building.allowed and not allowed.floor_guard.enabled


def test_a_seen_staircase_is_neither_a_node_nor_committed_nor_the_fallbacks_last_resort():
    policy, episode, world, rooms = forbidden_stairs(order=(STAIRS, 1, 0), stair_prob=0.9)
    building, obs = policy.building, obs_at(episode, 0, IN_A)
    command = policy.loop.plan(obs, world)
    assert policy.reasoned_with[0]["stairs"] == [], "the oracle is shown rooms only"
    assert policy.loop.stats["stairs_offered"] == 0 and policy.loop._stairs == {}
    assert policy.loop.next_room in rooms and command.info["kind"] != "portal/0"
    assert building.active is None and building.phase == "SEARCH" and not building.committed
    # Defence in depth: a direct commit and the fallback's explicit rule are refused too.
    assert building.commit(obs, building.portals[0], FLIGHT[0][:2]) is False
    assert building.events[-1]["event"] == "stairs_refused" and "allow_stair_traversal" in building.events[-1]["reason"]
    assert building.plan(obs, world, exhausted=True) is None
    assert not [e for e in building.events if e["event"] in ("portal_selected", "floor_decision", "traversal_started")]
    assert building.diagnostics()["allow_stair_traversal"] is False
    # The same scene with traversal allowed offers the node (the multi-storey regressions cover the climb).
    allowed, episode, world, _, _ = stair_policy(order=(STAIRS, 1, 0), stair_prob=0.9)
    allowed.loop.plan(obs_at(episode, 0, IN_A), world)
    assert allowed.reasoned_with[0]["stairs"] == [STAIRS] and allowed.loop.stats["stairs_offered"] == 1


def test_the_seen_staircase_footprint_is_impassable_and_no_goal_on_it_is_accepted():
    policy, episode, world, _ = forbidden_stairs()
    obs = obs_at(episode, 0, IN_A)
    foot = world.world_to_grid(*FLIGHT[0][:2])
    assert world.is_free(*foot), "the hand-drawn room has free floor at the foot of the flight"
    confined = policy.floor_guard.observe(obs, world)
    mask = stair_peek_mask(policy, world)
    assert mask.any() and confined is not world
    assert (confined.grid[mask] == world.values.occupied).all()
    assert (confined.grid[~mask] == world.grid[~mask]).all(), "nothing but the footprint changed"
    assert world.is_free(*foot), "the observed world itself is untouched"
    assert policy.floor_guard.stats["stair_cells"] == int(mask.sum())
    assert not policy.floor_guard.goal_allowed(obs, world, FLIGHT[0][:2], "frontier")
    assert policy.floor_guard.goal_allowed(obs, world, IN_A, "frontier")
    assert policy._navigate(obs, confined, tuple(FLIGHT[0][:2]), "frontier") is None
    assert policy.floor_guard.stats["goals_refused"] == 2
    refused = [e for e in policy.floor_guard.events if e["event"] == "goal_refused"]
    assert refused[-1]["kind"] == "frontier" and "staircase" in refused[-1]["reason"]
    assert tuple(FLIGHT[0][:2]) in policy._visited_frontiers, "a refused frontier is retired, not retried"


def test_an_observed_drop_below_the_storey_plane_is_impassable_and_the_guard_is_inert_when_allowed():
    policy, episode, world, _ = forbidden_stairs()
    policy.mapping._anchor = 0.0
    # A flat depth of 3 m straight ahead: the frame's lower rows land 0.3-1 m BELOW the floor plane,
    # 3 m in front of the agent -- a descending flight or a stairwell as the camera sees it.
    obs = at(episode, 0, 3.0, 3.0, depth=3.0)
    confined = policy.floor_guard.observe(obs, world)
    drops = policy.floor_guard.drops[policy.mapping.floor_id]
    assert policy.floor_guard.stats["drop_cells"] == int(drops.sum()) > 0
    ys, xs = np.nonzero(drops)
    assert all(abs(world.grid_to_world(x, y)[0] - 6.0) <= world.resolution for x, y in zip(xs, ys)), "all 3 m ahead"
    assert (confined.grid[drops] == world.values.occupied).all()
    assert not policy.floor_guard.goal_allowed(obs, world, (6.0, 3.0), "frontier")
    assert policy.floor_guard.goal_allowed(obs, world, (4.5, 3.0), "frontier"), "the floor in between is fine"
    allowed, episode, world, _, _ = stair_policy(order=(STAIRS, 1, 0))
    allowed.mapping._anchor = 0.0
    same = allowed.floor_guard.observe(at(episode, 0, 3.0, 3.0, depth=3.0), world)
    assert same is world and allowed.floor_guard.mask is None and allowed.floor_guard.goal_allowed(obs, world, (6.0, 3.0), "x")


def test_a_storey_off_the_spawn_plane_refuses_every_goal_until_the_plane_is_regained():
    policy, episode, world, _ = forbidden_stairs()
    obs = obs_at(episode, 0, IN_A)
    policy.floor_guard.observe(obs, world)
    assert policy.floor_guard.spawn_z == 0.0 and not policy.floor_guard.off_plane
    policy.mapping._anchor = 2.7                                      # the atlas now calls the upper storey "this floor"
    policy.floor_guard.observe(at(episode, 1, 3.0, 3.0, z=2.7), world)
    assert policy.floor_guard.off_plane and policy.floor_guard.events[-1]["event"] == "spawn_plane_left"
    assert not policy.floor_guard.goal_allowed(obs, world, IN_A, "room_entry")
    assert "spawn plane" in policy.floor_guard.events[-1]["reason"]
    policy.mapping._anchor = 0.3                                      # within the 0.5 m plane: this floor again
    policy.floor_guard.observe(at(episode, 2, 3.0, 3.0, z=0.3), world)
    assert not policy.floor_guard.off_plane and policy.floor_guard.events[-1]["event"] == "spawn_plane_regained"
    assert policy.floor_guard.goal_allowed(obs, world, IN_A, "room_entry")
    assert policy.floor_guard.stats["off_plane_actions"] == 1
    assert policy.episode_info()["spawn_floor_guard"]["stats"]["goals_refused"] == 1


def test_an_unplanned_height_departure_starts_no_traversal_when_forbidden():
    policy, episode, world, _ = forbidden_stairs()
    building = policy.building
    stepped = at(episode, 1, 4.0, 1.2, z=0.6)                        # beyond departure_m, on the seen flight
    building.prepare_observation(stepped)
    assert building.transition is None and building.active is None and building.phase == "SEARCH"
    ignored = [e for e in building.events if e["event"] == "height_departure_ignored"]
    assert len(ignored) == 1 and "allow_stair_traversal" in ignored[0]["reason"]
    building.prepare_observation(at(episode, 2, 4.0, 1.25, z=0.7))
    assert len([e for e in building.events if e["event"] == "height_departure_ignored"]) == 1, "one event per excursion"
    allowed, episode, world, _, _ = stair_policy(order=(1, 0))
    allowed.building.prepare_observation(at(episode, 1, 4.0, 1.2, z=0.6))
    assert not [e for e in allowed.building.events
                if e["event"] == "height_departure_ignored" and "allow_stair_traversal" in e["reason"]]


def test_the_policy_plans_on_the_confined_world_and_the_recorder_sees_it():
    policy, episode, world, _ = forbidden_stairs()
    policy.plan(obs_at(episode, 0, IN_A))
    confined = policy.last_world
    mask = stair_peek_mask(policy, confined)
    assert mask.any() and (confined.grid[mask] == confined.values.occupied).all()
    assert policy.mapping.worlds[policy.mapping.floor_id] is confined, "the display panels read the same map"


def test_entrypoint_defaults_sit_below_the_policy_config_and_above_the_settings_default(tmp_path, monkeypatch):
    """``gibson.run._method``: RPTSettings default < entrypoint ``defaults`` < ``--policy-config``."""
    import json
    from types import SimpleNamespace
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson import run
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods import perception, services
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import FakeDetector, FakeLLM

    class Verified(FakeLLM):
        def __init__(self, client):
            super().__init__()

        def health(self):
            return {"model": "fake", "digest": "0"}

    class Detector(FakeDetector):
        def __init__(self, url, vocabulary, expected_backend=None, timeout_s=30.0):
            super().__init__("bed")
            self.timeout_s = timeout_s

        def health(self):
            return {"metadata": {"backend": "yolo_world", "detector_config": {"conf_thresh": 0.05}}}

    monkeypatch.setattr(services, "VerifiedLLMClient", Verified)
    monkeypatch.setattr(perception, "HttpDetector", Detector)
    config = tmp_path / "policy.json"
    config.write_text(json.dumps({"allow_stair_traversal": False}))

    def args(policy_config=None):
        return SimpleNamespace(agent="rpt", policy_config=policy_config, explorer=None, seed=0,
                               detector_url="http://127.0.0.1:1", detector_backend="yolo_world", detector_timeout_s=30.0)

    assert run._method(args())[1]["allow_stair_traversal"] is False
    assert run._method(args(), defaults={"allow_stair_traversal": True})[1]["allow_stair_traversal"] is True
    assert run._method(args(config), defaults={"allow_stair_traversal": True})[1]["allow_stair_traversal"] is False


def test_a_settled_half_level_is_adopted_as_part_of_the_spawn_storey_after_thirty_actions():
    """Klickitat 2026-10-05: a split-level house, the agent on a plateau 0.6 m up that the atlas read as a
    landing for ever -- no map integrated, every route unplannable, 361 idle turns. After HALF_LEVEL_ACTIONS
    on the settled plateau the guard adopts it: a floor of its own, goals allowed, the storey below not a drop.
    A plateau a storey away (1.5 m or more) is never adopted."""
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.spawn_floor_guard import HALF_LEVEL_ACTIONS
    policy, episode, world, _ = forbidden_stairs()
    atlas = policy.mapping.atlas
    policy.floor_guard.observe(obs_at(episode, 0, IN_A), world)
    assert policy.floor_guard.spawn_z == 0.0
    # Walk up onto a plateau 0.6 m above the spawn plane and settle there (translated, level).
    step = 1
    for i in range(6):
        pose = at(episode, step, 3.0 + 0.3 * i, 3.0, z=0.6)
        atlas.update(pose.pose)
        step += 1
    assert atlas.in_transition and atlas.destination_height_m == pytest.approx(0.6, abs=0.05)
    assert atlas.diagnostics()["transition_ticks"] >= 1 and len(atlas.floors) == 1, "a landing to the atlas: no floor"
    # Not yet thirty actions on it: still confined, nothing adopted.
    policy.floor_guard.observe(at(episode, step, 4.5, 3.0, z=0.6), world)
    assert policy.floor_guard.half_levels == set() and atlas.in_transition
    while atlas.transition_ticks < HALF_LEVEL_ACTIONS:
        atlas.update(at(episode, step, 4.5, 3.0, z=0.6).pose)
        step += 1
    obs = at(episode, step, 4.5, 3.0, z=0.6)
    policy.floor_guard.observe(obs, world)
    assert not atlas.in_transition and len(atlas.floors) == 2 and atlas.elevation_m == pytest.approx(0.6, abs=0.05)
    assert policy.floor_guard.half_levels == {1} and policy.floor_guard.stats["half_levels_adopted"] == 1
    adopted = [e for e in policy.floor_guard.events if e["event"] == "half_level_adopted"]
    assert len(adopted) == 1 and adopted[0]["floor_id"] == 1 and adopted[0]["after_actions"] >= HALF_LEVEL_ACTIONS
    assert atlas.completion_reason == "half_level_adopted"
    # The adopted level is the spawn storey to the guard: goals on it are allowed.
    policy.mapping.floor_id, policy.mapping._anchor = 1, atlas.elevation_m
    policy.floor_guard.observe(at(episode, step + 1, 4.5, 3.0, z=0.6), world)
    assert not policy.floor_guard.off_plane and policy.floor_guard.goal_allowed(obs, world, (4.0, 3.0), "room_entry")
    assert policy.episode_info()["spawn_floor_guard"]["half_levels"] == [1]
    # A plateau a storey away is another storey: the atlas makes a floor of it itself, the guard adopts
    # nothing, and every goal on it is refused as before.
    policy, episode, world, _ = forbidden_stairs()
    atlas = policy.mapping.atlas
    policy.floor_guard.observe(obs_at(episode, 0, IN_A), world)
    for i in range(HALF_LEVEL_ACTIONS + 10):
        atlas.update(at(episode, i + 1, 3.0 + 0.3 * min(i, 5), 3.0, z=1.6).pose)
    assert not atlas.in_transition and len(atlas.floors) == 2 and atlas.completion_reason == "destination_platform_confirmed"
    policy.mapping.floor_id, policy.mapping._anchor = atlas.active_id, atlas.elevation_m
    policy.floor_guard.observe(at(episode, 60, 4.5, 3.0, z=1.6), world)
    assert policy.floor_guard.half_levels == set() and policy.floor_guard.stats["half_levels_adopted"] == 0
    assert policy.floor_guard.off_plane and not policy.floor_guard.goal_allowed(obs, world, (4.0, 3.0), "room_entry")


# -- the mask never traps the agent (Collierville/000000, 2026-10-08: 309 idle turns 0.6 m from the foot) --------
RUN_UP = [[4.0, 0.2, 0.0], [4.0, 1.0, 0.0]] + FLIGHT[1:]          # a flat run on the floor, then the same flight
RUN_UP_STRUCTURE = dict(STRUCTURE, stair_connectors=[dict(STRUCTURE["stair_connectors"][0], bottom_xyz=RUN_UP[0],
                                                           polyline_xyz=RUN_UP, length_m=4.8)])


def test_the_floor_run_of_a_flight_is_not_masked_while_its_treads_are():
    """The polyline begins on the storey's own floor; within FLOOR_RUN_M of the plane it is floor, not flight."""
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.peek_stairs import FLOOR_RUN_M
    policy, episode, world, _ = forbidden_stairs(metadata=RUN_UP_STRUCTURE)
    mask = stair_peek_mask(policy, world)
    assert mask.any()
    gx, gy = world.world_to_grid(4.0, 0.2)
    assert not mask[gy, gx], "the flat run-up on the floor is walkable"
    gx, gy = world.world_to_grid(4.0, 1.3)
    assert mask[gy, gx], "the first treads (0.27 m up) are masked"
    gx, gy = world.world_to_grid(4.0, 1.5)
    assert mask[gy, gx], "... up to the departure band"
    # The mask starts where the flight leaves the floor run: 0.1 m of rise is 0.11 m of run here.
    ys, xs = np.nonzero(mask)
    assert min(world.grid_to_world(x, y)[1] for x, y in zip(xs, ys)) >= 1.0 + FLOOR_RUN_M / 0.9 - 0.7, "capsule margin only"


def test_an_agent_inside_the_stair_capsule_is_never_on_an_occupied_cell_and_can_plan_out(monkeypatch):
    """Collierville: the agent walked past the foot of a flight 0.6 m from its centreline; the capsule covered
    its cell as a GENUINE obstacle, A* had no start, and the fallback held for the rest of the episode."""
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods import peek_stairs, spawn_floor_guard
    policy, episode, world, _ = forbidden_stairs()
    beside = (4.6, 1.3)                                              # 0.6 m east of the flight's first treads
    obs = at(episode, 0, *beside)
    cell = world.world_to_grid(*beside)
    assert world.is_free(*cell)
    # Without the exemption the capsule swallows the cell: the regression.
    zeros = lambda policy_, world_, pose=None: np.zeros(world_.grid.shape, dtype=bool)
    monkeypatch.setattr(peek_stairs, "walked_exemption", zeros)
    monkeypatch.setattr(spawn_floor_guard, "walked_exemption", zeros)
    trapped = policy.floor_guard.observe(obs, world)
    assert trapped.grid[cell[1], cell[0]] == world.values.occupied
    assert policy._navigate(obs, trapped, IN_A, "frontier") is None, "no start cell: every goal fails"
    monkeypatch.undo()
    # With it the agent's own footprint is carved out of the mask and A* leaves the capsule.
    confined = policy.floor_guard.observe(obs, world)
    assert confined.grid[cell[1], cell[0]] == world.values.free
    assert policy.floor_guard.stats["walked_exempt_cells"] > 0
    assert stair_peek_mask(policy, confined, obs.pose)[cell[1], cell[0]] == False
    command = policy._navigate(obs, confined, IN_A, "frontier")
    assert command is not None and command.waypoints, "a route out of the capsule exists"
    # The flight itself is still impassable: no goal on it, no route onto it.
    assert not policy.floor_guard.goal_allowed(obs, confined, (4.0, 1.4), "frontier")
    assert policy._navigate(obs, confined, (4.0, 1.4), "frontier") is None


def test_the_trail_stays_carved_after_the_agent_moves_on():
    """Cells the agent stood on remain passable in later masks, so the way back always exists."""
    policy, episode, world, _ = forbidden_stairs()
    for step, xy in enumerate(((4.6, 1.3), (4.6, 1.6), (4.6, 1.9))):
        obs = at(episode, step, *xy)
        confined = policy.floor_guard.observe(obs, world)
        policy.sight.observe(obs, confined)
    later = policy.floor_guard.observe(at(episode, 3, 3.0, 3.0), world)
    for xy in ((4.6, 1.3), (4.6, 1.6), (4.6, 1.9)):
        gx, gy = world.world_to_grid(*xy)
        assert later.grid[gy, gx] == world.values.free, "the trail is never masked"
    assert policy.floor_guard.goal_allowed(at(episode, 3, 3.0, 3.0), later, (4.6, 1.6), "frontier")


# -- the capsule never closes a corridor (Markleeville/000000, 2026-10-08) -------------------------------------
def along_the_wall(x):
    """The same flight, standing ``x`` from the west face of the wall at x = 5.5 (room A's east wall)."""
    flight = [[x, 1.0, 0.0], [x, 2.0, 0.9], [x, 3.0, 1.8], [x, 4.0, 2.7]]
    return dict(STRUCTURE, stair_connectors=[dict(STRUCTURE["stair_connectors"][0], bottom_xyz=flight[0],
                                                   top_xyz=flight[-1], polyline_xyz=flight)])


def test_the_stair_capsule_shrinks_to_keep_the_corridor_between_the_flight_and_the_wall_passable():
    """Markleeville: the only way from the south of the room to the north ran between the stairwell and the
    wall, 0.55 m wide; the 0.6 m capsule left 0.25 m of it. Here the flight stands 0.7 m from room A's east
    wall with a cabinet run closing its west side, so that corridor is the only way north."""
    from scipy import ndimage
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.peek_stairs import _passable
    policy, episode, world, _ = forbidden_stairs(metadata=along_the_wall(4.8))     # 0.7 m from the wall
    world.grid[5:55, 30:46] = world.values.occupied                                 # x 3.0-4.6, y 0.5-5.5: no way round
    obs = obs_at(episode, 0, (5.25, 0.7))                                            # at the corridor's south end
    confined = policy.floor_guard.observe(obs, world)
    margins = policy.floor_guard.stair_margins_m
    preferred = policy.settings.body_radius_m + policy.converter_params.goal_tolerance_m + world.resolution
    assert list(margins.values()) == [0.3] and 0.3 < preferred, "six cells asked for, three drawn: the widest that leaves a body-wide corridor"
    body = int(math.ceil(policy.settings.body_radius_m / world.resolution))
    free = confined.grid == world.values.free
    passable = _passable(free, ~free, body)
    labels, _ = ndimage.label(passable, structure=np.ones((3, 3)))
    south, north = world.world_to_grid(5.3, 0.5), world.world_to_grid(5.3, 4.8)
    assert labels[south[1], south[0]] and labels[south[1], south[0]] == labels[north[1], north[0]], \
        "the corridor along the wall still joins the south of the room to the north"
    assert policy._navigate(obs, confined, (5.3, 4.8), "frontier") is not None, "A* uses it"
    for y in (1.3, 1.5):
        gx, gy = world.world_to_grid(4.8, y)
        assert confined.grid[gy, gx] == world.values.occupied, "the first treads stay impassable"
    assert not policy.floor_guard.goal_allowed(obs, confined, (4.8, 1.4), "frontier")


def test_the_capsule_keeps_its_preferred_margin_where_nothing_is_in_the_way():
    policy, episode, world, _ = forbidden_stairs(metadata=along_the_wall(2.0))     # 3.5 m of open floor to the wall
    policy.floor_guard.observe(obs_at(episode, 0, IN_A), world)
    preferred = policy.settings.body_radius_m + policy.converter_params.goal_tolerance_m + world.resolution
    assert list(policy.floor_guard.stair_margins_m.values()) == [round(math.ceil(preferred / world.resolution) * world.resolution, 2)]
    assert policy.episode_info()["spawn_floor_guard"]["stair_margins_m"] == policy.floor_guard.stair_margins_m
