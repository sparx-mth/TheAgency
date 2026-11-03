"""``allow_stair_traversal=False`` (the default): stairs are known, masked and never taken."""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.core.planning.object_search.world_model.peek_stairs import stair_peek_mask
from sparx_agency.core.localization.floor_atlas import MultiFloorParams
from sparx_agency.core.planning.object_search.settings import RPTSettings
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
    # The connector's anchor is the storey's own floor in front of the first riser: not masked, a goal like any
    # other (the square foot, 2026-10-21); the first treads, 0.27 m up, are the staircase.
    assert confined.grid[foot[1], foot[0]] == world.values.free and policy.floor_guard.goal_allowed(obs, world, FLIGHT[0][:2], "frontier")
    treads = (4.0, 1.3)
    assert not policy.floor_guard.goal_allowed(obs, world, treads, "frontier")
    assert policy.floor_guard.goal_allowed(obs, world, IN_A, "frontier")
    assert policy._navigate(obs, confined, treads, "frontier") is None
    assert policy.floor_guard.stats["goals_refused"] == 2
    refused = [e for e in policy.floor_guard.events if e["event"] == "goal_refused"]
    assert refused[-1]["kind"] == "frontier" and "staircase" in refused[-1]["reason"]
    assert treads in policy._visited_frontiers, "a refused frontier is retired, not retried"


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
    """``policy_factory.build_policy`` (behind ``gibson.run._method``): RPTSettings default < entrypoint ``defaults`` < ``--policy-config``."""
    import json
    from types import SimpleNamespace
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson import run
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.services import http_detector, llm
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

    monkeypatch.setattr(llm, "VerifiedLLMClient", Verified)
    monkeypatch.setattr(http_detector, "HttpDetector", Detector)
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
    from sparx_agency.core.planning.object_search.multifloor.spawn_floor_guard import HALF_LEVEL_ACTIONS
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
    from sparx_agency.core.planning.object_search.world_model.peek_stairs import FLOOR_RUN_M
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


def test_the_capsule_begins_at_the_flight_and_not_a_margin_before_it():
    """Darden (2026-10-21): the flight's foot stands in the junction of the hallway and the room beside it, and the
    capsule's round end -- 0.55 m of plain floor in front of the first riser -- closed the junction on the planning
    grid. The storey end is cut square; the sides keep their margin and the far end its cap."""
    from sparx_agency.core.planning.object_search.world_model.peek_stairs import FLOOR_RUN_M, _capsule, _foot_of
    policy, episode, world, _ = forbidden_stairs()
    confined = policy.floor_guard.observe(obs_at(episode, 0, IN_A), world)
    first_drawn = 1.0 + FLOOR_RUN_M / 0.9                         # the flight leaves the floor run here (slope 0.9)
    for y in (first_drawn - 0.15, first_drawn - 0.35, first_drawn - 0.5):
        gx, gy = world.world_to_grid(4.0, y)
        assert confined.grid[gy, gx] == world.values.free, "the floor in front of the first riser at y=%.2f is floor" % y
    for xy in ((4.0, first_drawn + 0.1), (4.4, first_drawn + 0.1), (3.6, first_drawn + 0.1), (4.4, 1.45)):
        gx, gy = world.world_to_grid(*xy)
        assert confined.grid[gy, gx] == world.values.occupied, "the flight and its side margin at %s stay masked" % (xy,)
    gx, gy = world.world_to_grid(4.0, 1.5 + 0.3)                   # the far end keeps its cap: more flight behind it
    assert confined.grid[gy, gx] == world.values.occupied
    # The cut is the half-disc behind the foot and nothing else: the body's own cells are all still drawn.
    segments = [((40, 10), (40, 15))]
    plain, square = _capsule((60, 60), segments, 5), _capsule((60, 60), segments, 5, _foot_of(segments))
    assert square.sum() < plain.sum() and not square[:9, :].any() and (square[10:16, 35:46] == plain[10:16, 35:46]).all()
    assert _foot_of([((40, 10), (40, 10))]) is None, "a point flight has no direction to cut across"


def test_an_agent_inside_the_stair_capsule_is_never_on_an_occupied_cell_and_can_plan_out(monkeypatch):
    """Collierville: the agent walked past the foot of a flight 0.6 m from its centreline; the capsule covered
    its cell as a GENUINE obstacle, A* had no start, and the fallback held for the rest of the episode."""
    from sparx_agency.core.planning.object_search.world_model import peek_stairs
    from sparx_agency.core.planning.object_search.multifloor import spawn_floor_guard
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
    wall, 0.55 m wide; the 0.6 m capsule left 0.25 m of it. Here the flight stands 0.9 m from room A's east
    wall with a cabinet run closing its west side, so that corridor is the only way north (0.7 m until
    2026-10-25: the 0.4 m the body-radius capsule then left is under the planner's floor rung of 0.25 m --
    a squeeze a non-sliding body cannot walk -- and is no corridor to spare any more)."""
    from scipy import ndimage
    from sparx_agency.core.planning.object_search.world_model.peek_stairs import _passable
    policy, episode, world, _ = forbidden_stairs(metadata=along_the_wall(4.6))     # 0.9 m from the wall
    world.grid[5:55, 30:44] = world.values.occupied                                 # x 3.0-4.4, y 0.5-5.5: no way round
    obs = obs_at(episode, 0, (5.25, 0.7))                                            # at the corridor's south end
    confined = policy.floor_guard.observe(obs, world)
    margins = policy.floor_guard.stair_margins_m
    preferred = policy.settings.stair_margin_m
    # Five cells asked for, three drawn: the widest that leaves a corridor the planner's floor rung
    # (``min_clearance_m`` 0.25 m, three cells; the body radius alone until 2026-10-25) can stand in.
    assert list(margins.values()) == [0.3] and 0.3 < preferred, "five cells asked for, three drawn: the widest that leaves a rung-wide corridor"
    rung = int(math.ceil(policy.planner_params.inflate_floor_m / world.resolution))
    free = confined.grid == world.values.free
    passable = _passable(free, ~free, rung)
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
    preferred = policy.settings.stair_margin_m
    assert preferred == 0.45 and list(policy.floor_guard.stair_margins_m.values()) == [
        round(math.ceil(preferred / world.resolution) * world.resolution, 2)] == [0.5], (
        "stairs only, 0.5 m drawn (2026-10-21; the implicit body + tolerance + cell drew 0.6)")
    assert policy.episode_info()["spawn_floor_guard"]["stair_margins_m"] == policy.floor_guard.stair_margins_m
    # The knob is the stairs' alone and never under the body radius.
    from sparx_agency.core.planning.object_search.settings import RPTSettings
    with pytest.raises(ValueError):
        RPTSettings(stair_margin_m=0.1)
    wider, episode_w, world_w, _ = forbidden_stairs(metadata=along_the_wall(2.0), settings={"stair_margin_m": 0.55})
    wider.floor_guard.observe(obs_at(episode_w, 0, IN_A), world_w)
    assert list(wider.floor_guard.stair_margins_m.values()) == [0.6], "the older six cells, asked for explicitly"


def test_the_drop_threshold_is_a_knob_at_the_atlas_tolerance_and_a_flights_floor_is_a_drop_its_treads_are_not():
    """``RPTSettings.step_down_m`` (0.30 m): a surface this far below the plane is a drop. Not lower -- at 0.15 m
    Corozal/000142 painted 3,316 cells of its own storey as drops (2026-10-14), the plane being the pose height and
    the floor the camera sees lying 0.15-0.30 m under it there. A flight's lower floor at -0.45 m is a drop; its
    shallow treads (-0.11, -0.18, -0.21 m) are not, and remain the open item."""
    assert RPTSettings().step_down_m == 0.30
    with pytest.raises(ValueError, match="step_down_m"):
        RPTSettings(step_down_m=0.0)
    policy, episode, world, _ = forbidden_stairs()
    policy.mapping._anchor = 0.0
    assert policy.floor_guard.drop_m == 0.30
    k = episode.camera.intrinsics
    camera_z = episode.camera.height_m

    def frame_of(z_surface, step):
        depth = np.full((k.height, k.width), 4.0, np.float32)
        for v in range(k.height):
            dz = (v - k.cy) / k.fy
            if dz > 1e-3:
                depth[v, :] = min(4.0, (camera_z - z_surface) / dz)
        return replace(observation(episode, step), depth_m=depth, pose=AgentPose(3.0, 3.0, 0.0, 0.0))
    for z in (-0.11, -0.18, -0.21):
        policy.floor_guard.observe(frame_of(z, 1), world)
        assert policy.floor_guard.stats["drop_cells"] == 0, "a tread inside the tolerance is not a drop"
    confined = policy.floor_guard.observe(frame_of(-0.45, 2), world)
    drops = policy.floor_guard.drops[policy.mapping.floor_id]
    assert policy.floor_guard.stats["drop_cells"] > 0 and (confined.grid[drops] == world.values.occupied).all()
    ys, xs = np.nonzero(drops)
    assert not policy.floor_guard.goal_allowed(frame_of(-0.45, 3), world, world.grid_to_world(xs[0], ys[0]), "frontier")
    # A tighter knob masks a floor 0.2 m under the pose plane -- the Corozal storey -- which the default does not.
    tight, episode2, world2, _ = forbidden_stairs(settings={"step_down_m": 0.15})
    tight.mapping._anchor = 0.0
    tight.floor_guard.observe(frame_of(-0.20, 1), world2)
    assert tight.floor_guard.stats["drop_cells"] > 0, "0.15: the storey's own floor, 0.2 m under the pose, is a drop"


# -- a flight walked down by accident (00814/000009, 00873/000049, 00873/000002, 2026-10-25) -----------------
def test_a_flight_walked_down_with_traversal_forbidden_is_walked_back_up_the_trail_not_crashed_on(monkeypatch):
    """Observed stairs, traversal forbidden: the atlas alone says the agent is off its storey after three treads.

    Until 2026-10-25 ``MultiFloorSearch.plan`` dereferenced the traversal it
    had refused to start, on every action for the rest of the episode (417,
    108 and 941 recorded decision failures in the three HM3D runs). Now the
    coordinator walks the storey trail back; the atlas records a source
    return, and the search resumes.
    """
    policy, episode = setup_policy("bed", multifloor={"stair_source": "observed"})
    building, atlas = policy.building, policy.mapping.atlas
    # A storey: its level support is of floor size (``confirm_area_m2``), so walking it confirms it.
    monkeypatch.setattr(type(building.terrain), "plateau_area", lambda self, pose: 6.0)
    assert building.ground_truth is None and not building.allowed
    step = 0
    for i in range(6):                                                    # along the storey
        policy.plan(at(episode, step, 1.0 + 0.25 * i, 1.0, depth=3.0)); step += 1
    for z in (-0.18, -0.35, -0.52):                                       # three treads down a flight the route led onto
        policy.plan(at(episode, step, 1.0 + 0.25 * step, 1.0, z=z, depth=3.0)); step += 1
    assert atlas.in_transition and building.transition is None and building.traversing
    command = policy.plan(at(episode, step, 1.0 + 0.25 * step, 1.0, z=-0.52, depth=3.0)); step += 1
    assert not policy.fallback.failures, "no decision failure: the missing traversal is not dereferenced"
    assert building.phase == "RETREAT" and command.info["kind"] == "RETREAT" and command.info["phase"] == "RETREAT"
    assert command.waypoints[0] == pytest.approx((1.0 + 0.25 * (step - 1), 1.0))
    assert list(command.waypoints[-1]) == pytest.approx([1.0, 1.0]), "back along the trail to its earliest storey pose"
    started = [e for e in building.events if e["event"] == "retreat_started"]
    assert len(started) == 1 and started[0]["height_m"] == pytest.approx(-0.52) and started[0]["trail_points"] == 6
    assert policy.filter_action(at(episode, step, 1.0 + 0.25 * step, 1.0, z=-0.52), DiscreteAction.MOVE_FORWARD) == DiscreteAction.MOVE_FORWARD, \
        "the terrain cannot veto a step onto ground the body stood on"
    x = 1.0 + 0.25 * step
    for z in (-0.35, -0.18, 0.0, 0.0, 0.0, 0.0, 0.0):                     # up the treads and onto the plane
        x -= 0.25
        policy.plan(at(episode, step, x, 1.0, z=z, depth=3.0)); step += 1
    assert not atlas.in_transition and atlas.completion_reason == "source_platform_returned"
    assert building._retreat is None and building.phase == "SEARCH" and not building.traversing
    ended = [e for e in building.events if e["event"] == "retreat_ended"]
    assert len(ended) == 1 and ended[0]["reason"] == "source_platform_returned"
    assert not policy.fallback.failures
    # A retreat that regains nothing within the budget is given up once, and the search goes on without a crash.
    policy, episode = setup_policy("bed", multifloor={"stair_source": "observed"})
    building = policy.building
    policy.plan(at(episode, 0, 1.0, 1.0, depth=3.0))
    policy.plan(at(episode, 1, 1.25, 1.0, depth=3.0))
    policy.plan(at(episode, 2, 1.5, 1.0, depth=3.0))                   # two level steps: the storey is confirmed
    assert building.storey_confirmed
    for step in range(3, 3 + building.params.retreat_actions + 2):
        command = policy.plan(at(episode, step, 2.0, 1.0, z=-0.6, depth=3.0))
    assert [e["reason"] for e in building.events if e["event"] == "retreat_given_up"] == ["retreat_actions exhausted off the storey plane"]
    assert command.info["kind"] != "RETREAT" and not policy.fallback.failures


def test_a_retreat_refused_on_its_trail_is_given_up_and_the_stalled_plateau_is_adopted():
    """00802/000015 and /000088 of the GT-oracle sweep (2026-10-26): off the storey, the trail retreat pushed at a
    ledge for 351 and 394 actions (334 and 371 refused steps), and when it was given up the map stayed frozen
    with the transition -- the atlas's plateau test wants translation an agent boxed in cannot make. Since
    2026-10-28 ``retreat_blocked`` refused steps end the retreat, the phase reads SEARCH from then on, and after
    HALF_LEVEL_ACTIONS at one height the guard adopts the plateau without the translated settle."""
    from sparx_agency.core.planning.object_search.multifloor.spawn_floor_guard import HALF_LEVEL_ACTIONS, STALLED_SAMPLES
    policy, episode = setup_policy("bed", multifloor={"stair_source": "observed"})
    building, atlas, guard = policy.building, policy.mapping.atlas, policy.floor_guard
    assert building.params.retreat_blocked == 6
    for step in range(12):
        policy.plan(at(episode, step, 1.0 + 0.25 * step, 1.0, depth=3.0))          # the storey is confirmed, a long trail
    step = 12
    command = policy.plan(at(episode, step, 4.0, 1.0, z=-0.6, depth=3.0))         # dropped off a ledge
    assert building.retreating and building.phase == "RETREAT" and command.info["kind"] == "RETREAT"
    for _ in range(building.params.retreat_blocked):                              # every step back up is refused
        obs = at(episode, step, 4.0, 1.0, z=-0.6, depth=3.0)
        policy.notify_blocked(obs)
        step += 1
        command = policy.plan(obs)
    assert not building.retreating and building.phase == "SEARCH" and command.info["kind"] != "RETREAT"
    assert [e["reason"] for e in building.events if e["event"] == "retreat_given_up"] == [
        "%d refused steps on the storey trail" % building.params.retreat_blocked]
    assert building.diagnostics()["trail_retreat"]["blocked"] == building.params.retreat_blocked
    assert step < building.params.retreat_actions, "given up long before the action allowance"
    assert atlas.in_transition and not policy.fallback.failures
    # Standing still at one height: the plateau the atlas cannot settle is adopted once the wait is over.
    while atlas.transition_ticks < HALF_LEVEL_ACTIONS + STALLED_SAMPLES and atlas.in_transition:
        policy.plan(at(episode, step, 4.0, 1.0, z=-0.6, depth=3.0)); step += 1
    assert not atlas.in_transition and atlas.completion_reason == "half_level_adopted"
    assert guard.half_levels == {1} and guard.stats["half_levels_adopted"] == 1 and guard.stats["stalled_adoptions"] == 1
    adopted = [e for e in guard.events if e["event"] == "half_level_adopted"]
    assert len(adopted) == 1 and adopted[0]["stalled"] is True and adopted[0]["height_m"] == pytest.approx(-0.6, abs=0.05)
    policy.plan(at(episode, step, 4.0, 1.0, z=-0.6, depth=3.0)); step += 1       # the mapping takes the atlas's floor
    assert policy.mapping.floor_id == 1 and building.phase == "SEARCH" and not policy.fallback.failures
    assert policy.mapping.worlds[1].grid.max() >= 0, "the new level is mapped: integration resumed with the transition over"
    # Within ``floor_match_m`` of the plane -- the ledge climbed but for a step -- the storey itself is confirmed in place.
    policy, episode = setup_policy("bed", multifloor={"stair_source": "observed"})
    building, atlas, guard = policy.building, policy.mapping.atlas, policy.floor_guard
    for step in range(12):
        policy.plan(at(episode, step, 1.0 + 0.25 * step, 1.0, depth=3.0))
    policy.plan(at(episode, 12, 4.0, 1.0, z=-0.6, depth=3.0))
    assert atlas.in_transition and building.retreating
    step = 13
    for _ in range(building.params.retreat_blocked):
        obs = at(episode, step, 3.75, 1.0, z=-0.2, depth=3.0)                     # one step up, no further
        policy.notify_blocked(obs)
        step += 1
        policy.plan(obs)
    assert not building.retreating and atlas.in_transition
    while atlas.transition_ticks < HALF_LEVEL_ACTIONS + STALLED_SAMPLES and atlas.in_transition:
        policy.plan(at(episode, step, 3.75, 1.0, z=-0.2, depth=3.0)); step += 1
    assert not atlas.in_transition and atlas.completion_reason == "source_platform_returned"
    assert guard.half_levels == set() and guard.stats["stalled_returns"] == 1 and len(atlas.floors) == 1
    assert [e for e in guard.events if e["event"] == "transition_settled_in_place"][0]["offset_m"] == pytest.approx(-0.2, abs=0.05)
    assert policy.mapping.floor_id == 0 and not policy.fallback.failures
    with pytest.raises(ValueError):
        MultiFloorParams(retreat_blocked=0)


def test_a_one_pose_trail_is_extended_past_its_last_point_and_the_return_is_confirmed_at_the_plane_height():
    """00873/000002 as replayed on 2026-10-25 with the retreat alone: the spawn was a landing and the first step a
    tread, so the storey trail was one pose; the retreat called it spent 0.25 m short of the plane, the atlas kept
    waiting for three translated samples that a blocked agent never gave, and `notify_blocked` -- short-circuited
    while "traversing" -- let the escape rung push into one wall 473 times. Now the trail is extended two steps
    past its last pose, the return is confirmed the action the body is back at the plane height, and a refused
    step is remembered by the probe and written to the map as on the storey.

    Since 2026-10-26 a storey never walked level is provisional and its departure a traversal, not a retreat
    (``test_a_departure_from_a_spawn_storey_never_walked_level_is_a_traversal_to_the_nearer_storey``); the
    one-pose retreat below is the CONFIRMED storey's -- the trail starts afresh after every arrival."""
    policy, episode = setup_policy("bed", multifloor={"stair_source": "observed"})
    building, atlas = policy.building, policy.mapping.atlas
    policy.plan(at(episode, 0, 1.0, 1.0, depth=3.0))
    policy.plan(at(episode, 1, 1.0, 1.0, yaw=0.5, depth=3.0))
    building.storey_confirmed = True                                    # a storey the agent has stood on
    step = 2
    for z in (-0.18, -0.35, -0.52):
        policy.plan(at(episode, step, 1.0 + 0.25 * (step - 1), 1.0, z=z, depth=3.0)); step += 1
    command = policy.plan(at(episode, step, 1.75, 1.0, z=-0.52, depth=3.0)); step += 1
    assert command.info["kind"] == "RETREAT" and list(command.waypoints[-1]) == pytest.approx([1.0, 1.0])
    x = 1.75
    for z in (-0.35, -0.18):
        x -= 0.25
        command = policy.plan(at(episode, step, x, 1.0, z=z, depth=3.0)); step += 1
        assert command.info["kind"] == "RETREAT", "0.25 m from the one trail pose and still a tread down: not spent"
    assert command.waypoints[-1][0] < 1.0, "the goal lies past the trail's last pose, onto the storey"
    assert not [e for e in building.events if e["event"] == "retreat_given_up"]
    x -= 0.25
    command = policy.plan(at(episode, step, x, 1.0, z=-0.02, depth=3.0)); step += 1
    assert not atlas.in_transition and atlas.completion_reason == "source_platform_returned"
    assert building.phase == "SEARCH" and command.info["kind"] != "RETREAT"
    assert [e["event"] for e in building.events if e["event"].startswith("retreat")] == ["retreat_started", "retreat_ended"]
    # A refused step on the storey after the return is the probe's and the map's business again.
    obs = at(episode, step, x, 1.0, z=-0.02, depth=3.0)
    policy.plan(obs)
    policy.notify_blocked(obs)
    assert policy.escape.stats["blocked"] == 1 and policy.mapping.integrity()[0]["bumps"] == 1
    assert not policy.fallback.failures


# -- a spawn on the stairs (00873/000002, ``02cc26973cab``, 2026-10-26) ------------------------------------------
def test_a_departure_from_a_spawn_storey_never_walked_level_is_a_traversal_to_the_nearer_storey(monkeypatch):
    """The first pose was a tread: the atlas's floor 0 is the middle of a flight. The replay of 2026-10-25 retreated
    to it and then spent 447 actions boxed in by the two masked flights. A storey never walked ``stable_distance_m``
    level is provisional; its departure is walked as a traversal -- toward the nearer storey the terrain can see,
    else the way already taken -- and never retreated from."""
    policy, episode = setup_policy("bed", multifloor={"stair_source": "observed"})
    building, atlas = policy.building, policy.mapping.atlas
    assert not building.allowed and not building.storey_confirmed
    policy.plan(at(episode, 0, 1.0, 1.0, depth=3.0))
    policy.plan(at(episode, 1, 1.0, 1.0, yaw=0.5, depth=3.0))              # the warm-up turns in place: still provisional
    assert not building.storey_confirmed
    step = 2
    for z in (-0.18, -0.35, -0.52):                                        # the frontier route leads down the flight
        command = policy.plan(at(episode, step, 1.0 + 0.25 * (step - 1), 1.0, z=z, depth=3.0)); step += 1
    assert atlas.in_transition and building.transition is not None and building.traversing
    assert building.phase == "TRAVERSE" and command.info["phase"] == "TRAVERSE"
    assert building._retreat is None and not [e for e in building.events if e["event"].startswith("retreat")]
    provisional = [e for e in building.events if e["event"] == "spawn_storey_provisional"]
    assert len(provisional) == 1 and provisional[0]["direction"] == "down" and provisional[0]["flights_seen"] == []
    assert provisional[0]["height_m"] == pytest.approx(-0.52)
    started = [e for e in building.events if e["event"] == "traversal_started"]
    assert len(started) == 1 and started[0]["direction"] == -1 and started[0]["selected_by"] == "unplanned"
    assert building.diagnostics()["storey_confirmed"] is False
    assert not policy.fallback.failures
    # The terrain sees a flight up with the smaller rise: that is the nearer storey, whichever way the first step went.
    policy, episode = setup_policy("bed", multifloor={"stair_source": "observed"})
    building = policy.building
    policy.plan(at(episode, 0, 1.0, 1.0, depth=3.0))
    building.terrain.candidates = [{"direction": -1, "rise_m": 2.4, "path": []}, {"direction": 1, "rise_m": 0.5, "path": []}]
    building.terrain._candidate_paths = lambda pose: None                 # keep the stubbed flights through the frame's update
    policy.plan(at(episode, 1, 1.25, 1.0, z=-0.52, depth=3.0))
    provisional = [e for e in building.events if e["event"] == "spawn_storey_provisional"]
    assert provisional[0]["direction"] == "up" and building.active["direction"] == 1
    assert provisional[0]["flights_seen"] == [{"direction": "up", "rise_m": 0.5}, {"direction": "down", "rise_m": 2.4}]
    # Two level steps first on a storey of floor size, and the same flight is the accidental one: the trail retreat.
    policy, episode = setup_policy("bed", multifloor={"stair_source": "observed"})
    building = policy.building
    monkeypatch.setattr(type(building.terrain), "plateau_area", lambda self, pose: 6.0)
    for i in range(3):
        policy.plan(at(episode, i, 1.0 + 0.25 * i, 1.0, depth=3.0))
    assert building.storey_confirmed
    for i, z in enumerate((-0.18, -0.35, -0.52)):
        command = policy.plan(at(episode, 3 + i, 1.75 + 0.25 * i, 1.0, z=z, depth=3.0))
    assert building.transition is None and building.phase == "RETREAT" and command.info["kind"] == "RETREAT"
    assert not [e for e in building.events if e["event"] == "spawn_storey_provisional"]


def test_a_level_walk_across_a_landing_keeps_the_spawn_storey_provisional_and_the_flight_is_walked(monkeypatch):
    """00800/000040 (2026-11-02): 0.35 m across a 1.7 m2 landing and onto the first tread confirmed the landing a
    storey; the reach fell to departure_m, the flight above was masked and 460 actions went to escaping a box
    0.4 x 0.6 m. A level walk confirms the storey only on support of floor size (``confirm_area_m2``)."""
    policy, episode = setup_policy("bed", multifloor={"stair_source": "observed"})
    building, atlas = policy.building, policy.mapping.atlas
    monkeypatch.setattr(type(building.terrain), "plateau_area", lambda self, pose: 1.7)
    for i in range(3):                                                     # 0.5 m level across the landing
        policy.plan(at(episode, i, 1.0 + 0.25 * i, 1.0, depth=3.0))
    assert not building.storey_confirmed and policy.floor_guard.provisional
    assert policy.floor_guard.step_reach_m == pytest.approx(policy.settings.step_surface_band_m)
    kept = [e for e in building.events if e["event"] == "spawn_storey_kept_provisional"]
    assert len(kept) == 1 and kept[0]["plateau_m2"] == pytest.approx(1.7)
    step = 3
    for z in (0.18, 0.35, 0.52):                                           # up the flight
        command = policy.plan(at(episode, step, 1.5 + 0.25 * (step - 2), 1.0, z=z, depth=3.0)); step += 1
    assert atlas.in_transition and building.transition is not None and building.phase == "TRAVERSE"
    assert [e["direction"] for e in building.events if e["event"] == "spawn_storey_provisional"] == ["up"]
    assert building._retreat is None and command.info["phase"] == "TRAVERSE"
    # The same walk on support of floor size confirms the storey, and the flight is the accidental one as before.
    policy, episode = setup_policy("bed", multifloor={"stair_source": "observed"})
    building = policy.building
    monkeypatch.setattr(type(building.terrain), "plateau_area", lambda self, pose: 6.0)
    for i in range(3):
        policy.plan(at(episode, i, 1.0 + 0.25 * i, 1.0, depth=3.0))
    assert building.storey_confirmed
    assert [e["plateau_m2"] for e in building.events if e["event"] == "spawn_storey_confirmed"] == [6.0]
    assert MultiFloorParams().confirm_area_m2 == pytest.approx(5.0)
    with pytest.raises(ValueError, match="confirm_area_m2"):
        MultiFloorParams(confirm_area_m2=0.0)


def test_the_storey_a_provisional_spawn_settles_on_becomes_the_spawn_plane():
    """Up three treads onto a floor 0.5 m above the tread the agent spawned on: a settled plateau of floor size is
    adopted at once -- not after thirty actions -- and becomes the spawn plane itself, with goals on it allowed.
    Down to a storey the atlas creates on its own: the plane moves there too, where before every goal on it was
    refused as "off the spawn plane"."""
    policy, episode = setup_policy("bed", multifloor={"stair_source": "observed"})
    building, atlas, guard = policy.building, policy.mapping.atlas, policy.floor_guard
    policy.plan(at(episode, 0, 1.0, 1.0, depth=3.0))
    assert guard.spawn_z == 0.0 and guard.spawn_floor == 0 and guard.provisional
    step = 1
    for z in (0.18, 0.35, 0.52):                                           # up the flight the agent spawned on
        policy.plan(at(episode, step, 1.0 + 0.25 * step, 1.0, z=z, depth=3.0)); step += 1
    assert building.transition is not None and building.phase == "TRAVERSE"
    building.terrain.plateau_area = atlas.plateau_observer = lambda pose: 4.0   # the floor at the top is of storey size
    for i in range(4):                                                      # settled: level, translated past the samples
        policy.plan(at(episode, step, 2.0 + 0.25 * i, 1.0, z=0.5, depth=3.0)); step += 1
        if atlas.active_id != 0:
            break
    assert atlas.active_id == 1 and not atlas.in_transition and atlas.completion_reason == "half_level_adopted"
    assert atlas.diagnostics()["transition_ticks"] == 0
    rebased = [e for e in guard.events if e["event"] == "spawn_plane_rebased"]
    assert len(rebased) == 1 and rebased[0]["floor_id"] == 1 and rebased[0]["height_m"] == pytest.approx(0.5, abs=0.05)
    assert rebased[0]["after_actions"] < 30, "no thirty-action wait on a storey the agent never stood on"
    assert guard.spawn_z == pytest.approx(0.5, abs=0.05) and guard.spawn_floor == 1 and guard.half_levels == set()
    assert guard.stats["rebased"] == 1 and guard.stats["half_levels_adopted"] == 0
    assert building.storey_confirmed and building.transition is None and building.phase == "SEARCH"
    assert [e["event"] for e in building.events if e["event"] in ("floor_arrival", "retreat_started")] == ["floor_arrival"]
    obs = at(episode, step, 3.0, 1.0, z=0.5, depth=3.0)
    policy.plan(obs)
    assert not guard.off_plane and guard.goal_allowed(obs, policy.last_world, (3.5, 1.0), "room_entry")
    assert not guard.provisional and policy.mapping.floor_id == 1
    assert policy.episode_info()["spawn_floor_guard"]["spawn_floor"] == 1
    assert not policy.fallback.failures
    # Down a whole storey: the atlas's own arrival, and the plane follows it instead of refusing every goal.
    policy, episode = setup_policy("bed", multifloor={"stair_source": "observed"})
    building, atlas, guard = policy.building, policy.mapping.atlas, policy.floor_guard
    policy.plan(at(episode, 0, 1.0, 1.0, depth=3.0))
    building.terrain.plateau_area = atlas.plateau_observer = lambda pose: 4.0
    step = 1
    for z in (-0.4, -0.8, -1.2, -1.6, -2.0):
        policy.plan(at(episode, step, 1.0 + 0.25 * step, 1.0, z=z, depth=3.0)); step += 1
    assert building.transition is not None and atlas.in_transition
    for i in range(8):
        policy.plan(at(episode, step, 2.25 + 0.25 * i, 1.0, z=-2.0, depth=3.0)); step += 1
        if atlas.active_id != 0:
            break
    assert atlas.active_id == 1 and atlas.completion_reason == "destination_platform_confirmed"
    assert guard.spawn_z == pytest.approx(-2.0, abs=0.05) and guard.spawn_floor == 1 and not guard.off_plane
    assert [e["reason"] for e in guard.events if e["event"] == "spawn_plane_rebased"] == ["the atlas settled on a storey from the provisional spawn storey"]
    assert building.storey_confirmed and building.transition is None
    obs = at(episode, step, 4.5, 1.0, z=-2.0, depth=3.0)
    policy.plan(obs)
    assert guard.goal_allowed(obs, policy.last_world, (5.0, 1.0), "room_entry") and guard.stats["goals_refused"] == 0
    assert not policy.fallback.failures


def test_a_stair_spawn_on_a_landing_walks_the_flight_it_saw_to_the_nearer_storey_on_purpose():
    """00800/000028 (2026-11-02): provisional, the flight up seen, 300 tread cells walkable -- and 500 actions at the
    spawn height, because no frontier route led onto the flight. From ``spawn_leave_after_actions`` a spawn storey
    still provisional, with a flight seen and less than ``min_floor_area_m2`` of level support under the agent,
    walks the flight to the nearer storey; once."""
    def run(plateau, flights):
        policy, episode = setup_policy("bed", multifloor={"stair_source": "observed"})
        building = policy.building
        building.terrain.plateau_area = lambda pose: plateau
        building.terrain._candidate_paths = lambda pose: None              # keep the stubbed flights through updates
        events = []
        for step in range(35):
            if step == 14 and flights:
                building.terrain.candidates = [dict(c) for c in flights]   # the flight comes into view, then goes
            if step == 16:
                building.terrain.candidates = []
            policy.plan(at(episode, step, 1.0, 1.0, yaw=0.3 * step, depth=3.0))
            events = [e for e in building.events if e["event"] == "spawn_stairs_departure"]
            if events:
                return policy, building, step, events
        return policy, building, None, events

    flights = [{"direction": -1, "rise_m": 2.4, "path": [(0.5, 1.0, -0.2), (0.0, 1.0, -0.4)]},
               {"direction": 1, "rise_m": 1.3, "path": [(1.5, 1.0, 0.2), (2.0, 1.0, 0.4)]}]
    policy, building, step, events = run(1.0, flights)
    assert step == policy.building.params.spawn_leave_after_actions
    assert len(events) == 1 and events[0]["direction"] == "up" and building.transition is not None
    assert building.active["direction"] == 1 and building.phase == "TRAVERSE" and not building.storey_confirmed
    assert building._spawn_departure_due(at(policy.episode, step + 1, 1.0, 1.0)) is False, "once"
    # Support of floor size under the agent, or no flight ever seen from the spawn: no departure.
    assert run(4.0, flights)[2] is None
    assert run(1.0, [])[2] is None


def test_a_traversal_that_never_takes_a_tread_settles_back_and_makes_no_second_floor_at_the_same_height():
    """2026-11-02: departures and switches that stalled at their source were adopted after thirty actions as a NEW
    floor 0.01-0.12 m from the old one (00802/000056, 00814/000004, 00800/000086) -- "floor changes" that were none.
    A plateau within ``floor_match_m`` of a known floor is that floor: the transition settles back on it."""
    policy, episode = setup_policy("bed", multifloor={"stair_source": "observed"})
    building, atlas, guard = policy.building, policy.mapping.atlas, policy.floor_guard
    building.terrain.plateau_area = lambda pose: 1.0
    atlas.plateau_observer = lambda pose: 1.0
    building.terrain._candidate_paths = lambda pose: None
    flights = [{"direction": 1, "rise_m": 1.3, "path": [(1.5, 1.0, 0.2), (2.0, 1.0, 0.4)]}]
    step = 0
    for step in range(31):
        if step == 14:
            building.terrain.candidates = [dict(c) for c in flights]
        if step == 16:
            building.terrain.candidates = []
        policy.plan(at(episode, step, 1.0, 1.0, yaw=0.3 * step, depth=3.0))
    assert building.transition is not None and atlas.in_transition, "the departure began"
    for step in range(31, 140):                  # the flight never found: walking about at the source, no tread
        policy.plan(at(episode, step, 1.0 + 0.3 * (step % 4), 1.0 + 0.3 * ((step // 4) % 2), yaw=0.3 * step, depth=3.0))
    assert len(atlas.floors) == 1 and atlas.active_id == 0, [f.elevation_m for f in atlas.floors.values()]
    assert not atlas.in_transition and building.transition is None and building.phase == "SEARCH"
    assert [e["event"] for e in guard.events if e["event"] == "transition_settled_on_known_floor"]
    assert not [e for e in building.events if e["event"] == "floor_arrival"]
    assert not policy.fallback.failures


def test_a_departure_bumped_back_to_the_spawn_height_is_retried_a_bounded_number_of_times():
    """00802/000003 (2026-11-02): the departure was bumped back onto the spawn tread's height, the atlas called it a
    return, and the agent bounced between two treads in the explorer's escape for 400 actions. Each departure that
    ends back at the spawn storey's height is retried ``spawn_leave_after_actions`` later, at most
    ``spawn_departure_attempts`` in all; never while one is under way."""
    policy, episode = setup_policy("bed", multifloor={"stair_source": "observed"})
    building, atlas, params = policy.building, policy.mapping.atlas, policy.building.params
    building.terrain.plateau_area = lambda pose: 1.0
    atlas.plateau_observer = lambda pose: 1.0
    building.terrain._candidate_paths = lambda pose: None
    flights = [{"direction": 1, "rise_m": 1.3, "path": [(1.5, 1.0, 0.2), (2.0, 1.0, 0.4)]}]
    for step in range(400):
        if step == 14:
            building.terrain.candidates = [dict(c) for c in flights]
        if step == 16:
            building.terrain.candidates = []
        policy.plan(at(episode, step, 1.0 + 0.3 * (step % 4), 1.0 + 0.3 * ((step // 4) % 2), yaw=0.3 * step, depth=3.0))
    starts = [e["action"] for e in building.events if e["event"] == "spawn_stairs_departure"]
    ends = [e["action"] for e in building.events if e["event"] == "portal_deferred"]
    assert [e["attempt"] for e in building.events if e["event"] == "spawn_stairs_departure"] == [1, 2, 3]
    assert starts[0] == params.spawn_leave_after_actions
    assert all(later - end >= params.spawn_leave_after_actions for end, later in zip(ends, starts[1:]))
    assert all(start < end for start, end in zip(starts, ends)), "never one while one is under way"
    assert len(atlas.floors) == 1 and not building.storey_confirmed, "no storey reached, none made"
    assert building._spawn_departure_due(at(episode, 400, 1.0, 1.0)) is False


def test_a_stair_spawn_searches_the_nearer_storey_then_switches_once_to_the_flights_other_end():
    """2026-11-02: the nearer storey holds the target in 112 of 174 stair starts; in most of the rest it is the
    flight's other end (00800/000038 went down, the bed was up). Once the first storey is searched out the agent
    walks back along the flight to the far storey -- once: two storeys at most, never back."""
    policy, episode = setup_policy("bed", multifloor={"stair_source": "observed"})
    building, atlas, guard = policy.building, policy.mapping.atlas, policy.floor_guard
    building.terrain.plateau_area = atlas.plateau_observer = lambda pose: 6.0
    policy.plan(at(episode, 0, 1.0, 1.0, depth=3.0))
    step = 1
    for z in (0.18, 0.35, 0.52):                                           # up the flight the agent spawned on
        policy.plan(at(episode, step, 1.0 + 0.25 * step, 1.0, z=z, depth=3.0)); step += 1
    for i in range(6):
        policy.plan(at(episode, step, 2.0 + 0.25 * i, 1.0, z=0.5, depth=3.0)); step += 1
        if atlas.active_id != 0:
            break
    assert atlas.active_id == 1 and guard.spawn_floor == 1
    switch = building.switch
    assert switch.flight is not None and switch.flight["direction"] == -1 and switch.state is None
    armed = [e for e in building.events if e["event"] == "spawn_switch_armed"]
    assert len(armed) == 1 and armed[0]["towards"] == "down" and armed[0]["how"] == "arrival"
    entry = switch.flight["entry"]
    # Its guide is the climb the atlas measured, walked back: storey first, the spawn at the far end (00814/000004
    # wandered the storey's floor beside the flight with only the settle point to aim from).
    guide = switch.flight["guide"]
    assert armed[0]["guide_points"] == len(guide) >= 4
    assert guide[0][2] == pytest.approx(0.5) and guide[-1][:2] == pytest.approx((1.0, 1.0))
    heights = [round(p[2], 2) for p in guide]
    assert heights.index(0.35) < heights.index(0.18) < heights.index(0.0), "down the treads, storey to spawn"
    building.can_leave_floor = lambda obs=None: True                     # the storey's rooms are all peeked
    early = at(episode, step, entry[0], entry[1], z=0.5, depth=3.0)
    assert building.plan(early, policy.last_world, exhausted=True) is None and switch.state is None, "too early"
    step = switch.flight["step"] + building.params.spawn_switch_min_floor_actions
    assert building.plan(at(episode, step, entry[0] + 3.0, entry[1], z=0.5, depth=3.0), policy.last_world) is None, (
        "not searched out: nothing")
    late = at(episode, step, entry[0], entry[1], z=0.5, depth=3.0)
    command = building.plan(late, policy.last_world, exhausted=True)
    assert switch.state == "traverse" and building.transition is not None and building.active["direction"] == -1
    assert building.active["path"] == [entry] + guide
    assert command is not None and [e["direction"] for e in building.events if e["event"] == "spawn_switch_started"] == ["down"]
    # Down past the tread it spawned on to a storey 1.7 m under the first: within 1.5 m of that tread "floor",
    # so the atlas alone would call it a landing; the switch's traversal settles it and it is adopted.
    step += 1
    x = entry[0]
    for z in (0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9, -1.2):
        x -= 0.25
        policy.plan(at(episode, step, x, 1.0, z=z, depth=3.0)); step += 1
    for i in range(10):
        x -= 0.25
        policy.plan(at(episode, step, x, 1.0, z=-1.2, depth=3.0)); step += 1
        if switch.state == "done":
            break
    assert switch.state == "done" and atlas.active_id == 2 and guard.spawn_floor == 2
    assert guard.spawn_z == pytest.approx(-1.2, abs=0.05) and building.transition is None
    assert [e["event"] for e in building.events if e["event"].startswith("spawn_switch")] == [
        "spawn_switch_armed", "spawn_switch_started", "spawn_switch_adopted", "spawn_switch_arrived"]
    # Searched out again, later: nothing arms or starts again, and the agent never goes back up.
    final = at(episode, step + 200, x, 1.0, z=-1.2, depth=3.0)
    assert building.plan(final, policy.last_world, exhausted=True) is None and building.active is None
    assert len([e for e in building.events if e["event"] == "floor_arrival"]) == 2
    assert policy.episode_info()["spawn_floor_guard"]["spawn_floor"] == 2


def test_the_spawn_switch_also_fires_after_a_long_unfruitful_search_but_never_with_a_target_known():
    """00800/000038 (2026-11-02): fourteen rooms kept frontier on the storey below for all 450 actions while the plant
    was upstairs. After ``spawn_switch_after_actions`` on the first storey the switch fires unasked -- unless a target
    is known there."""
    def settled():
        policy, episode = setup_policy("bed", multifloor={"stair_source": "observed"})
        building, atlas = policy.building, policy.mapping.atlas
        building.terrain.plateau_area = atlas.plateau_observer = lambda pose: 6.0
        policy.plan(at(episode, 0, 1.0, 1.0, depth=3.0))
        step = 1
        for z in (0.18, 0.35, 0.52):
            policy.plan(at(episode, step, 1.0 + 0.25 * step, 1.0, z=z, depth=3.0)); step += 1
        for i in range(6):
            policy.plan(at(episode, step, 2.0 + 0.25 * i, 1.0, z=0.5, depth=3.0)); step += 1
            if atlas.active_id != 0:
                break
        assert building.switch.flight is not None
        return policy, episode, building
    policy, episode, building = settled()
    switch, params = building.switch, building.params
    entry = switch.flight["entry"]
    before = switch.flight["step"] + params.spawn_switch_after_actions - 1
    assert building.plan(at(episode, before, entry[0], entry[1], z=0.5), policy.last_world) is None and switch.state is None
    building.plan(at(episode, before + 1, entry[0], entry[1], z=0.5), policy.last_world)
    assert switch.state == "traverse" and building.transition is not None
    # A target the pursuit remembers on this storey keeps the agent here.
    policy, episode, building = settled()
    policy.pursuit.targets.append({"xyz": (3.0, 3.0, 0.5), "floor": building.floor_id, "label": "bed"})
    late = building.switch.flight["step"] + params.spawn_switch_after_actions + 5
    assert building.plan(at(episode, late, entry[0], entry[1], z=0.5), policy.last_world) is None
    assert building.switch.state is None
    # Too few actions left for the far storey: no switch either.
    policy, episode, building = settled()
    last = int(policy.episode.max_steps) - params.spawn_switch_min_actions_left + 1
    assert building.plan(at(episode, last, entry[0], entry[1], z=0.5), policy.last_world) is None
    assert building.switch.state is None


def test_a_landing_reached_from_a_stair_spawn_is_never_adopted_as_its_storey():
    """00843/000022 (v3, 2026-11-02): a climb from a spawn tread stalled on a landing 0.85 m up; after thirty actions
    the guard adopted it as the spawn storey, which armed the switch, which walked the agent back down to the
    spawn's height -- up and down between "floors". A plateau under ``min_floor_area_m2`` reached from a stair
    spawn is a landing however long the agent stands on it."""
    policy, episode = setup_policy("bed", multifloor={"stair_source": "observed"})
    building, atlas, guard = policy.building, policy.mapping.atlas, policy.floor_guard
    building.terrain.plateau_area = atlas.plateau_observer = lambda pose: 1.0
    building.terrain._candidate_paths = lambda pose: None
    flights = [{"direction": 1, "rise_m": 1.3, "path": [(1.5, 1.0, 0.2), (2.0, 1.0, 0.6)]}]
    for step in range(31):
        if step == 14:
            building.terrain.candidates = [dict(c) for c in flights]
        if step == 16:
            building.terrain.candidates = []
        policy.plan(at(episode, step, 1.0, 1.0, yaw=0.3 * step, depth=3.0))
    assert atlas.in_transition and building.transition is not None
    for step in range(31, 120):                 # up onto the landing and about on it, never further
        z = min(0.6, 0.2 * (step - 30))
        policy.plan(at(episode, step, 1.5 + 0.3 * (step % 3), 1.0 + 0.3 * ((step // 3) % 2), z=z, depth=3.0))
        assert len(atlas.floors) == 1 and atlas.active_id == 0, (step, [f.elevation_m for f in atlas.floors.values()])
    assert building.switch.flight is None and not guard.stats["stalled_adoptions"]
    assert not [e for e in guard.events if e["event"] == "spawn_plane_rebased"]


def test_the_switch_never_settles_back_on_the_spawn_tread_it_passes():
    """00843/000022 (v3, 2026-11-02): the switch walked down past the tread the agent spawned on, stalled at its
    height, and the guard settled it there -- the spawn height again. While the switch traverses the guard stands
    aside, and no floor known when it set off can end its traversal."""
    policy, episode = setup_policy("bed", multifloor={"stair_source": "observed"})
    building, atlas, guard = policy.building, policy.mapping.atlas, policy.floor_guard
    building.terrain.plateau_area = atlas.plateau_observer = lambda pose: 6.0
    policy.plan(at(episode, 0, 1.0, 1.0, depth=3.0))
    step = 1
    for z in (0.18, 0.35, 0.52):
        policy.plan(at(episode, step, 1.0 + 0.25 * step, 1.0, z=z, depth=3.0)); step += 1
    for i in range(6):
        policy.plan(at(episode, step, 2.0 + 0.25 * i, 1.0, z=0.5, depth=3.0)); step += 1
        if atlas.active_id != 0:
            break
    switch = building.switch
    entry = switch.flight["entry"]
    building.can_leave_floor = lambda obs=None: True
    step = switch.flight["step"] + building.params.spawn_switch_min_floor_actions
    building.plan(at(episode, step, entry[0], entry[1], z=0.5, depth=3.0), policy.last_world, exhausted=True)
    assert switch.state == "traverse" and building.active["avoid_floors"] == [0, 1]
    step += 1
    x = entry[0]
    for z in (0.3, 0.1, 0.0):                                      # down to the spawn tread's height, and no further
        x -= 0.25
        policy.plan(at(episode, step, x, 1.0, z=z, depth=3.0)); step += 1
    for i in range(80):
        policy.plan(at(episode, step, x - 0.3 * (i % 3), 1.0 + 0.3 * ((i // 3) % 2), z=0.0, depth=3.0)); step += 1
        assert atlas.active_id == 1, "never back on the spawn's floor"
    assert not [e for e in guard.events if e["event"] in ("transition_settled_on_known_floor", "spawn_plane_rebased")
                and e["action"] > switch.flight["step"]]
    assert switch.state in ("traverse", "abandoned")


def test_a_switch_approach_given_up_before_the_flight_is_tried_once_more_and_no_more():
    """00814/000004 (2026-11-02): the switch's walk to the flight was given up in a cramped corner 4.6 m short of it,
    and the switch was spent with the agent never off its storey. An approach that never reached the flight is tried
    once more ``spawn_switch_min_floor_actions`` later; a second one given up spends the switch."""
    policy, episode = setup_policy("bed", multifloor={"stair_source": "observed"})
    building, atlas = policy.building, policy.mapping.atlas
    building.terrain.plateau_area = atlas.plateau_observer = lambda pose: 6.0
    policy.plan(at(episode, 0, 1.0, 1.0, depth=3.0))
    step = 1
    for z in (0.18, 0.35, 0.52):
        policy.plan(at(episode, step, 1.0 + 0.25 * step, 1.0, z=z, depth=3.0)); step += 1
    for i in range(6):
        policy.plan(at(episode, step, 2.0 + 0.25 * i, 1.0, z=0.5, depth=3.0)); step += 1
        if atlas.active_id != 0:
            break
    switch, params = building.switch, building.params
    building.can_leave_floor = lambda obs=None: True
    far = (switch.flight["entry"][0] + 3.0, switch.flight["entry"][1])
    first = switch.flight["step"] + params.spawn_switch_min_floor_actions
    building.plan(at(episode, first, *far, z=0.5, depth=3.0), policy.last_world, exhausted=True)
    assert switch.state == "approach" and switch.attempts == 1
    building._abandon(at(episode, first + 5, *far, z=0.5), "spawn switch: the flight's entry is unreachable")
    assert switch.state is None and switch.retry_at == first + 5 + params.spawn_switch_min_floor_actions
    early = at(episode, switch.retry_at - 1, *far, z=0.5, depth=3.0)
    assert building.plan(early, policy.last_world, exhausted=True) is None and switch.state is None
    building.plan(at(episode, switch.retry_at, *far, z=0.5, depth=3.0), policy.last_world, exhausted=True)
    assert switch.state == "approach" and switch.attempts == 2
    building._abandon(at(episode, switch.retry_at + 5, *far, z=0.5), "spawn switch: the flight's entry is unreachable")
    assert switch.state == "abandoned"
    assert building.plan(at(episode, switch.retry_at + 200, *far, z=0.5, depth=3.0), policy.last_world,
                         exhausted=True) is None and switch.state == "abandoned"
    events = [e["event"] for e in building.events if e["event"].startswith("spawn_switch")]
    assert events == ["spawn_switch_armed", "spawn_switch_started", "spawn_switch_deferred", "spawn_switch_started",
                      "spawn_switch_abandoned"]


def _step_terrain(policy, monkeypatch, cells, height, area=6.0):
    """The stair terrain reporting ``cells`` (world XY) as walkable support at ``height`` (one, or one per cell), and ``area`` m2 of plateau."""
    xy = np.asarray(cells, dtype=float).reshape(-1, 2)
    heights = np.broadcast_to(np.asarray(height, dtype=float), (len(xy),)).copy()
    monkeypatch.setattr(policy.building.terrain, "step_surfaces", lambda plane, band, min_offset_m=0.10: (xy, heights))
    monkeypatch.setattr(policy.building.terrain, "plateau_area", lambda pose: float(area))


def test_a_step_down_the_terrain_can_walk_is_free_and_never_a_drop(monkeypatch):
    """2026-10-31: a sunken floor 0.4 m below the plane (00835-q3zU7Yy5E5s/000006, the television there) was a drop."""
    policy, episode, world, _ = forbidden_stairs()
    policy.mapping._anchor = 0.0
    obs = at(episode, 0, 3.0, 3.0, depth=3.0)
    policy.floor_guard.observe(obs, world)
    drops = policy.floor_guard.drops[policy.mapping.floor_id].copy()
    assert drops.any()
    ys, xs = np.nonzero(drops)
    sunken = [world.grid_to_world(x, y) for x, y in zip(xs, ys)]
    _step_terrain(policy, monkeypatch, sunken, -0.4)
    confined = policy.floor_guard.observe(at(episode, 1, 3.0, 3.0, depth=3.0), world)
    assert policy.floor_guard.stats["step_drop_exempt"] == int(drops.sum())
    assert (confined.grid[drops] == world.values.free).all(), "walkable support is floor, not a drop"
    assert policy.floor_guard.goal_allowed(obs, world, sunken[0], "frontier")


def test_steps_free_occupied_cells_but_never_a_cell_the_body_touched(monkeypatch):
    policy, episode, world, _ = forbidden_stairs()
    policy.mapping._anchor = 0.0
    grid = world.grid.copy()
    gx, gy = world.world_to_grid(3.0, 4.0)
    grid[gy, gx] = grid[gy, gx + 1] = world.values.occupied            # a 0.2 m riser the obstacle band painted
    world = type(world)(grid, world.params, values=world.values)
    policy.mapping._touched[policy.mapping.floor_id] = {(gx + 1, gy): 0}
    _step_terrain(policy, monkeypatch, [world.grid_to_world(gx, gy), world.grid_to_world(gx + 1, gy)], 0.2)
    confined = policy.floor_guard.observe(at(episode, 0, 3.0, 3.0), world)
    assert confined.grid[gy, gx] == world.values.free and confined.grid[gy, gx + 1] == world.values.occupied


@pytest.mark.parametrize("walked_level", [True, False])
def test_standing_level_on_a_floor_sized_plane_a_step_below_re_measures_the_storey(monkeypatch, walked_level):
    policy, episode, world, _ = forbidden_stairs()
    monkeypatch.setattr(type(policy.floor_guard), "provisional", property(lambda self: not walked_level))
    atlas = policy.mapping.atlas
    assert atlas.elevation_m == pytest.approx(0.0)
    _step_terrain(policy, monkeypatch, [], -0.35)
    policy.floor_guard.observe(at(episode, 0, 2.8, 3.0, z=0.0), world)                    # the spawn, on the plane
    policy.floor_guard.drops[policy.mapping.floor_id] = np.ones(world.grid.shape, bool)   # measured from the step above
    for step in range(1, 4):
        policy.floor_guard.observe(at(episode, step, 3.0 + 0.1 * step, 3.0, z=-0.35), world)
    assert atlas.elevation_m == pytest.approx(-0.35) and policy.mapping._anchor == pytest.approx(-0.35)
    assert policy.floor_guard.stats["plane_remeasured"] == 1
    assert policy.floor_guard.events[-1]["event"] == "storey_plane_remeasured"
    assert not policy.floor_guard.drops[policy.mapping.floor_id].all(), "the old plane's drops went with it"
    # A storey walked level keeps its spawn plane; a spawn on a tread (provisional) moves it to the floor.
    assert policy.floor_guard.spawn_z == pytest.approx(0.0 if walked_level else -0.35)


def test_a_floor_just_more_than_level_below_the_spawn_is_re_measured_with_no_gap(monkeypatch):
    """00843/000083 (2026-11-02): spawned 0.149 m above its storey, the agent stepped down onto it and stood there
    neither level with the plane (more than ``stable_height_m`` off) nor off it by the old 0.15 m re-measure floor."""
    policy, episode, world, _ = forbidden_stairs()
    atlas = policy.mapping.atlas
    _step_terrain(policy, monkeypatch, [], -0.149)
    policy.floor_guard.observe(at(episode, 0, 2.8, 3.0, z=0.0), world)
    for step in range(1, 4):
        policy.floor_guard.observe(at(episode, step, 3.0 + 0.1 * step, 3.0, z=-0.149), world)
    assert atlas.elevation_m == pytest.approx(-0.149) and policy.floor_guard.stats["plane_remeasured"] == 1
    # Within stable_height_m the agent is level with the plane: nothing to re-measure.
    policy, episode, world, _ = forbidden_stairs()
    _step_terrain(policy, monkeypatch, [], -0.10)
    policy.floor_guard.observe(at(episode, 0, 2.8, 3.0, z=0.0), world)
    for step in range(1, 4):
        policy.floor_guard.observe(at(episode, step, 3.0 + 0.1 * step, 3.0, z=-0.10), world)
    assert policy.mapping.atlas.elevation_m == pytest.approx(0.0) and policy.floor_guard.stats["plane_remeasured"] == 0


def test_no_re_measure_on_a_small_plateau_or_beyond_the_atlas_departure(monkeypatch):
    policy, episode, world, _ = forbidden_stairs()
    atlas = policy.mapping.atlas
    _step_terrain(policy, monkeypatch, [], -0.35, area=1.0)              # a landing, not a floor
    for step in range(4):
        policy.floor_guard.observe(at(episode, step, 3.0, 3.0, z=-0.35), world)
    assert atlas.elevation_m == pytest.approx(0.0)
    assert atlas.remeasure(0.2) is True and atlas.elevation_m == pytest.approx(0.2)


@pytest.mark.parametrize("walked_level", [True, False])
def test_steps_are_the_storey_within_the_atlas_departure_from_a_storey_walked_level_and_the_whole_band_otherwise(
        monkeypatch, walked_level):
    """00835/000006 (2026-10-31): a frontier route down a flight's freed treads, 0.59 m off the plane, then a RETREAT.

    The guard offers no step the atlas would call a departure: from a storey
    walked level the retreat walks the agent back from past ``departure_m``.
    From a provisional spawn storey a departure is walked and rebased, so the
    whole band is the storey there.
    """
    policy, episode, world, _ = forbidden_stairs()
    guard, departure = policy.floor_guard, policy.mapping.atlas.params.departure_m
    monkeypatch.setattr(type(guard), "provisional", property(lambda self: not walked_level))
    assert departure < guard.step_band_m
    policy.mapping._anchor = 0.0
    obs = at(episode, 0, 3.0, 3.0, depth=3.0)
    guard.observe(obs, world)
    drops = guard.drops[policy.mapping.floor_id].copy()
    ys, xs = np.nonzero(drops)
    half = len(xs) // 2
    assert half > 0
    near, far = np.zeros_like(drops), np.zeros_like(drops)
    near[ys[:half], xs[:half]] = True                                  # 0.05 m inside the departure
    far[ys[half:], xs[half:]] = True                                   # 0.15 m past it, well inside the band
    heights = np.where(np.arange(len(xs)) < half, -(departure - 0.05), -(departure + 0.15))
    _step_terrain(policy, monkeypatch, [world.grid_to_world(x, y) for x, y in zip(xs, ys)], heights)
    confined = guard.observe(at(episode, 1, 3.0, 3.0, depth=3.0), world)
    assert guard.step_reach_m == pytest.approx(departure if walked_level else guard.step_band_m)
    assert (confined.grid[near] == world.values.free).all()
    far_goal = world.grid_to_world(xs[-1], ys[-1])
    if walked_level:
        assert (confined.grid[far] == world.values.occupied).all(), "a drop still: the retreat would undo the step"
        assert guard.stats["step_cells_withheld"] == int(far.sum())
        assert guard.stats["step_drop_exempt"] == int(near.sum())
        assert not guard.goal_allowed(obs, world, far_goal, "frontier")
    else:
        assert (confined.grid[far] == world.values.free).all(), "a spawn on a tread: the whole band"
        assert guard.stats["step_cells_withheld"] == 0
        assert guard.goal_allowed(obs, world, far_goal, "frontier")


@pytest.mark.parametrize("walked_level", [True, False])
def test_the_stair_mask_is_lifted_from_steps_by_the_same_bound_that_frees_them(monkeypatch, walked_level):
    """One cut for free cells, drop exemptions and stair exemptions: never a tread free on one count and masked on another."""
    policy, episode, world, _ = forbidden_stairs()
    guard, departure = policy.floor_guard, policy.mapping.atlas.params.departure_m
    monkeypatch.setattr(type(guard), "provisional", property(lambda self: not walked_level))
    policy.mapping._anchor = 0.0
    obs = at(episode, 0, 1.0, 3.0)                                      # far from the flight: no walked carve on it
    stairs = stair_peek_mask(policy, world, obs.pose, carve=False)
    ys, xs = np.nonzero(stairs)
    half = len(xs) // 2
    assert half > 0
    heights = np.where(np.arange(len(xs)) < half, departure - 0.15, departure + 0.15)
    _step_terrain(policy, monkeypatch, [world.grid_to_world(x, y) for x, y in zip(xs, ys)], heights)
    confined = guard.observe(obs, world)
    near, far = np.zeros_like(stairs), np.zeros_like(stairs)
    near[ys[:half], xs[:half]] = True
    far[ys[half:], xs[half:]] = True
    assert not (confined.grid[near] == world.values.occupied).any(), "steps the storey owns are no staircase"
    if walked_level:
        assert guard.stats["step_stair_exempt"] == int(near.sum())
        assert (confined.grid[far] == world.values.occupied).all(), "past the departure: still the flight"
    else:
        assert guard.stats["step_stair_exempt"] == int(stairs.sum())
        assert not (confined.grid[far] == world.values.occupied).any()


def test_the_closing_may_stand_where_the_guard_frees_and_nowhere_further():
    """The standoff tolerance is the guard's step reach: the band from a provisional spawn storey, ``departure_m`` after."""
    from sparx_agency.core.planning.object_search.closing.target_path import TargetApproachPath
    policy, _, _, _ = forbidden_stairs()
    guard, departure = policy.floor_guard, policy.mapping.atlas.params.departure_m
    assert not guard.provisional, "a ground-truth storey is a storey"
    assert TargetApproachPath(policy)._height_tolerance() == pytest.approx(guard.step_reach_m) == pytest.approx(departure)
    assert departure > 0.41, "00835/000006: the television's nearest view points stand 0.41 m down the treads"
