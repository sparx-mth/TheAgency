"""Regressions for the sight ledger (2026-10-05): looked-through unknown, pockets, and the rooms they finish.

Built on the two-room world of ``test_room_search_loop``: room A (west) has
an unexplored west end, room B (east) a small enclosed unknown patch near
the door and an unexplored east end. The Hanson recording of 2026-10-05
walked to both ends of a balcony seen whole from its threshold (the
railing read as two exits), back to an island of unknown between the spawn
point and the bed (O4, 31 actions), and behind a bed (O9, 42 actions); and
it offered the balcony again at 0.25, "never entered", 120 actions after
standing at its far end.
"""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from sparx_agency.core.mapping.topology.room_registry import TrackedRoom
from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams
from sparx_agency.core.planning.exploration.object_search_supervisor import EXHAUSTED, RoomFacts, SELECT, TRANSIT
from sparx_agency.core.planning.exploration.room_search_policy import RoomOption
from sparx_agency.core.planning.exploration.view_gain import UnknownView, cone_from_camera
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_scans import FRAGMENT, SEEN_THROUGH
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.sightlines import (
    SightLedger, SightSettings, cone_seen, enclosed_pockets, floor_band)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import observation
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_opening_nodes import openings_of, refresh
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import (
    FREE, IN_A, IN_B, OCC, RES, UNK, VALUES, loop_policy, obs_at, two_room_world)

CAMERA = PROTOCOL.camera()
HALF_VFOV = math.atan(0.5 * CAMERA.intrinsics.height / CAMERA.intrinsics.fy)
HALF_HFOV = math.atan(0.5 * CAMERA.intrinsics.width / CAMERA.intrinsics.fx)


def pitched(episode, step, xy, yaw=0.0, pitch=0.0):
    return replace(observation(episode, step), pose=AgentPose(xy[0], xy[1], 0.0, yaw, pitch))


# -- geometry -----------------------------------------------------------------------
def test_settings_reject_nonsense():
    for kwargs in (dict(min_looks=0), dict(far_m=0.0), dict(pocket_max_m2=-1.0), dict(enabled="yes"), dict(near_margin_m=-0.1)):
        with pytest.raises(ValueError):
            SightSettings(**kwargs)


def test_the_floor_band_is_the_blind_radius_to_the_depth_range_when_level_and_the_feet_when_pitched_down():
    near, far = floor_band(CAMERA.height_m, 0.0, HALF_VFOV, CAMERA.min_depth_m, CAMERA.max_depth_m)
    assert near == pytest.approx(0.88 / math.tan(HALF_VFOV), abs=0.02), "a level camera 0.88 m up sees the floor from ~1.4 m"
    assert far == pytest.approx(CAMERA.max_depth_m)
    near, far = floor_band(CAMERA.height_m, math.radians(30.0), HALF_VFOV, CAMERA.min_depth_m, CAMERA.max_depth_m)
    assert near == pytest.approx(0.88 / math.tan(math.radians(30.0) + HALF_VFOV), abs=0.02), "LOOK_DOWN: the floor from ~0.5 m"
    assert far == pytest.approx((CAMERA.max_depth_m - 0.88 * 0.5) / math.cos(math.radians(30.0))), (
        "the depth clip is along the tilted axis: a little farther along the ground")
    assert floor_band(CAMERA.height_m, math.radians(-60.0), HALF_VFOV, CAMERA.min_depth_m, CAMERA.max_depth_m) is None, "sky"


def test_cone_seen_reaches_free_cells_in_the_cone_and_stops_at_unknown_and_walls():
    world, _ = two_room_world()
    seen = cone_seen(world, IN_A, 0.0, HALF_HFOV, 5.0)
    assert seen[world.world_to_grid(4.0, 3.0)[::-1]], "straight ahead, free"
    assert not seen[world.world_to_grid(2.0, 3.0)[::-1]], "behind the camera"
    assert not seen[world.world_to_grid(3.0, 5.0)[::-1]], "outside the cone"
    assert not seen[world.world_to_grid(8.5, 3.0)[::-1]], "beyond the depth range"
    seen = cone_seen(world, IN_A, math.pi, HALF_HFOV, 5.0)
    assert seen[world.world_to_grid(1.5, 3.0)[::-1]] and not seen[world.world_to_grid(0.5, 3.0)[::-1]], "unknown ends a ray"


def test_a_small_enclosed_unknown_region_is_a_pocket_the_open_ends_and_the_edge_of_the_map_are_not():
    world, _ = two_room_world()
    pockets = enclosed_pockets(world, 3.0)
    assert pockets[world.world_to_grid(7.0, 5.5)[::-1]], "room B's patch by the door: 0.3 m2, walled by known cells"
    assert not pockets[world.world_to_grid(0.5, 3.0)[::-1]], "room A's west end runs to the edge of the map"
    assert not pockets[world.world_to_grid(11.4, 3.0)[::-1]], "so does room B's east end"
    assert not enclosed_pockets(world, 0.2).any(), "under the area bound nothing is a pocket"
    g = world.grid.copy()
    g[20:50, 70:100] = UNK                                  # a 9 m2 enclosed region: too big to write off
    assert not enclosed_pockets(OccupancyGrid2D(g, world.params, values=VALUES), 3.0)[25, 75]


# -- the look -------------------------------------------------------------------------
def test_unknown_looked_at_from_two_poses_within_range_without_a_return_is_resolved_and_the_blind_radius_is_not():
    policy, episode, world, rooms, _ = loop_policy(visit="scan")
    sight = policy.sight
    west_edge = world.world_to_grid(1.0, 3.0)[::-1]        # the first unknown cell west of room A's known floor
    sight.observe(pitched(episode, 0, (3.0, 3.0), yaw=math.pi), world)
    assert not sight.looked_through(world)[west_edge], "one frame is not a look"
    sight.observe(pitched(episode, 1, (3.0, 3.0), yaw=math.pi), world)
    assert not sight.looked_through(world)[west_edge], "the same pose again counts once"
    sight.observe(pitched(episode, 2, (2.6, 3.0), yaw=math.pi), world)
    assert sight.looked_through(world)[west_edge], "a second pose: looked through"
    assert sight.resolved(world)[west_edge]
    resolved = sight.resolved(world)
    assert not resolved[world.grid != UNK].any(), "only unknown cells are ever resolved"
    far = world.world_to_grid(0.3, 3.0)[::-1]
    assert not resolved[far], "the band is a few cells deep, not the whole unknown"
    # From 1.2 m the edge is inside the camera's blind radius: passed over, never marked.
    policy, episode, world, rooms, _ = loop_policy(visit="scan")
    for step, x in enumerate((2.3, 2.2)):
        policy.sight.observe(pitched(episode, step, (x, 3.0), yaw=math.pi), world)
    assert not policy.sight.looked_through(world).any()
    # Looking the other way marks nothing at the west end.
    policy, episode, world, rooms, _ = loop_policy(visit="scan")
    for step, x in enumerate((3.0, 2.6)):
        policy.sight.observe(pitched(episode, step, (x, 3.0), yaw=0.0), world)
    assert not policy.sight.looked_through(world)[west_edge]


def narrow_west_gap(world):
    """The two-room world with room A's west end walled down to a 3.3 m gap (y 1.4-4.6), inside one cone from 2 m."""
    g = world.grid.copy()
    g[1:14, 1:11] = OCC
    g[47:59, 1:11] = OCC
    return OccupancyGrid2D(g, world.params, values=VALUES)


def test_the_overlay_writes_resolved_cells_occupied_for_the_frontier_logic_and_leaves_the_map_alone():
    policy, episode, world, rooms, _ = loop_policy(visit="scan")
    world = narrow_west_gap(world)
    for step, y in enumerate((3.0, 2.9)):                   # two poses (different bins), both facing the gap from 2 m
        policy.sight.observe(pitched(episode, step, (3.0, y), yaw=math.pi), world)
    overlay = policy.sight.overlay(world)
    assert overlay is not world and (overlay.grid == OCC).sum() > (world.grid == OCC).sum()
    assert (world.grid == UNK)[world.world_to_grid(1.0, 3.0)[::-1]], "the real map is untouched"
    assert policy.graph.resolved_provider is not None, "the floor's scene graph reads the ledger"
    frontier_world = policy.graph.frontier_world(world)
    assert (frontier_world.grid == OCC)[world.world_to_grid(1.0, 3.0)[::-1]]
    refresh(policy, world, IN_A, yaw=math.pi)
    goals = policy.graph.frontier_inventory.goals
    assert not any(g.xy[0] < 2.0 for g in goals), "room A's west end is no frontier goal any more"
    assert policy.graph.facts[0].frontier_clusters == 0
    assert not [o for o in openings_of(policy, episode, world) if o.room_pid == 0], "... and no opening"
    assert policy.graph.facts[1].frontier_clusters == 1, "room B's east end still is (its patch is a pocket)"
    # Without the ledger the same map still shows the gap as a goal and an opening.
    policy.sight.settings = replace(policy.sight.settings, enabled=False)
    refresh(policy, world, IN_A, yaw=math.pi)
    assert any(g.xy[0] < 2.0 for g in policy.graph.frontier_inventory.goals)
    assert [o for o in openings_of(policy, episode, world) if o.room_pid == 0]


def test_the_glance_view_counts_no_gain_through_resolved_unknown():
    policy, episode, world, rooms, _ = loop_policy(visit="scan")
    cone = cone_from_camera(CAMERA.intrinsics.width, CAMERA.intrinsics.fx, 5.0, 0.6)
    before = UnknownView(world, cone).unknown_ahead((3.0, 3.0), math.pi).sum()
    for step, x in enumerate((3.0, 2.6)):
        policy.sight.observe(pitched(episode, step, (x, 3.0), yaw=math.pi), world)
    after = UnknownView(world, cone, resolved=policy.sight.resolved(world)).unknown_ahead((3.0, 3.0), math.pi).sum()
    assert before > 0 and after < before * 0.2, "the west end is a railing now: the rays stop at it"
    with pytest.raises(ValueError):
        UnknownView(world, cone, resolved=np.zeros((3, 3), bool))


def test_the_ledger_is_off_by_a_switch_and_records_poses_either_way():
    policy, episode, world, rooms, _ = loop_policy(visit="scan")
    policy.sight.settings = replace(policy.sight.settings, enabled=False)
    for step, x in enumerate((3.0, 2.6)):
        policy.sight.observe(pitched(episode, step, (x, 3.0), yaw=math.pi), world)
    assert policy.sight.resolved(world) is None and policy.sight.overlay(world) is world
    assert policy.graph.frontier_world(world) is world
    assert len(policy.sight.poses()) == 2 and policy.sight.stood_in(world, rooms[0].mask)
    assert not policy.sight.stood_in(world, rooms[1].mask)
    assert policy.sight.diagnostics()["floors"]["0"]["poses"] == 2


def test_the_policy_feeds_the_ledger_every_action_and_the_record_carries_it():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import setup_policy
    policy, episode = setup_policy(label="desk")                  # a context class: the search runs, no takeover
    policy.plan(observation(episode, 0, depth=3.0))
    policy.plan(observation(episode, 1, depth=3.0))
    assert len(policy.sight.poses()) == 2 and policy.sight.stats["frames"] == 2
    info = policy.episode_info()["sight"]
    assert info["floors"]["0"]["poses"] == 2 and info["stats"]["frames"] == 2
    assert policy.configuration()["sight"]["enabled"] is True and "pockets" in policy.configuration()["sight"]["rule"]
    assert policy.graph.resolved_provider is not None


# -- the rooms the ledger finishes ---------------------------------------------------------
def narrow_hall_world():
    """A 1.2 m wide corridor (room 0) into the middle of a 5 x 5 m room (room 1), both fully observed; nothing unknown."""
    h, w = 90, 120
    g = np.full((h, w), OCC, np.int8)
    g[30:80, 60:110] = FREE                 # the room: x 6-11, y 3-8
    g[49:61, 10:60] = FREE                  # the corridor: x 1-6, y 4.9-6.1
    world = OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)
    labels = np.zeros((h, w), np.int32)
    labels[49:61, 10:60] = 1
    labels[30:80, 60:110] = 2
    return world, labels


def install(policy, world, labels):
    rooms = {}
    for pid, label in ((0, 1), (1, 2)):
        mask = labels == label
        ys, xs = np.nonzero(mask)
        centroid = world.grid_to_world(int(round(xs.mean())), int(round(ys.mean())))
        rooms[pid] = TrackedRoom(id=pid, mask=mask, n_cells=int(mask.sum()), centroid=(float(centroid[0]), float(centroid[1])))
    g = policy.graph
    g.registry.rooms = rooms
    g.labels = labels
    g.probs = {0: 0.5, 1: 0.5}
    g.options = [RoomOption(room_id=pid, prob=0.5, xy=room.centroid, label="R%d" % pid) for pid, room in rooms.items()]
    g.facts = {pid: RoomFacts(pid, 0, 0.0, room.n_cells) for pid, room in rooms.items()}
    g.doors = []
    return rooms


def test_a_narrow_room_walked_through_with_no_frontier_left_is_finished_a_wide_one_merely_stepped_into_is_not():
    policy, episode, _, _, _ = loop_policy(visit="scan")
    world, labels = narrow_hall_world()
    rooms = install(policy, world, labels)
    ledger = policy.scans
    assert ledger.status(world, 0, rooms[0], frontier=0) is None, "nobody has been in the corridor yet"
    policy.sight.observe(pitched(episode, 0, (3.0, 5.5), yaw=0.0), world)       # standing in the corridor
    assert ledger.status(world, 0, rooms[0], frontier=1) is None, "a frontier left: not finished"
    assert ledger.status(world, 0, rooms[0], frontier=0) == SEEN_THROUGH, "walked through, nothing left to look at"
    assert ledger.status(world, 0, rooms[0]) == SEEN_THROUGH, "sticky"
    # One step into the wide room facing along its west wall: in it, but a sixth of it in view.
    policy.sight.observe(pitched(episode, 1, (6.5, 5.5), yaw=math.pi / 2), world)
    assert ledger.status(world, 1, rooms[1], frontier=0) is None, "a wide room stepped into is not walked through"
    # Facing into it from the same spot, most of it is in one view: finished like a room a scan saw.
    policy.sight.observe(pitched(episode, 2, (6.5, 5.5), yaw=0.0), world)
    assert ledger.status(world, 1, rooms[1], frontier=0) == SEEN_THROUGH
    # The walk-through half of the rule is a knob.
    policy, episode, _, _, _ = loop_policy(visit="scan")
    rooms = install(policy, world, labels)
    policy.scans.walkthrough_clearance_m = 0.0
    policy.sight.observe(pitched(episode, 0, (3.0, 5.5), yaw=math.pi / 2), world)   # in the corridor, facing its wall
    assert policy.scans.status(world, 0, rooms[0], frontier=0) is None


def test_a_room_one_pose_saw_more_than_half_of_through_its_door_is_finished_without_a_step_inside():
    policy, episode, _, _, _ = loop_policy(visit="scan")
    world, labels = narrow_hall_world()
    rooms = install(policy, world, labels)
    for step, x in enumerate((2.0, 2.5)):
        policy.sight.observe(pitched(episode, step, (x, 5.5), yaw=math.pi), world)    # looking away, down the corridor
    assert policy.scans.status(world, 1, rooms[1], frontier=0) is None
    policy.sight.observe(pitched(episode, 2, (5.8, 5.5), yaw=0.0), world)            # at the door, looking in
    assert policy.scans.status(world, 1, rooms[1], frontier=0) == SEEN_THROUGH
    diagnostics = policy.scans.diagnostics()
    assert "f0/r1:seen_through" in diagnostics["finished"]


def test_a_small_doorless_room_with_no_frontier_is_a_fragment_a_doored_or_bigger_one_is_not():
    policy, episode, _, _, _ = loop_policy(visit="scan")
    h, w = 60, 60
    g = np.full((h, w), OCC, np.int8)
    g[10:50, 10:50] = FREE
    world = OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)
    strip = np.zeros((h, w), bool)
    strip[10:50, 10:15] = True              # 4 m x 0.5 m: the strip behind a bed, 2 m2
    room = TrackedRoom(id=3, mask=strip, n_cells=int(strip.sum()), centroid=(1.25, 3.0))
    ledger = policy.scans
    assert ledger.status(world, 3, room, frontier=1) is None
    assert ledger.status(world, 3, room, frontier=0, doored=True) is None, "a door makes it a room"
    assert ledger.status(world, 3, room, frontier=0, doored=False) == FRAGMENT
    big = np.zeros((h, w), bool)
    big[10:50, 10:30] = True                # 8 m2
    room = TrackedRoom(id=4, mask=big, n_cells=int(big.sum()), centroid=(2.0, 3.0))
    assert ledger.status(world, 4, room, frontier=0, doored=False) is None
    policy.scans.fragment_max_m2 = 0.0
    assert ledger.status(world, 5, replace(room, id=5, mask=strip, n_cells=int(strip.sum())), frontier=0) is None, "disabled"


def test_a_finished_room_is_withheld_at_the_loop_point_and_released_on_the_way_to_it():
    """The balcony: stood at its far end at action 48, offered again at 206 as 'never entered'."""
    policy, episode, _, _, _ = loop_policy(order=(0, 1), visit="scan")
    world, labels = narrow_hall_world()
    rooms = install(policy, world, labels)
    policy.sight.observe(pitched(episode, 0, (3.0, 5.5), yaw=0.0), world)          # the corridor, walked through
    policy.loop._exclusions(obs_at(episode, 1, (8.0, 5.0)), world)
    assert policy.loop._excluded == {0: "scanned:%s" % SEEN_THROUGH}
    # A room chosen while unfinished and finished by what the walk showed is released before it is entered.
    policy, episode, _, _, _ = loop_policy(order=(0, 1), visit="scan")
    world, labels = narrow_hall_world()
    rooms = install(policy, world, labels)
    policy.graph.facts[0] = RoomFacts(0, 1, 0.0, rooms[0].n_cells)                 # a frontier left: a node
    policy.loop.plan(obs_at(episode, 0, (8.0, 5.0)), world)
    assert policy.supervisor.state == TRANSIT and policy.supervisor.room_id == 0
    policy.graph.facts[0] = RoomFacts(0, 0, 0.0, rooms[0].n_cells)                 # ... resolved as the agent walked
    policy.sight.observe(pitched(episode, 1, (5.5, 5.5), yaw=math.pi), world)      # and the agent has stepped into it
    policy.loop.plan(obs_at(episode, 1, (5.5, 5.5), yaw=math.pi), world)
    assert policy.supervisor.history[-1][:2] == (0, EXHAUSTED)
    assert policy.loop.stats["finished_in_transit"] == 1
    assert [e for e in policy.loop.events if e["event"] == "finished_in_transit"][0]["how"] == SEEN_THROUGH
    assert policy.supervisor.room_id == 1, "the order went on to the room"
