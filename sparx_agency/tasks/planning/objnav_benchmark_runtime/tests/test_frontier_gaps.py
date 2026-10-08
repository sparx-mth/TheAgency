"""Gaps behind furniture: shallow or thin unknown behind a frontier is demoted now and settled for later."""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.exploration_fallback import split_exits
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.frontier_gaps import (
    GapSettings, TARGET_FOOTPRINT_M, probe_gap, target_min_dim_m, unknown_heading)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.opening_nodes import detect_openings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_opening_nodes import opening_policy, openings_of, refresh
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_room_search_loop import FREE, IN_A, OCC, RES, UNK, VALUES, obs_at

SETTINGS = GapSettings()


def room(width_m=8.0, height_m=6.0):
    """An all-free room with occupied walls, 0.1 m cells, origin at (0, 0)."""
    h, w = int(height_m / RES), int(width_m / RES)
    g = np.full((h, w), FREE, np.int8)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = OCC
    return g


def world_of(g):
    return OccupancyGrid2D(g, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)


def cell(x, y):
    return int(math.floor(x / RES)), int(math.floor(y / RES))


# -- the probe --------------------------------------------------------------------------------------------
def test_an_unobserved_band_along_a_wall_is_shallow_and_a_gap():
    """The Wiconisco opening: a 2.8 m frontier whose unknown behind it is a 0.4 m band in front of the wall."""
    g = room()
    g[1:5, 20:48] = UNK                                   # y 0.1-0.5: a 0.4 m band along the south wall, 2.8 m long
    world = world_of(g)
    frontier = cell(3.4, 0.5)                             # the free cell just north of the band
    heading = unknown_heading(world, frontier)
    assert heading is not None and abs(heading + math.pi / 2) < 0.3, "the unknown lies south"
    probe = probe_gap(world, frontier, heading, SETTINGS)
    assert probe is not None and probe.deep_rays == 0
    assert probe.min_depth_m <= 0.45 and probe.depth_m < SETTINGS.depth_m, "0.4 m straight on, 0.4 / cos 45 on the oblique rays"
    assert probe.width_m <= 0.5 and probe.is_gap(SETTINGS.depth_m)
    assert all(g[gy, gx] == UNK for gx, gy in probe.cells) and len(probe.cells) >= 4


def test_a_slit_behind_a_sofa_entered_from_its_end_is_thin_and_a_gap():
    g = room()
    g[10:30, 1:5] = UNK                                   # x 0.1-0.5, y 1.0-3.0: a 0.4 m slit, 2 m long ...
    g[10:30, 5:13] = OCC                                  # ... behind a sofa (x 0.5-1.3) against the west wall
    world = world_of(g)
    frontier = cell(0.3, 0.9)                             # the free cell at the slit's south end
    heading = unknown_heading(world, frontier)
    probe = probe_gap(world, frontier, heading, SETTINGS)
    assert probe is not None and probe.deep_rays == 0
    assert probe.depth_m > SETTINGS.depth_m, "along its axis the slit runs 2 m"
    assert probe.width_m < SETTINGS.depth_m and probe.is_gap(SETTINGS.depth_m), "... but it is thin"


def test_a_doorway_onto_unseen_space_is_deep_and_not_a_gap():
    g = room(10.0, 8.0)
    g[30:50, 60:100] = UNK                                # x 6-10, y 3-5: a whole room not seen yet ...
    g[:, 60] = OCC
    g[36:44, 60] = FREE                                   # ... through a 0.8 m doorway in the wall at x = 6.0
    world = world_of(g)
    frontier = cell(5.95, 4.0)
    heading = unknown_heading(world, frontier)
    assert heading is not None and abs(heading) < 0.4, "the unknown lies east"
    probe = probe_gap(world, frontier, heading, SETTINGS)
    assert probe is not None and probe.deep_rays >= 1 and not probe.is_gap(SETTINGS.depth_m)
    assert not probe.is_gap(TARGET_FOOTPRINT_M["bed"][0]), "deep for a bed too"


def test_no_unknown_behind_the_cell_gives_no_probe():
    world = world_of(room())
    assert unknown_heading(world, cell(4.0, 3.0)) is None
    assert probe_gap(world, cell(4.0, 3.0), 0.0, SETTINGS) is None


# -- target scale --------------------------------------------------------------------------------------------
def alcove(depth_m):
    """A niche ``depth_m`` deep and 1.5 m wide off the north wall, unseen."""
    g = room(8.0, 6.0 + depth_m)
    rows = int(depth_m / RES)
    g[59:59 + rows, 20:35] = UNK                          # x 2-3.5, y 5.9-5.9+depth: the niche
    g[59:59 + rows, 19] = g[59:59 + rows, 35] = OCC       # its side walls
    g[59 + rows, 19:36] = OCC                             # its back wall
    world = world_of(g)
    return world, cell(2.75, 5.8)


def test_an_alcove_a_bed_cannot_stand_in_is_a_gap_for_a_bed_search_only():
    world, frontier = alcove(0.8)
    heading = unknown_heading(world, frontier)
    probe = probe_gap(world, frontier, heading, SETTINGS)
    assert probe is not None and probe.deep_rays == 0
    assert 0.7 <= probe.depth_m < 0.95 and probe.width_m >= 0.8
    assert not probe.is_gap(SETTINGS.depth_m), "0.8 m deep: not a gap by the plain rule"
    assert probe.is_gap(TARGET_FOOTPRINT_M["bed"][0]), "... but nothing 0.9 m across stands in it"
    assert not probe.is_gap(TARGET_FOOTPRINT_M["potted plant"][0])


def test_target_scale_reads_the_target_labels_and_their_aliases():
    class Target:
        query, category, accept = "couch", "couch", ("couch", "sofa")
    assert target_min_dim_m(Target()) == TARGET_FOOTPRINT_M["sofa"][0]
    assert target_min_dim_m("tv") == TARGET_FOOTPRINT_M["television"][0]
    assert target_min_dim_m("bed") == 0.9 and target_min_dim_m("fire hydrant") is None and target_min_dim_m(None) is None


@pytest.mark.parametrize("kwargs", [{"depth_m": 0.0}, {"depth_m": 3.0, "probe_m": 2.5}, {"half_cone_deg": 120.0},
                                    {"rays": 0}, {"enabled": "yes"}])
def test_gap_settings_are_validated(kwargs):
    with pytest.raises(ValueError):
        GapSettings(**kwargs)


# -- the policy: demoted now, settled for later, excluded from the openings ---------------------------------------
def test_the_probe_demotes_a_gap_from_the_inventory_and_settles_it_in_the_ledger():
    """Room B's small unknown patch (0.5 x 0.6 m) is a gap: the fallback demotes its frontier this action with the
    prober's reason, the ledger settles it, and the next inventory has no frontier there at all."""
    policy, episode, world, rooms, _ = opening_policy()
    policy.sight.settings = replace(policy.sight.settings, enabled=True)
    obs = obs_at(episode, 0, IN_A)
    inventory = policy.graph.frontier_inventory
    patch = [g for g in inventory.goals if math.dist(g.xy, (7.0, 5.5)) < 1.0]
    assert patch, "the patch near the door is a frontier goal before the probe"
    reasons = policy.frontier_gaps.probe(obs, world, inventory, policy.target)
    assert any(tuple(g.cell) in reasons for g in patch), "... and a gap to the prober"
    assert policy.frontier_gaps.stats["gaps"] >= 1 and policy.frontier_gaps.stats["cells_settled"] > 0
    west = [g for g in inventory.goals if g.xy[0] < 1.5]
    assert west and not any(tuple(g.cell) in reasons for g in west), "room A's unexplored west end runs 1 m deep: an exit"
    # This action: the fallback and the openings read the verdict.
    exits, demoted = split_exits(list(inventory.goals), [], world.resolution, policy.fallback.settings,
                                 gaps=reasons, type_rule=False)
    assert any(why.startswith("gap:") for _, why in demoted) and all(g not in exits for g in patch)
    openings = detect_openings(policy, obs, world, policy.loop.settings.openings, policy.openings)
    assert not any(math.dist(o.xy, (7.0, 5.5)) < 1.0 for o in openings), "no opening on the patch"
    assert policy.openings.gaps_refused >= 1 and policy.openings.diagnostics()["last_gaps"]
    # The next action: settled, the patch is no frontier -- not for the oracle's room facts either.
    refresh(policy, world, IN_A)
    later = policy.graph.frontier_inventory
    assert not any(math.dist(g.xy, (7.0, 5.5)) < 1.0 for g in later.goals)
    assert policy.sight.diagnostics()["floors"]["0"]["gap_cells"] > 0
    assert policy.episode_info()["frontier_gaps"]["stats"]["gaps"] >= 1


def test_a_disabled_prober_judges_nothing():
    policy, episode, world, rooms, _ = opening_policy()
    policy.gap_settings = policy.frontier_gaps.settings = GapSettings(enabled=False)
    reasons = policy.frontier_gaps.probe(obs_at(episode, 0, IN_A), world, policy.graph.frontier_inventory, policy.target)
    assert reasons == {} and policy.frontier_gaps.stats["gaps"] == 0
    assert policy.configuration()["frontier_gaps"]["rule"].startswith("none")
    assert policy.frontier_gaps.diagnostics()["settings"]["enabled"] is False
