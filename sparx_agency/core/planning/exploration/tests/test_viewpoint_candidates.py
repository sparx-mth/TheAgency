"""Candidate viewpoints on a reachable map and their price in actions (2026-11-11)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.exploration.viewpoint_candidates import (
    ActionPricing, geodesic_field, sample_viewpoints)

RES = 0.1
FREE, OCC, UNK = 0, 100, -1
VALUES = OccupancyValues(free=FREE, occupied=OCC, unknown=UNK)


def world(data):
    return OccupancyGrid2D(data, OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)


def hall_with_a_wall():
    """A 10 x 6 m hall, free and known; a wall across its middle leaves a 1 m gap at the top."""
    g = np.full((60, 100), FREE, np.int8)
    g[:, 50] = OCC
    g[50:60, 50] = FREE
    return world(g)


def cost_of(w):
    """A passable cost: 1 on free cells, inf elsewhere (no inflation, which these tests do not need)."""
    return np.where(w.grid == FREE, 1.0, np.inf)


def test_action_pricing_counts_steps_and_turns_and_refuses_a_zero_step():
    price = ActionPricing(0.25, math.radians(30))
    assert price.turns(0.0) == 0 and price.turns(math.radians(30)) == 1 and price.turns(-math.radians(31)) == 2
    assert price.travel_actions(2.0, 0.0) == pytest.approx(8 + 0.5 * 2.0)
    assert price.travel_actions(1.0, math.radians(90)) == pytest.approx(4 + 3 + 0.5)
    assert ActionPricing(0.25, math.radians(30), path_turn_per_m=0.0).travel_actions(1.0, 0.0) == 4.0
    for bad in (dict(forward_step_m=0.0, turn_rad=1.0), dict(forward_step_m=0.25, turn_rad=0.0),
                dict(forward_step_m=0.25, turn_rad=1.0, path_turn_per_m=-1.0)):
        with pytest.raises(ValueError):
            ActionPricing(**bad)


def test_the_geodesic_field_walks_round_a_wall_and_is_infinite_past_it_when_the_gap_is_closed():
    w = hall_with_a_wall()
    dist = geodesic_field(w, cost_of(w), (2.5, 0.5), 50.0)
    gx, gy = w.world_to_grid(2.5, 0.5)
    assert dist[gy, gx] == 0.0
    across = w.world_to_grid(7.5, 0.5)
    straight = 5.0
    assert dist[across[1], across[0]] > straight + 6.0, "the way round through the gap at the top is far longer"
    closed = hall_with_a_wall()
    closed.grid[50:60, 50] = OCC
    sealed = geodesic_field(closed, cost_of(closed), (2.5, 0.5), 50.0)
    assert not np.isfinite(sealed[across[1], across[0]])
    with pytest.raises(ValueError):
        geodesic_field(w, np.ones((5, 5)), (2.5, 0.5), 50.0)


def test_viewpoints_lie_on_a_lattice_within_reach_and_keep_their_distance_from_the_spots_already_used():
    w = hall_with_a_wall()
    dist = geodesic_field(w, cost_of(w), (2.5, 3.0), 50.0)
    points = sample_viewpoints(w, dist, (2.5, 3.0), 0.0, spacing_m=0.5, max_geodesic_m=2.0, min_distance_m=0.75)
    assert points, "a 2 m disc on a half-metre lattice holds dozens of candidates"
    assert all(0.75 <= p.geodesic_m <= 2.0 for p in points)
    cx, cy = w.grid_to_world(*w.world_to_grid(2.5, 3.0))          # the agent's cell centre phases the lattice
    assert all(round((p.xy[0] - cx) / 0.5, 6) % 1 == 0 and round((p.xy[1] - cy) / 0.5, 6) % 1 == 0 for p in points)
    assert points == sorted(points, key=lambda p: p.geodesic_m)
    ahead = [p for p in points if abs(p.xy[1] - cy) < 1e-6 and p.xy[0] > cx]
    assert ahead and all(p.heading_error_rad == pytest.approx(0.0, abs=0.1) for p in ahead)
    behind = [p for p in points if abs(p.xy[1] - cy) < 1e-6 and p.xy[0] < cx]
    assert behind and all(abs(p.heading_error_rad) == pytest.approx(math.pi, abs=0.1) for p in behind)
    kept = sample_viewpoints(w, dist, (2.5, 3.0), 0.0, 0.5, 2.0, min_distance_m=0.75,
                             avoid=[(3.5, 3.0)], avoid_radius_m=0.75)
    assert all(math.dist(p.xy, (3.5, 3.0)) > 0.75 for p in kept) and len(kept) < len(points)
    mask = np.zeros(w.grid.shape, bool)
    mask[:, :30] = True
    west = sample_viewpoints(w, dist, (2.5, 3.0), 0.0, 0.5, 2.0, within=mask)
    assert west and all(p.xy[0] < 3.0 for p in west)
    assert sample_viewpoints(w, np.full(w.grid.shape, np.inf), (2.5, 3.0), 0.0, 0.5, 2.0) == []
    with pytest.raises(ValueError):
        sample_viewpoints(w, np.ones((3, 3)), (2.5, 3.0), 0.0, 0.5, 2.0)
