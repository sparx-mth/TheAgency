"""Bounded FALCON tests with real core geometry; no simulator/model dependency."""
from dataclasses import replace
from itertools import permutations
from types import SimpleNamespace
import math

import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.exploration.falcon.bursts import BurstMachine, DONE, EXPLORE, REASON, SELECT, VERIFY
from sparx_agency.core.planning.exploration.falcon.connectivity import Connectivity
from sparx_agency.core.planning.exploration.falcon.frontiers import CameraVisibility, FrontierMemory
from sparx_agency.core.planning.exploration.falcon.ordering import Deadline, PlanningDeadline, solve_order
from sparx_agency.core.planning.exploration.falcon.params import FalconParams, SOURCE
from sparx_agency.core.planning.exploration.falcon.planner import FalconPlanner
from sparx_agency.core.planning.exploration.falcon.travel import GridRoutes, segment_clear
from sparx_agency.core.planning.objnav.agent.headless_agent import HeadlessObjNavAgent
from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.falcon_regions import ObservedRegions
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_policy import RPTSearchPolicy, RPTSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import observation, setup_policy


def world_of(data):
    return OccupancyGrid2D(np.asarray(data, dtype=np.int8), OccupancyGrid2DParams(0.1, 0, 0),
                           values=OccupancyValues(free=0, occupied=100, unknown=-1))


def open_fixture():
    grid = np.full((60, 60), -1, np.int8)
    grid[15:45, 15:45] = 0
    world = world_of(grid)
    pose = AgentPose(3.05, 3.05, 0, 0)
    obs = SimpleNamespace(pose=pose, camera=PROTOCOL.camera(), step=0, action_spec=PROTOCOL.actions())
    return world, np.where(grid == 0, 1.0, np.inf), np.ones(grid.shape, bool), obs


def test_burst_and_global_budget_counts_all_action_types_and_no_refill():
    m = BurstMachine(FalconParams(burst_actions=4), 7)
    m.begin("provisional")
    m.charge(0)
    with pytest.raises(RuntimeError, match="refilled"):
        m.begin("renumbered room")
    m.interrupt()
    m.charge(1)  # verification turn
    m.resume()
    assert m.burst.actions == 2
    m.charge(2)  # failed movement is still one action
    m.charge(3)
    m.check_limits()
    assert m.phase == REASON and m.burst.status == "budget_exhausted"
    m.transition(SELECT, "reasoned partial revisit")
    m.begin("selected revisit")
    for step in range(4, 7):
        m.charge(step)
    m.check_limits()
    assert m.phase == DONE and m.actions == 7
    assert sum(m.allocation.values()) == 7
    with pytest.raises(RuntimeError, match="global"):
        m.charge(7)


def test_verification_cannot_refill_exploration_and_expiry_resumes_reasoning():
    m = BurstMachine(FalconParams(burst_actions=2), 500)
    m.begin("initial")
    m.charge(0)
    m.interrupt()
    m.charge(1)
    m.check_limits()
    assert m.phase == VERIFY and m.burst.status == "budget_exhausted"
    m.resume()
    assert m.phase == REASON and len(m.history) == 1


def test_room_entry_debounce_uses_distinct_actions_not_room_ids():
    m = BurstMachine(FalconParams(), 500)
    m.begin("initial")
    m.charge(0)
    assert not m.entered(True, 0)
    assert not m.entered(True, 0)
    assert not m.entered(False, 1)
    assert not m.entered(True, 2)
    assert m.entered(True, 3)
    assert m.burst.actions == 1


def test_frontiers_update_on_new_observations_and_exhaust():
    params = FalconParams()
    memory = FrontierMemory(params)
    grid = np.full((30, 30), -1, np.int8)
    grid[8:20, 8:20] = 0
    first = memory.update(grid).copy()
    grid[20:25, 8:20] = 0
    second = memory.update(grid).copy()
    assert memory.revision == 2 and not np.array_equal(first, second)
    memory.update(np.zeros_like(grid))
    assert not memory.in_scope(np.ones_like(grid, bool), 0.1, Deadline(1))


def test_visibility_has_real_fov_and_known_occlusion():
    world, _, scope, obs = open_fixture()
    camera = CameraVisibility(world, obs.camera, 0.0, 0.88, FalconParams())
    assert camera.in_frustum((30, 30), 0, np.array([[40, 30], [20, 30]])).tolist() == [True, False]
    gain = camera.gain((30, 30), 0, scope)
    assert gain > 0
    blocked = world.grid.copy()
    blocked[:, 35] = 100
    camera.world = world_of(blocked)
    assert camera.gain((30, 30), 0, scope) == 0


def test_motion_routes_never_cross_unknown_or_cut_corners():
    cost = np.full((5, 5), np.inf)
    cost[1, 1] = cost[2, 2] = 1
    routes = GridRoutes(cost, 0.1)
    assert not routes.path((1, 1), (2, 2), Deadline(1))
    assert not segment_clear(np.isfinite(cost), (1, 1), (2, 2))
    assert not segment_clear(np.isfinite(cost), (1, 1), (-1, 1))


def test_connectivity_splits_obstacles_and_reuses_unchanged_tiles():
    world, cost, scope, _ = open_fixture()
    connectivity = Connectivity(FalconParams(cell_size_m=3))
    connectivity.update(world, np.isfinite(cost), scope, Deadline(5))
    before = connectivity.updated_tiles
    connectivity.update(world, np.isfinite(cost), scope, Deadline(5))
    assert connectivity.updated_tiles == before
    grid = world.grid.copy()
    grid[:, 29:31] = 100
    changed = world_of(grid)
    connectivity.update(changed, grid == 0, scope, Deadline(5))
    a, b = connectivity.zone_at((20, 30)), connectivity.zone_at((40, 30))
    assert not math.isfinite(connectivity.graph_distances[a, b])


def test_atsp_sop_objectives_equal_bruteforce_and_preserve_precedence():
    rng = np.random.RandomState(7)
    cost = rng.uniform(0.1, 5, (6, 6))
    np.fill_diagonal(cost, 0)
    for constraints in ((), ((1, 3), (3, 4))):
        result = solve_order(cost, constraints)
        options = [(0,) + order for order in permutations(range(1, 6))]
        expected = min(sum(cost[a, b] for a, b in zip(order, order[1:])) for order in options
                       if all(order.index(a) < order.index(b) for a, b in constraints))
        assert result.status == "optimal" and result.cost == pytest.approx(expected)
        assert all(result.order.index(a) < result.order.index(b) for a, b in constraints)
    beam = solve_order(cost, exact_limit=2, beam_width=8)
    assert beam.status == "beam" and len(set(beam.order)) == 6


def test_deadline_is_distinct_from_simulated_actions():
    clock = iter([0.0, 2.0]).__next__
    with pytest.raises(PlanningDeadline):
        Deadline(1, clock).check()
    m = BurstMachine(FalconParams(), 500)
    assert m.actions == 0


def test_full_falcon_pipeline_has_cp_sop_refinement_and_routes_deterministically():
    world, cost, scope, obs = open_fixture()
    plans = []
    for _ in range(2):
        planner = FalconPlanner(FalconParams(planning_deadline_s=10, cell_size_m=3))
        planner.observe(world)
        result = planner.plan(world, cost, scope, obs, 0.88)
        assert result.status == "planned", result.diagnostics
        assert result.diagnostics["unknown_zones"] > 0
        assert result.diagnostics["sop_status"] in ("optimal", "beam")
        assert result.diagnostics["viewpoint_candidates"] > 1
        for route in result.routes:
            assert all(segment_clear(np.isfinite(cost), a, b) for a, b in zip(route, route[1:]))
        plans.append(result.viewpoints)
    assert plans[0] == plans[1]


def test_scope_and_visit_history_survive_room_splits_and_renumbering():
    world, cost, scope, obs = open_fixture()
    params = FalconParams(scope_radius_m=2)
    regions = ObservedRegions(params)
    mask = world.grid == 0
    room = SimpleNamespace(mask=mask, centroid=(3, 3))
    regions.start(world, obs.pose, {1: room}, 1)
    left, right = mask.copy(), mask.copy()
    left[:, 30:] = False
    right[:, :30] = False
    assert len(regions.history_for(left)) == len(regions.history_for(right)) == 1
    before = regions.envelope.copy()
    regions.scope(world, {99: room})
    assert np.array_equal(before, regions.envelope) and len(regions.visits) == 1


def test_policy_selection_provenance_and_no_silent_backend_fallback():
    p, episode = setup_policy("bed")
    p = RPTSearchPolicy(p.detector, p.llm_client, replace(p.settings, local_exploration="falcon"))
    config = p.configuration()
    assert config["falcon_running"] and config["falcon_source"] == SOURCE
    assert config["falcon_source"]["fallback_explorer"] is None
    with pytest.raises(ValueError, match="fallback"):
        RPTSettings(local_exploration="made-up")
    assert episode.max_steps == 500


def test_real_converter_emitted_action_is_accounted():
    p, episode = setup_policy("bed")
    agent = HeadlessObjNavAgent(p, gibson_label_mapper())
    agent.reset(episode)
    decision = agent.act(observation(episode, 0, depth=3))
    report = p.telemetry.report()
    assert report["actions"][decision.action.name] == 1
    assert sum(report["actions_by_phase"].values()) == 1
