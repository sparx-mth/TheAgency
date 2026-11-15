"""The map change ledger: every changed planning cell attributed to observation or the guard, in or out of view."""
from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.core.planning.object_search.world_model.map_ledger import JUMP_CELLS, MapChangeLedger, field_of_view
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import observation, setup_policy

VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)
RES = 0.1


def grid(fill=VALUES.unknown):
    return OccupancyGrid2D(np.full((60, 80), fill, np.int8), OccupancyGrid2DParams(RES, 0.0, 0.0, "world"), values=VALUES)


def test_the_field_of_view_is_the_horizontal_frustum_to_the_camera_range_plus_the_body():
    policy, episode = setup_policy("bed")
    world = grid()
    pose = AgentPose(2.0, 3.0, 0.0, 0.0)                                   # facing +x
    fov = field_of_view(world, pose, episode.camera)
    inside = lambda x, y: fov[world.world_to_grid(x, y)[1], world.world_to_grid(x, y)[0]]
    assert inside(4.0, 3.0) and inside(4.0, 4.4), "ahead, and inside the 79 degree frame"
    assert inside(6.9, 3.0) and not inside(7.5, 3.0), "to the camera's 5 m range and not beyond"
    assert not inside(0.5, 3.0) and not inside(2.0, 5.0), "behind, and beside"
    assert inside(1.85, 3.0), "the body's own cells"


def test_changes_are_attributed_to_observation_or_the_guard_and_counted_in_and_out_of_view():
    policy, episode = setup_policy("bed")
    ledger = MapChangeLedger(jump_cells=20)
    pose = AgentPose(2.0, 3.0, 0.0, 0.0)
    obs = replace(observation(episode, 0), pose=pose)
    observed, planning = grid(), grid()
    ledger.record(obs, 0, observed, planning, plane_m=0.0, plane_before_m=0.0)
    assert ledger.last["changed"] == 0 and ledger.stats["actions"] == 1
    # An observation ahead: the export and the planning world change the same cells, in view.
    observed2, planning2 = grid(), grid()
    observed2.grid[28:32, 30:40] = VALUES.free
    planning2.grid[28:32, 30:40] = VALUES.free
    obs1 = replace(observation(episode, 1), pose=pose)
    ledger.record(obs1, 0, observed2, planning2, plane_m=0.0, plane_before_m=0.0)
    assert ledger.last["changed"] == 40 == ledger.last["in_fov"] == ledger.last["observation"]
    assert ledger.last["outside_fov"] == 0 and ledger.last["guard"] == 0 and not ledger.events
    # The guard masks 30 cells BEHIND the camera with the export unchanged, as the plane moves: a jump, out of view.
    mask = np.zeros(planning2.grid.shape, bool)
    mask[10:13, 2:12] = True
    planning3 = grid(); planning3.grid[:] = planning2.grid; planning3.grid[mask] = VALUES.occupied
    obs2 = replace(observation(episode, 2), pose=pose)
    ledger.record(obs2, 0, observed2, planning3, mask=mask, plane_m=-0.2, plane_before_m=0.0)
    last = ledger.last
    assert last["changed"] == 30 == last["outside_fov"] == last["guard"] == last["guard_detail"]["mask_on"]
    assert last["outside_fov_detail"]["guard"] == 30 and last["plane_moved"] and last["outside_fov_bbox_m"] == [0.2, 1.0, 1.2, 1.3]
    assert ledger.stats["jumps"] == 1 and ledger.events[-1]["event"] == "map_changed_out_of_view"
    assert ledger.events[-1]["plane_before_m"] == 0.0 and ledger.events[-1]["plane_m"] == -0.2
    assert ledger.max_outside_fov == 30
    # Releasing them again: mask_off, out of view, a second jump.
    obs3 = replace(observation(episode, 3), pose=pose)
    ledger.record(obs3, 0, observed2, planning2, mask=None, plane_m=0.0, plane_before_m=-0.2)
    assert ledger.last["guard_detail"]["mask_off"] == 30 and ledger.stats["jumps"] == 2
    diagnostics = ledger.diagnostics()
    assert diagnostics["stats"]["guard_outside_fov"] == 60 and len(diagnostics["events"]) == 2
    assert JUMP_CELLS == 50


def test_the_policy_records_every_action_and_the_record_carries_the_summary():
    policy, episode = setup_policy("bed")
    for step in range(3):
        policy.plan(observation(episode, step))
    assert policy.map_ledger.stats["actions"] == 3 and policy.map_ledger.last["action"] == 2
    assert policy.episode_info()["map_changes"]["stats"]["actions"] == 3
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.visualization import method_snapshot
    assert method_snapshot(policy)["map_changes"]["changed"] >= 0
