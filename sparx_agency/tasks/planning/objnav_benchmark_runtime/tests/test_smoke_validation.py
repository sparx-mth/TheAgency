"""Regression cases discovered by real five-scene smoke execution."""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.planning.exploration.object_search_supervisor import ObjectSearchSupervisor
from sparx_agency.core.planning.exploration.rpt_room_solver import SolveRecord
from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark.kinematics import check_motion
from sparx_agency.tasks.planning.objnav_benchmark.results_io import strict_json
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.dataset import GibsonDataset
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.env import GibsonEnv
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_gibson import write_dataset, LineSimulator
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import setup_policy, observation


def test_selected_start_validation_does_not_inspect_unselected_invalid_episode(tmp_path):
    grid = np.zeros((7, 120, 120), np.uint8)
    grid[0] = 1
    grid[1:, 60, 20] = 1
    paths = write_dataset(tmp_path, grid, per_scene=2,
                          edit=lambda rows: rows[1].update(start_position=[100, 0, 100]))
    env = GibsonEnv(GibsonDataset(*paths, full=False), simulator=LineSimulator())
    try:
        env.validate_starts(env.episode_ids()[:1])
        with pytest.raises(ValueError, match="outside"):
            env.validate_starts()  # full validation remains strict
    finally:
        env.close()


def test_verified_native_sliding_is_allowed_but_large_steps_are_rejected():
    before = AgentPose(0, 0, 0, 0)
    check_motion(DiscreteAction.MOVE_FORWARD, before, AgentPose(0.2765, 0.0689, 0, 0),
                 PROTOCOL.actions(), PROTOCOL.kinematics())
    with pytest.raises(EnvContractError, match="advance at most"):
        check_motion(DiscreteAction.MOVE_FORWARD, before, AgentPose(0.4, 0, 0, 0),
                     PROTOCOL.actions(), PROTOCOL.kinematics())
    assert PROTOCOL.actions().forward_step_m == 0.25


def test_undefined_solver_bounds_do_not_erase_room_diagnostics():
    policy, episode = setup_policy(label="bed")

    class FallbackSolver:
        calls = 0
        last = SolveRecord(reason="synthetic fallback")

        def __call__(self, candidates, instance):
            self.calls += 1
            return [candidates[0].room_id]

    policy.solver = FallbackSolver()
    policy.supervisor = ObjectSearchSupervisor(policy.supervisor_params, solver=policy.solver)
    policy.plan(observation(episode, 0, depth=3.0))
    data = policy.episode_info()
    assert data["solver_records"][0]["expected_cost"] is None
    assert data["solver_records"][0]["bound_ratio"] is None
    assert "last_reasoning" in data and "doors" in data
    strict_json(data, "episode diagnostics")

