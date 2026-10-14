"""Coarse-to-fine phases: saved maps load back as priors, and a prior seeds the fine map."""
import numpy as np
import pytest

from sparx_agency.core.common.types import Intrinsics
from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction, DiscreteActionSpec
from sparx_agency.core.planning.objnav.types.camera import CameraSpec
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark_runtime.experiments.multi_resolution.transfer_phase import (
    TransferFalcon, load_prior, plan_statistics, save_phase_map)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_settings import RPTSettings

VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)


def camera():
    return CameraSpec(Intrinsics(640, 480, 400.0, 400.0, 319.5, 239.5), height_m=1.0, min_depth_m=0.1, max_depth_m=5.0)


def actions():
    return DiscreteActionSpec(forward_step_m=0.25, turn_angle_deg=30.0,
                              actions=(DiscreteAction.STOP, DiscreteAction.MOVE_FORWARD,
                                       DiscreteAction.TURN_LEFT, DiscreteAction.TURN_RIGHT))


def settings(**overrides):
    values = dict(body_height_m=1.0, body_radius_m=0.2, preferred_clearance_m=0.3, stop_distance_m=0.75)
    values.update(overrides)
    return RPTSettings(**values)


def grid(cells, resolution=0.5, origin=(-5.0, -5.0)):
    params = OccupancyGrid2DParams(resolution, origin[0], origin[1], "world")
    return OccupancyGrid2D(np.array(cells, dtype=np.int16), params, values=VALUES)


def episode():
    return ObjNavEpisode("e/1", "scene", "hm3d_v2", "train", "chair", camera(), actions(), 50, metadata={})


def test_saved_map_loads_back_as_the_same_grid(tmp_path):
    world = grid([[0, 100], [-1, 0]])
    observed = np.array([[True, False], [False, True]])
    save_phase_map(tmp_path / "map.npz", world, observed, floor_height_m=1.25)
    loaded = load_prior(tmp_path / "map.npz")
    assert np.array_equal(loaded.grid, world.grid)
    assert (loaded.resolution, loaded.origin_x, loaded.origin_y) == (0.5, -5.0, -5.0)
    assert loaded.values == VALUES


def test_loading_a_missing_map_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="nothing.npz"):
        load_prior(tmp_path / "nothing.npz")


def test_a_prior_is_seeded_before_the_first_observation():
    cells = np.full((40, 40), -1)
    cells[0, 0], cells[1, 1] = 0, 100
    prior = grid(cells, resolution=0.25, origin=(-5.0, -5.0))
    policy = TransferFalcon(settings(map_resolution_m=0.25, map_size_m=10.0), 0.8, prior=prior)
    policy.reset(episode(), None)
    _, percent = policy.mapping.grid.get_grid()
    assert np.count_nonzero(percent != -1) == 2
    assert not policy.mapping.grid.observed_mask.any()


def test_without_a_prior_the_map_starts_unknown_and_pose_anchored():
    policy = TransferFalcon(settings(map_resolution_m=0.25, map_size_m=10.0), 0.8)
    policy.reset(episode(), None)
    _, percent = policy.mapping.grid.get_grid()
    assert np.count_nonzero(percent != -1) == 0


def test_plan_statistics_summarise_statuses_and_latency():
    records = [{"status": "planned", "planning_ms": 10.0, "frontier_clusters": 3},
               {"status": "planned", "planning_ms": 30.0, "frontier_clusters": 5},
               {"status": "frontiers_exhausted", "planning_ms": 2.0}]
    stats = plan_statistics(records)
    assert stats["plans"] == 3
    assert stats["statuses"] == {"planned": 2, "frontiers_exhausted": 1}
    assert stats["max_frontier_clusters"] == 5
    assert stats["planning_ms_max"] == 30.0


def test_plan_statistics_of_no_plans_is_empty_not_an_error():
    assert plan_statistics([])["plans"] == 0


class StubAgent:
    def __init__(self, stop_after):
        self.calls, self.stop_after = 0, stop_after

    def act(self, observation):
        self.calls += 1
        action = DiscreteAction.STOP if self.calls > self.stop_after else DiscreteAction.MOVE_FORWARD
        return type("Decision", (), {"action": action})()


def frame():
    return np.zeros((480, 640, 3), np.uint8), np.ones((480, 640), np.float32), AgentPose(0.0, 0.0, 0.0, 0.0, camera_pitch=0.0)


class StubSimulator:
    def step(self, action):
        return frame()


def test_drive_records_coverage_at_each_checkpoint_reached():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.experiments.multi_resolution.transfer_phase import (
        PhaseConfig, _drive)
    policy = type("Policy", (), {"last_plan_status": "planned", "stop_reason": None})()
    protocol = type("Protocol", (), {"camera": staticmethod(camera)})()
    row = type("Row", (), {"category": "chair"})()
    config = PhaseConfig("fine", settings(), 50, 0, checkpoints=(2, 4, 100))
    trace, stop, curve = _drive(StubSimulator(), StubAgent(stop_after=5), policy, row, protocol, config,
                                frame(), probe=lambda: 0.5)
    assert len(trace) == 6 and stop == "policy_stop"
    assert curve == {2: 0.5, 4: 0.5}


def test_drive_without_a_probe_records_no_curve():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.experiments.multi_resolution.transfer_phase import (
        PhaseConfig, _drive)
    policy = type("Policy", (), {"last_plan_status": "planned", "stop_reason": None})()
    protocol = type("Protocol", (), {"camera": staticmethod(camera)})()
    row = type("Row", (), {"category": "chair"})()
    config = PhaseConfig("fine", settings(), 50, 0)
    _, _, curve = _drive(StubSimulator(), StubAgent(stop_after=3), policy, row, protocol, config, frame())
    assert curve == {}


def test_saved_resolution_round_trips_exactly(tmp_path):
    world = grid([[0]], resolution=0.3)
    save_phase_map(tmp_path / "map.npz", world, np.zeros((1, 1), bool), floor_height_m=0.0)
    assert load_prior(tmp_path / "map.npz").resolution == 0.3
