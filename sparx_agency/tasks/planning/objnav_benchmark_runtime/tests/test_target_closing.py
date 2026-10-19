"""Synthetic control/geometry regressions, not Habitat navigation scores."""
from dataclasses import replace
from types import SimpleNamespace
import math

import numpy as np
import pytest

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams
from sparx_agency.core.planning.objnav.action_converter.converter import DiscreteActionConverter
from sparx_agency.core.planning.objnav.action_converter.transition import apply_action
from sparx_agency.core.planning.objnav.camera_geometry import project_to_image
from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.mapping.scene_graph.serve.contract import DetectionWire
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.multifloor_dataset import MULTIFLOOR_PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import HabitatRGBDSimulator
from sparx_agency.core.planning.object_search.closing.target_closing import TargetClosingSettings
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_method import setup_policy, observation


def forbidden(*args, **kwargs):
    raise AssertionError("Global exploration was called after target takeover")


def freeze_global(monkeypatch, policy):
    for owner, names in ((policy, ("_decide", "_refresh_graph")),
                         (policy.loop, ("plan", "charge")), (policy.fallback, ("plan",)),
                         (policy.graph, ("reason",)), (policy.llm_client, ("chat_json",)),
                         (policy.building, ("plan", "prepare_observation", "observe", "filter_action"))):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)


@pytest.mark.parametrize("explorer", ["frontier", "falcon"])
def test_two_frames_take_over_both_explorers_without_global_calls(monkeypatch, explorer):
    p, ep = setup_policy(duplicate=True, local_exploration=explorer)
    freeze_global(monkeypatch, p)
    first = p.plan(observation(ep, 0))
    assert p.closing.active and not p.closing.locked and not first.stop
    assert p.plan(observation(ep, 0)) is first
    assert p.closing.count == 1
    command = p.plan(observation(ep, 1))
    assert command.stop and p.closing.locked
    assert p.filter_action(observation(ep, 1), DiscreteAction.STOP) == DiscreteAction.STOP
    p.notify_action(observation(ep, 1), DiscreteAction.STOP)
    assert p.graph.queries == p.solver.calls == 0
    assert p.episode_info()["target_closing"]["locked"]
    p.reset(ep, p.target)
    assert not p.closing.active


def test_confidence_threshold_does_not_enter_legacy_pursuit():
    p, ep = setup_policy()
    p.detector.detect = lambda rgb: [DetectionWire("chair", .49, (280, 200, 360, 280))]
    assert not p.plan(observation(ep, 0, depth=3)).stop
    assert not p.closing.active and p._target_xy is None


@pytest.mark.parametrize("depth", [float("nan"), float("inf"), 0.0])
def test_invalid_depth_never_starts_a_takeover(depth):
    """A confident box without depth support has nothing a second frame can be
    checked against: it must not suspend exploration, let alone lock."""
    p, ep = setup_policy()
    for step in range(3):
        command = p.plan(observation(ep, step, depth=depth))
        assert not command.stop and not p.closing.active and not p.closing.locked
        assert p.closing.phase == "SEARCH" and p.closing.count == 0
    assert p.closing.releases == 0


def test_depth_arrives_later_and_only_then_takes_over(monkeypatch):
    """Exploration keeps ownership through depthless frames; the first projected
    frame starts the takeover and the second consecutive one locks it."""
    p, ep = setup_policy()
    assert not p.plan(observation(ep, 0, depth=float("nan"))).stop and not p.closing.active
    p.plan(observation(ep, 1, depth=2))
    assert p.closing.active and p.closing.count == 1 and not p.closing.locked
    freeze_global(monkeypatch, p)
    p.plan(observation(ep, 2, depth=2))
    assert p.closing.locked and p.closing.count == 2


# -- detect high, associate low (Ranchester 2026-10-04: 0.68 at one heading, 0.42 centred) ----
def test_a_weak_resighting_on_the_anchor_completes_the_lock(monkeypatch):
    p, ep = setup_policy()
    p.detector.detect = lambda rgb: [DetectionWire("chair", .68, (280, 200, 360, 280))]
    p.plan(observation(ep, 0, depth=3.8))
    assert p.closing.active and p.closing.count == 1 and not p.closing.locked
    freeze_global(monkeypatch, p)
    p.detector.detect = lambda rgb: [DetectionWire("chair", .42, (280, 200, 360, 280))]   # same object, under 0.50
    p.plan(observation(ep, 1, depth=3.8))
    assert p.closing.locked and p.closing.count == 2
    assert p.closing.weak_resightings == 1
    assert p.episode_info()["target_closing"]["weak_resightings"] == 1


def test_a_weak_box_elsewhere_is_not_a_resighting_and_resets_the_run():
    p, ep = setup_policy()
    p.detector.detect = lambda rgb: [DetectionWire("chair", .68, (280, 200, 360, 280))]
    p.plan(observation(ep, 0, depth=3))
    assert p.closing.active and p.closing.count == 1
    p.detector.detect = lambda rgb: [DetectionWire("chair", .42, (40, 200, 120, 280))]     # 1.7 m to the side
    p.plan(observation(ep, 1, depth=3))
    assert p.closing.active and not p.closing.locked and p.closing.count == 0
    assert p.closing.weak_resightings == 0


def test_a_weak_box_never_starts_a_takeover():
    p, ep = setup_policy()
    p.detector.detect = lambda rgb: [DetectionWire("chair", .42, (280, 200, 360, 280))]
    for step in range(3):
        p.plan(observation(ep, step, depth=3))
        assert not p.closing.active and p.closing.count == 0
    assert p._target_xy is None


@pytest.mark.parametrize("kwargs", [{"track_confidence": 0.0}, {"track_confidence": 0.6},
                                    {"track_confidence": float("nan")}, {"track_confidence": True}])
def test_track_confidence_is_validated(kwargs):
    with pytest.raises(ValueError, match="track_confidence"):
        TargetClosingSettings(**kwargs)


def test_nonconsecutive_and_different_objects_do_not_confirm(monkeypatch):
    p, ep = setup_policy()
    freeze_global(monkeypatch, p)
    p.plan(observation(ep, 0, depth=2))
    p.plan(observation(ep, 1, depth=4))
    assert not p.closing.locked
    p.plan(observation(ep, 2, depth=2))
    assert p.closing.count == 1 and not p.closing.locked
    p.plan(observation(ep, 3, depth=2))
    assert p.closing.locked


def test_occlusion_retains_goal_and_same_path_beyond_old_loss_timeout(monkeypatch):
    p, ep = setup_policy(target_closing={"max_reacquire_steps": 2})
    freeze_global(monkeypatch, p)
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    p.plan(observation(ep, 0, depth=2))
    initial = p.plan(observation(ep, 1, depth=2))
    goal, xyz = p.closing.path.goal_xyz, p.closing.xyz
    plans = p._plan_calls
    p.detector.detect = lambda rgb: []
    for step in range(2, 12):
        command = p.plan(observation(ep, step, depth=2))
        assert not command.stop and command.waypoints == initial.waypoints and command.waypoints
        assert p.closing.xyz == xyz and p.closing.path.goal_xyz == goal
        assert p.closing.locked and p.closing.phase == "CLOSE_OCCLUDED"
    assert p._plan_calls == plans


def test_terminal_loss_does_not_move_or_stop_on_stale_evidence(monkeypatch):
    p, ep = setup_policy(target_closing={"max_reacquire_steps": 2})
    freeze_global(monkeypatch, p)
    p.plan(observation(ep, 0, depth=2))
    p.plan(observation(ep, 1, depth=2))
    p.detector.detect = lambda rgb: []
    near = replace(observation(ep, 2), pose=AgentPose(1.1, 0, 0, 0))
    for step in (2, 3):
        command = p.plan(replace(near, step=step))
        assert not command.stop and not command.waypoints
    command = p.plan(replace(near, step=4))
    assert not command.stop, "never a STOP on remembered range alone"
    assert not p.closing.locked and p.closing.phase == "RELEASED", "... and the lost lock is given back, the spot remembered"
    p, ep = setup_policy(target_closing={"max_reacquire_steps": 2, "release_on_failed_inspection": False})
    freeze_global(monkeypatch, p)
    p.plan(observation(ep, 0, depth=2))
    p.plan(observation(ep, 1, depth=2))
    p.detector.detect = lambda rgb: []
    for step in (2, 3):
        p.plan(replace(near, step=step))
    with pytest.raises(RuntimeError, match="terminal visual confirmation"):
        p.plan(replace(near, step=4))
    assert p.closing.locked and p.closing.phase == "FAILED"


def test_reobservation_refines_goal_but_missing_frames_do_not(monkeypatch):
    p, ep = setup_policy()
    freeze_global(monkeypatch, p)
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    p.plan(observation(ep, 0, depth=3))
    p.plan(observation(ep, 1, depth=3))
    goal = p.closing.path.goal_xyz
    p.plan(observation(ep, 2, depth=3.1))  # small update, keep stable endpoint
    assert p.closing.path.goal_xyz == goal
    # Precision-weighted fusion (since 2026-10-08): the three frames already in the memory
    # outweigh one, so a 0.4 m jump moves the estimate ~0.11 m; 0.7 m is a meaningful one.
    p.plan(observation(ep, 3, depth=3.7))  # still associated, meaningful refinement
    memory = p.closing.memory
    assert memory.n == 4 and memory.anchor[0] == pytest.approx(3.0, abs=0.05)
    assert 3.1 < memory.xyz[0] < 3.3 and memory.sigma_m < 0.07 and memory.spread_m > 0.2
    # A refinement that leaves the standoff inside the terminal radius KEEPS it (since 2026-10-11:
    # Collierville/000194 traded a reachable goal for a ring with no path): counted, re-referenced, same goal.
    assert p.closing.path.refinements == 1 and p.closing.path.kept == 1
    assert p.closing.path.goal_xyz == goal and math.dist(goal[:2], memory.xyz[:2]) <= p.settings.target_closing.terminal_distance_m
    # Two more far frames carry the memory beyond a metre from that goal: the first (3.32 m) is still
    # inside and kept again; the second (3.43 m) puts the goal 1.08 m out -- stale -- and it is re-selected
    # the action it falls out, although the memory moved under the refine distance since the last reference.
    p.plan(observation(ep, 4, depth=4.3))
    assert p.closing.path.goal_xyz == goal and p.closing.path.refinements == 2 and p.closing.path.kept == 2
    p.plan(observation(ep, 5, depth=4.3))
    assert math.dist(goal[:2], p.closing.memory.xyz[:2]) > p.settings.target_closing.terminal_distance_m
    assert p.closing.path.goal_xyz != goal and p.closing.path.refinements == 3 and p.closing.path.kept == 2
    assert p.closing.diagnostics()["goal_kept"] == 2


def test_expected_closing_failure_is_recordable_not_infrastructure():
    from sparx_agency.tasks.planning.objnav_benchmark.agent_contract import decide
    p, ep = setup_policy()
    agent = SimpleNamespace(act=lambda obs: p.closing.fail("terminal visual confirmation unavailable"))
    action, error = decide(agent, observation(ep, 0), ep.action_spec, "record")
    assert action == DiscreteAction.STOP and "terminal visual confirmation" in error
    assert p.closing.phase == "FAILED"  # forced STOP is not target-confirmed success


def test_planner_exception_does_not_reach_exploration_fallback(monkeypatch):
    p, ep = setup_policy()
    freeze_global(monkeypatch, p)
    p.plan(observation(ep, 0, depth=3))
    monkeypatch.setitem(p.__dict__, "_plan_to", lambda *a: (_ for _ in ()).throw(RuntimeError("planner broken")))
    with pytest.raises(RuntimeError, match="planner broken"):
        p.plan(observation(ep, 1, depth=3))
    assert p.closing.active and p.closing.locked


def test_unreachable_navmesh_never_uses_unsnapped_goal(monkeypatch):
    p, ep = setup_policy()
    freeze_global(monkeypatch, p)
    p.target_projector = lambda point: None
    monkeypatch.setitem(p.__dict__, "_plan_to", forbidden)
    p.plan(observation(ep, 0, depth=3))
    command = p.plan(observation(ep, 1, depth=3))
    assert not command.stop and not command.waypoints and p.closing.locked


@pytest.mark.parametrize("column, expected", [(100, DiscreteAction.TURN_LEFT), (540, DiscreteAction.TURN_RIGHT)])
def test_bbox_yaw_sign_and_offcenter_stop_rejection(column, expected):
    p, ep = setup_policy()
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (column-30, 200, column+30, 280))]
    p.plan(observation(ep, 0, depth=.6))
    command = p.plan(observation(ep, 1, depth=.6))
    assert not command.stop
    assert DiscreteActionConverter(ep.action_spec).step(AgentPose(0, 0, 0, 0), command).action == expected


@pytest.mark.parametrize("occlusion", [False, True])
def test_actual_converter_closes_low_target_looks_down_and_stops(monkeypatch, occlusion):
    p, original = setup_policy(target_closing={"max_reacquire_steps": 2})
    ep = replace(original, target_category="toilet", action_spec=MULTIFLOOR_PROTOCOL.actions())
    p.reset(ep, gibson_label_mapper().target_labels("toilet"))
    freeze_global(monkeypatch, p)
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    target = np.array([[2.0, 0.0, .45]])
    converter, pose, actions = DiscreteActionConverter(ep.action_spec), AgentPose(0, 0, 0, 0), []
    projections = []
    p.target_projector = lambda point: projections.append(point) or point
    for step in range(40):
        pixels, depths = project_to_image(target, ep.camera, pose)
        u, v = pixels[0]
        visible = depths[0] > 0 and 22 < u < 618 and 22 < v < 458
        boxes = [DetectionWire("toilet", .9, (u-20, v-20, u+20, v+20))] if visible and not (occlusion and 2 <= step <= 5) else []
        p.detector.detect = lambda rgb: boxes
        obs = replace(observation(ep, step, depth=float(depths[0])), pose=pose, target_category="toilet")
        command = p.plan(obs)
        action = converter.step(pose, command).action
        assert action is not None
        actions.append(action)
        p.notify_action(obs, action)
        if action == DiscreteAction.STOP:
            assert math.dist((pose.x, pose.y), target[0, :2]) <= p.settings.target_closing.terminal_distance_m
            assert visible and p.closing.locked
            break
        pose = apply_action(pose, action, ep.action_spec)
    assert actions[-1] == DiscreteAction.STOP
    assert DiscreteAction.MOVE_FORWARD in actions and DiscreteAction.LOOK_DOWN in actions
    assert projections and p.graph.queries == 0
    if occlusion:
        assert p.closing.occluded_path_steps >= 3, "the four hidden frames land in transit (one is spent reaching terminal range)"


def test_bbox_servo_does_not_interrupt_astar_corner_heading(monkeypatch):
    from sparx_agency.core.planning.objnav.types.command import NavigationCommand
    p, ep = setup_policy()
    p.plan(observation(ep, 0, depth=3))
    p.plan(observation(ep, 1, depth=3))
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (60, 200, 140, 280))]
    # Even if the live box asks left, a safe planned detour may need right.
    route = NavigationCommand.follow([(0, 0), (0, -1), (2, -1), (2, 0)])
    monkeypatch.setitem(p.closing.path.__dict__, "command", lambda *args: route)
    obs = replace(observation(ep, 2, depth=2.6), pose=AgentPose(0, 0, 0, -math.pi / 6))
    command = p.plan(obs)
    assert p.closing.last_seen == 2
    assert command.waypoints == route.waypoints and command.final_yaw is None
    assert DiscreteActionConverter(ep.action_spec).step(obs.pose, command).action == DiscreteAction.TURN_RIGHT


@pytest.mark.parametrize("snap", [(float("nan"), 0, 0), (-2, 4, -1), (-20, 3, -1)])
def test_navmesh_rejects_invalid_remote_or_other_floor_snaps(snap):
    sim = HabitatRGBDSimulator(None, None, .18, True, height_m=.88)
    sim._sim = SimpleNamespace(pathfinder=SimpleNamespace(snap_point=lambda point: snap))
    assert sim.project_target((1, 2, 3)) is None


def test_navmesh_round_trip_enu_native_coordinates():
    received = []
    sim = HabitatRGBDSimulator(None, None, .18, True, height_m=.88)
    sim._sim = SimpleNamespace(pathfinder=SimpleNamespace(snap_point=lambda point: received.append(point) or point))
    assert sim.project_target((1, 2, 3)) == (1, 2, 3)
    np.testing.assert_allclose(received[0], [-2, 3, -1])


@pytest.mark.parametrize("kwargs", [{"confirmation_frames": 1}, {"confirmation_frames": True},
                                    {"confidence": float("nan")}, {"confidence": 0}, {"max_verify_steps": 0},
                                    {"border_margin_px": -1}, {"border_margin_px": 2.0},
                                    {"release_unverified": 1}, {"rejection_radius_m": 0},
                                    {"stop_overshoot_steps": -1}, {"stop_overshoot_steps": 1.0},
                                    {"stop_overshoot_min_range_m": 0}, {"stop_overshoot_within_m": float("inf")},
                                    {"stop_overshoot_max_steps": 0}, {"stop_overshoot_max_steps": 2.0},
                                    {"stop_overshoot_steps": 3, "stop_overshoot_max_steps": 3, "stop_overshoot_max_actions": 2}])
def test_invalid_closing_settings(kwargs):
    with pytest.raises(ValueError):
        TargetClosingSettings(**kwargs)


@pytest.mark.parametrize("box", [(0, 269, 84, 479), (300, 0, 360, 80), (560, 200, 640, 280), (300, 440, 340, 480)])
def test_border_clipped_box_does_not_start_takeover(box):
    """The Ranchester sofa: an 84 px sliver on the left/bottom edge of a bed."""
    p, ep = setup_policy()
    p.detector.detect = lambda rgb: [DetectionWire("chair", .87, box)]
    p.plan(observation(ep, 0, depth=3))
    assert not p.closing.active and p.closing.border_rejections >= 1
    assert p.plan(observation(ep, 1, depth=3)) is not None and not p.closing.active


def test_locked_target_tolerates_border_spill(monkeypatch):
    p, ep = setup_policy()
    freeze_global(monkeypatch, p)
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    p.plan(observation(ep, 0, depth=2))
    p.plan(observation(ep, 1, depth=2))
    assert p.closing.locked
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (260, 120, 380, 480))]
    p.plan(observation(ep, 2, depth=2))
    assert p.closing.last_seen == 2 and p.closing.phase == "CLOSE"


def test_unverified_candidate_releases_to_exploration_and_is_not_retried():
    p, ep = setup_policy(target_closing={"max_verify_steps": 3})
    calls = []
    original = p.loop.plan
    p.loop.plan = lambda *a, **kw: calls.append(a[0].step) or original(*a, **kw)
    p.plan(observation(ep, 0, depth=3))
    assert p.closing.active and not p.closing.locked
    anchor = p.closing.anchor
    p.detector.detect = lambda rgb: []
    for step in (1, 2):
        command = p.plan(observation(ep, step, depth=3))
        assert p.closing.active and not command.stop
    assert calls == [0] or calls == []  # takeover owned steps 0-2
    released_at = len(calls)
    command = p.plan(observation(ep, 3, depth=3))
    assert not p.closing.active and p.closing.phase == "RELEASED" and p.closing.releases == 1
    assert p._target_xy is None and not command.stop
    assert command.info.get("kind") != "target_closing"
    assert len(calls) > released_at  # global exploration resumed on the release step
    assert p.episode_info()["target_closing"]["rejected"][0]["xyz"] == list(anchor)
    # The same spot flickering again (under the override confidence) must not start a second takeover...
    p.detector.detect = lambda rgb: [DetectionWire("chair", .7, (280, 200, 360, 280))]
    p.plan(observation(ep, 4, depth=3))
    assert not p.closing.active
    # ...but a confident, depth-projected object elsewhere still does (1.5 m away
    # from the rejected spot, and close enough to be exempt from the cooldown).
    p.plan(observation(ep, 5, depth=1.5))
    assert p.closing.active
    # A box the depth sensor cannot place never starts one at all.
    p.closing._release(observation(ep, 5, depth=1.5))
    p.plan(observation(ep, 6, depth=6))
    assert not p.closing.active


def test_release_disabled_keeps_episode_ending_error(monkeypatch):
    p, ep = setup_policy(target_closing={"max_verify_steps": 2, "release_unverified": False})
    freeze_global(monkeypatch, p)
    p.plan(observation(ep, 0, depth=3))
    p.detector.detect = lambda rgb: []
    p.plan(observation(ep, 1, depth=3))
    with pytest.raises(RuntimeError, match="consecutive depth-consistent"):
        p.plan(observation(ep, 2, depth=3))
    assert p.closing.phase == "FAILED"


def test_a_target_box_at_a_confirmed_other_objects_spot_is_its_own_instance_and_may_start_a_takeover():
    """Five frames of a bed, then a box that says chair at the same spot. Until 2026-10-07 the chair box
    voted on the bed and lost (``map_rejections``); with classes never merging it is a second instance --
    the detector's word, verified by the takeover's own two frames like any other candidate."""
    p, ep = setup_policy()
    p.detector.detect = lambda rgb: [DetectionWire("bed", .9, (280, 200, 360, 280))]
    for step in range(5):
        p.plan(observation(ep, step, depth=3))
    [bed] = p.landmarks.confirmed()
    assert bed.class_name == "bed" and bed.count == 5 and bed.votes == {}, "no votes without class voting"
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (280, 200, 360, 280))]
    p.plan(observation(ep, 5, depth=3))
    assert p.closing.active and p.closing.map_rejections == 0
    assert sorted(lm.class_name for lm in p.landmarks.all_landmarks()) == ["bed", "chair"], "two instances at one spot"
    assert p.landmarks.confirmed() == [bed], "one chair frame confirms nothing yet"
    assert p.perception.counts.get("outvoted_detections", 0) == 0


# -- the terminal inspection on a big object at close range (Ranchester couch, 2026-10-04) ----
def locked_at_two_metres(monkeypatch, **closing):
    """A chair locked two metres ahead on an empty map; the agent then stands 1.1 m from it."""
    p, ep = setup_policy(target_closing=closing) if closing else setup_policy()
    freeze_global(monkeypatch, p)
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    p.plan(observation(ep, 0, depth=2))
    p.plan(observation(ep, 1, depth=2))
    assert p.closing.locked and abs(p.closing.xyz[0] - 2.0) < 0.1 and abs(p.closing.xyz[1]) < 0.1
    near = replace(observation(ep, 2, depth=0.9), pose=AgentPose(1.1, 0, 0, 0))
    return p, ep, near


def test_a_fresh_box_spanning_the_image_centre_is_aligned_whatever_its_centre_says(monkeypatch):
    """The Ranchester box: clipped at the left edge, centre 124 px left of the image centre, the
    couch filling the frame. Centring it meant turning off the couch; spanning the centre column
    is alignment enough, and the near edge of the upholstery is the range that counts."""
    p, ep, near = locked_at_two_metres(monkeypatch)
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (0, 25, 392, 404))]
    command = p.plan(near)
    assert command.stop and p.closing.phase == "STOP"
    assert command.info["box_spans_centre"] is True and command.info["bbox_yaw_error_rad"] == 0.0
    assert command.info["measured_m"] <= 1.05 and command.info["reason"] == "fresh terminal target confirmation"
    assert p.closing.observed_near_m is not None and p.closing.observed_near_m <= command.info["measured_m"] + 1e-9


def test_an_exhausted_inspection_stops_when_the_target_was_seen_fresh_in_terminal_range(monkeypatch):
    p, ep, near = locked_at_two_metres(monkeypatch, max_reacquire_steps=3)
    # A box that does NOT span the centre: its centre says "turn", so the inspection turns, but the
    # target was fresh and within range from this spot -- that sighting is remembered.
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (40, 25, 300, 404))]
    turning = p.plan(near)
    assert not turning.stop and turning.info["target_visible"] and turning.info["measured_m"] <= 1.05
    assert p.closing.inspection_sighting == 2 and p.closing.inspection_started == 2
    p.detector.detect = lambda rgb: []
    for step in (3, 4):
        held = p.plan(replace(near, step=step))
        assert not held.stop and not held.waypoints
    final = p.plan(replace(near, step=5))
    assert final.stop and p.closing.phase == "STOP"
    assert final.info["reason"].startswith("inspection exhausted") and "action 2" in final.info["reason"]


# -- the STOP overshoot (Markleeville/000188, 2026-10-14: STOPped 1.27 m from the footprint on a 0.84 m reading) ----
def decided_at(monkeypatch, x, depth, steps=2, **closing):
    """A chair locked two metres ahead; the agent stands at ``x`` seeing it fill the frame at ``depth``.

    Two unconditional overshoot steps unless told otherwise: the mechanism's tests want the second step."""
    closing.setdefault("stop_overshoot_max_steps", max(steps, TargetClosingSettings().stop_overshoot_max_steps))
    p, ep, _ = locked_at_two_metres(monkeypatch, stop_overshoot_steps=steps, **closing)
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (0, 25, 392, 404))]
    frame = replace(observation(ep, 2, depth=depth), pose=AgentPose(x, 0, 0, 0))
    return p, ep, frame


def test_a_decided_stop_walks_two_steps_toward_the_target_and_stops_without_consulting_a_frame(monkeypatch):
    """The Markleeville geometry: the memory 1.25 m ahead, the near edge read at 0.84 m. The STOP is
    decided on that frame and taken two MOVE_FORWARDs later, whatever the detector shows meanwhile."""
    p, ep, frame = decided_at(monkeypatch, x=0.75, depth=0.84)
    converter = DiscreteActionConverter(ep.action_spec, p.converter_params)
    first = p.plan(frame)
    assert not first.stop and first.waypoints and p.closing.phase == "ADVANCE"
    assert first.info["stop_overshoot"]["steps"] == 0 and first.info["reason"].startswith(
        "STOP decided at action 2 (fresh terminal target confirmation): step 1 of at most 2")
    assert converter.step(frame.pose, first).action == DiscreteAction.MOVE_FORWARD
    memory = tuple(p.closing.xyz)
    p.notify_action(frame, DiscreteAction.MOVE_FORWARD)
    p.detector.detect = lambda rgb: []                       # the frames after the decision are not evidence
    second = p.plan(replace(frame, step=3, pose=AgentPose(1.0, 0, 0, 0)))
    assert not second.stop and second.waypoints and second.info["stop_overshoot"]["steps"] == 1
    assert converter.step(AgentPose(1.0, 0, 0, 0), second).action == DiscreteAction.MOVE_FORWARD
    p.notify_action(replace(frame, step=3), DiscreteAction.MOVE_FORWARD)
    final = p.plan(replace(frame, step=4, pose=AgentPose(1.25, 0, 0, 0)))
    assert final.stop and p.closing.phase == "STOP" and final.info["reason"] == "fresh terminal target confirmation"
    assert final.info["stop_overshoot"]["ended"] == "2 step(s) taken" and final.info["stop_overshoot"]["walked_m"] == 0.5
    assert tuple(p.closing.xyz) == memory and p.closing.releases == 0 and p.closing.last_seen == 2
    assert p.closing.overshoot_commands == 2 and p.episode_info()["target_closing"]["stop_overshoot"]["steps"] == 2


def test_by_default_the_overshoot_is_one_step_and_a_second_only_while_the_memory_stands_beyond_0_9_m(monkeypatch):
    """Two steps always were verified and judged too dear for the SPL; one always STOPped Markleeville/000188
    3 cm outside the disc (2026-10-14 evening). One step always, a second while the memory is beyond 0.90 m."""
    defaults = TargetClosingSettings()
    assert (defaults.stop_overshoot_steps, defaults.stop_overshoot_max_steps, defaults.stop_overshoot_within_m) == (1, 2, 0.90)
    p, ep, frame = decided_at(monkeypatch, x=0.75, depth=0.84, steps=1)
    first = p.plan(frame)
    assert first.waypoints and first.info["reason"].endswith("step 1 of at most 2 toward the target before it")
    p.notify_action(frame, DiscreteAction.MOVE_FORWARD)
    # The memory stands 1.03 m from the deciding spot: a step leaves it at 0.78, inside 0.90 -- no second step.
    final = p.plan(replace(frame, step=3, pose=AgentPose(1.0, 0, 0, 0)))
    assert final.stop and final.info["stop_overshoot"]["ended"].startswith("1 step(s) taken and the memory stands 0.7")
    assert final.info["stop_overshoot"]["ended"].endswith("m away, within 0.90 m")
    assert final.info["stop_overshoot"]["walked_m"] == 0.25 and p.closing.overshoot_commands == 1


def test_the_second_overshoot_step_is_taken_when_the_first_leaves_the_memory_beyond_0_9_m(monkeypatch):
    """The Markleeville/000188 geometry: the memory 1.25 m ahead on the deciding frame, a 0.84 m near edge.
    One step leaves the memory at 1.00 m -- the evaluator's radius to the footprint -- so a second is taken."""
    p, ep, frame = decided_at(monkeypatch, x=0.75, depth=0.84, steps=1)
    assert p.plan(frame).waypoints
    closing = p.closing
    closing.overshoot = None
    closing.xyz = (2.0, 0.0, closing.xyz[2])                       # the memory 1.25 m ahead of x = 0.75
    first = closing._terminal(frame, 1.25, 0.84, {"reason": "test"})
    assert first.waypoints and first.info["stop_overshoot"]["memory_m"] == 1.25
    closing.note_action(DiscreteAction.MOVE_FORWARD)
    second = closing._overshoot_step(replace(frame, step=3, pose=AgentPose(1.0, 0, 0, 0)))
    assert second.waypoints and second.info["stop_overshoot"]["steps"] == 1
    assert second.info["reason"].endswith("step 2 of at most 2 toward the target before it; the memory still stands 1.00 m away, beyond 0.90 m")
    closing.note_action(DiscreteAction.MOVE_FORWARD)
    final = closing._overshoot_step(replace(frame, step=4, pose=AgentPose(1.25, 0, 0, 0)))
    assert final.stop and final.info["stop_overshoot"]["ended"] == "2 step(s) taken" and final.info["stop_overshoot"]["memory_m"] == 0.75


def test_a_single_unconditional_step_never_takes_a_second(monkeypatch):
    p, ep, frame = decided_at(monkeypatch, x=0.75, depth=0.84, steps=1, stop_overshoot_max_steps=1)
    assert p.plan(frame).waypoints
    closing = p.closing
    closing.overshoot = None
    closing.xyz = (2.0, 0.0, closing.xyz[2])
    assert closing._terminal(frame, 1.25, 0.84, {"reason": "test"}).waypoints
    closing.note_action(DiscreteAction.MOVE_FORWARD)
    final = closing._overshoot_step(replace(frame, step=3, pose=AgentPose(1.0, 0, 0, 0)))
    assert final.stop and final.info["stop_overshoot"]["ended"] == "1 step(s) taken"


def test_the_overshoot_ends_at_once_on_a_blocked_step(monkeypatch):
    p, ep, frame = decided_at(monkeypatch, x=0.75, depth=0.84)
    assert p.plan(frame).waypoints
    p.notify_action(frame, DiscreteAction.MOVE_FORWARD)
    p.notify_blocked(replace(frame, step=3))                 # the converter: the step did not move the agent
    final = p.plan(replace(frame, step=3))
    assert final.stop and final.info["stop_overshoot"]["ended"] == "a step was blocked"
    assert final.info["stop_overshoot"]["steps"] == 1 and final.info["stop_overshoot"]["walked_m"] == 0.0


def test_the_overshoot_keeps_the_memory_beyond_its_minimum_range(monkeypatch):
    """From 0.9 m the second step would leave the memory 0.40 m away: one step, then the STOP."""
    p, ep, frame = decided_at(monkeypatch, x=1.1, depth=0.9)
    assert p.plan(frame).waypoints
    p.notify_action(frame, DiscreteAction.MOVE_FORWARD)
    final = p.plan(replace(frame, step=3, pose=AgentPose(1.35, 0, 0, 0)))
    assert final.stop and final.info["stop_overshoot"]["ended"].startswith("the memory would stand 0.4")
    assert final.info["stop_overshoot"]["steps"] == 1 and final.info["stop_overshoot"]["memory_m"] < 0.7


def test_the_overshoot_is_bounded_in_actions_even_without_the_hooks(monkeypatch):
    """A harness that never reports the actions: the stage still ends by its action bound."""
    p, ep, frame = decided_at(monkeypatch, x=0.75, depth=0.84, stop_overshoot_max_actions=3)
    for step in (2, 3, 4):
        assert p.plan(replace(frame, step=step)).waypoints
    final = p.plan(replace(frame, step=5))
    assert final.stop and final.info["stop_overshoot"]["ended"] == "3 actions spent"


def test_the_overshoot_aims_along_the_heading_within_a_turn_and_at_the_memory_beyond_it(monkeypatch):
    p, ep, frame = decided_at(monkeypatch, x=0.75, depth=0.84)
    assert p.plan(frame).waypoints
    closing, spec = p.closing, ep.action_spec
    info = {"reason": "test"}
    closing.overshoot = None
    within = replace(frame, step=3, pose=AgentPose(0.75, 0, 0, math.radians(20)))
    command = closing._terminal(within, 1.25, 0.84, info)
    dx, dy = command.waypoints[1][0] - 0.75, command.waypoints[1][1]
    assert abs(math.atan2(dy, dx) - math.radians(20)) < 1e-9, "within a turn: the heading that framed the object"
    closing.overshoot = None
    beyond = replace(frame, step=4, pose=AgentPose(0.75, 0, 0, math.radians(45)))
    command = closing._terminal(beyond, 1.25, 0.84, info)
    dx, dy = command.waypoints[1][0] - 0.75, command.waypoints[1][1]
    assert abs(math.atan2(dy, dx) - math.atan2(closing.xyz[1], closing.xyz[0] - 0.75)) < 1e-9, "beyond: the memory's bearing"
    assert DiscreteActionConverter(spec, p.converter_params).step(beyond.pose, command).action == DiscreteAction.TURN_RIGHT
    closing.overshoot = None
    away = replace(frame, step=5, pose=AgentPose(0.75, 0, 0, math.pi))
    command = closing._terminal(away, 1.25, 0.84, info)
    assert command.stop and command.info["stop_overshoot"]["ended"].startswith("the memory is 17")
    assert command.info["stop_overshoot"]["ended"].endswith("degrees off the heading")


def test_the_overshoot_switched_off_stops_on_the_deciding_frame(monkeypatch):
    p, ep, near = locked_at_two_metres(monkeypatch, stop_overshoot_steps=0, stop_overshoot_max_steps=0)
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (0, 25, 392, 404))]
    command = p.plan(replace(observation(ep, 2, depth=0.84), pose=AgentPose(0.75, 0, 0, 0)))
    assert command.stop and "stop_overshoot" not in command.info and p.closing.overshoot is None


def test_without_an_unconditional_step_the_overshoot_steps_only_while_the_memory_stands_beyond_0_9_m(monkeypatch):
    """``stop_overshoot_steps`` 0 with ``max_steps`` 2: a memory inside 0.90 m STOPs on the deciding frame."""
    p, ep, near = locked_at_two_metres(monkeypatch, stop_overshoot_steps=0, stop_overshoot_max_steps=2)
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (0, 25, 392, 404))]
    command = p.plan(replace(observation(ep, 2, depth=0.7), pose=AgentPose(1.2, 0, 0, 0)))
    assert command.stop and command.info["stop_overshoot"]["ended"].startswith("0 step(s) taken and the memory stands")
    assert command.info["stop_overshoot"]["memory_m"] <= 0.90
    closing = p.closing
    closing.overshoot = None
    closing.xyz = (2.45, 0.0, closing.xyz[2])                      # 1.25 m ahead: beyond 0.90, so a step
    assert closing._terminal(replace(observation(ep, 3), pose=AgentPose(1.2, 0, 0, 0)), 1.25, 0.7, {"reason": "test"}).waypoints


def test_an_exhausted_inspection_without_any_in_range_sighting_releases_the_lock_and_remembers_the_spot(monkeypatch):
    """Ranchester same-storey run: a sofa from four metres that was nothing from one. The lock was wrong;
    the episode must not end on it with the real couch 2.85 m away."""
    p, ep, near = locked_at_two_metres(monkeypatch, max_reacquire_steps=2)
    p.detector.detect = lambda rgb: []
    anchor = p.closing.anchor
    for step in (2, 3):
        assert not p.plan(replace(near, step=step)).stop
    command = p.plan(replace(near, step=4))
    assert not command.stop and command.info["kind"] == "target_released" and command.final_yaw is not None
    assert not command.waypoints, "one turn in place: the closing never explores, the next action is the search's"
    assert p.closing.inspection_sighting is None and not p.closing.active and not p.closing.locked
    assert p.closing.phase == "RELEASED" and p.closing.inspection_releases == 1 and p.closing.releases == 1
    assert p.closing.last_release == "inspection saw nothing"
    [(spot, floor, step, radius, why, stood)] = p.closing.rejected
    assert spot == anchor and step == 4 and radius == pytest.approx(2.0 * p.settings.target_closing.rejection_radius_m)
    assert why == "inspection saw nothing", "the memory keeps WHY: a failed inspection is never overridden"
    assert p.closing.rejected_near(anchor[:2], floor), "a landmark there is not worth another look"
    assert p.closing.rejected_near((anchor[0] + 1.5, anchor[1]), floor), "... within twice the plain radius"
    assert not p.closing.rejected_near((anchor[0] + 2.5, anchor[1]), floor)
    assert p._target_xy is None, "the legacy latch is cleared with the lock"
    assert p.closing.diagnostics()["rejected"][0]["radius_m"] == radius


def test_the_failed_inspection_release_can_be_switched_off(monkeypatch):
    p, ep, near = locked_at_two_metres(monkeypatch, max_reacquire_steps=2, release_on_failed_inspection=False)
    p.detector.detect = lambda rgb: []
    for step in (2, 3):
        assert not p.plan(replace(near, step=step)).stop
    with pytest.raises(RuntimeError, match="terminal visual confirmation"):
        p.plan(replace(near, step=4))
    assert p.closing.inspection_sighting is None and p.closing.phase == "FAILED"
    with pytest.raises(ValueError):
        TargetClosingSettings(release_on_failed_inspection="yes")
    with pytest.raises(ValueError):
        TargetClosingSettings(failed_inspection_radius_factor=0.0)


def test_an_inspection_that_tried_every_view_and_saw_nothing_releases_before_its_budget(monkeypatch):
    """The same-storey run spent 24 actions turning in place at a phantom. Three yaws (x pitches) once
    each is the whole inspection; nothing in any of them means nothing is there."""
    p, ep, near = locked_at_two_metres(monkeypatch, max_reacquire_steps=40)
    p.detector.detect = lambda rgb: []
    views = len(p.closing._inspection_views(p.closing._pitch(near, 0.9)))
    assert 3 <= views <= 9
    released_at, pose = None, near.pose
    for step in range(2, 40):
        command = p.plan(replace(near, step=step, pose=pose))
        assert not command.stop
        if p.closing.phase == "RELEASED":
            released_at = step
            break
        # The agent obeys: the commanded heading and pitch are the next frame's pose.
        pose = AgentPose(pose.x, pose.y, pose.z,
                         pose.yaw if command.final_yaw is None else float(command.final_yaw),
                         pose.camera_pitch if command.camera_pitch is None else float(command.camera_pitch))
    assert released_at is not None and released_at - 2 < 40, "released well inside the budget"
    assert released_at - 2 <= 2 * views + 1, "one or two actions per view: the turn and the tilt"
    assert p.closing.last_release == "inspection saw nothing in any view" and p.closing.inspection_releases == 1
    assert command.info["kind"] == "target_released"


def test_a_confirmed_object_of_another_class_at_the_anchor_is_a_second_object_and_the_lock_stands(monkeypatch):
    """Until 2026-10-07 four ``bed`` observations at a locked chair's anchor voted the chair into a bed and the
    map released the lock ("contradicted by the map"). Classes never merge now: the bed is a second
    landmark beside the chair, the map has no contradiction to make, and the lock stands."""
    p, ep, near = locked_at_two_metres(monkeypatch)
    anchor = p.closing.anchor
    chair = p.landmarks.all_landmarks()[0]
    for frame in range(10, 14):
        p.landmarks.observe("bed", anchor[:2], frame_id=frame, radius_m=chair.radius_m)
    classes = sorted(lm.class_name for lm in p.landmarks.confirmed())
    assert classes == ["bed", "chair"] and chair.class_name == "chair" and chair.votes == {}
    p.detector.detect = lambda rgb: []
    obs = replace(near, step=2)
    p.perception.observe(obs)
    p.closing.observe(obs)                                          # the policy's own order: perceive, then the closing
    assert p.closing.active and p.closing.locked and p.closing.phase != "RELEASED"
    assert p.closing.map_releases == 0 and p.closing.last_release is None and p.closing.rejected == []
    from sparx_agency.core.planning.object_search.perception.cycle import contradicted_by_map
    assert contradicted_by_map(p, "chair", anchor) is None, "no class vote, no contradiction"
    # The library's voting mode keeps the historical release available for whoever wires it.
    from sparx_agency.core.mapping.objects.landmarks import ObjectLandmarkMap
    voting = ObjectLandmarkMap(nearest_match=True, class_votes=True, footprint_iou=0.15)
    for frame in range(5):
        lm = voting.observe("bed", (0.0, 0.0), frame_id=frame, radius_m=0.8)
    assert voting.observe("sofa", (0.1, 0.0), frame_id=5, radius_m=0.8) is lm and lm.class_name == "bed"


def test_the_exhausted_inspection_stop_can_be_switched_off(monkeypatch):
    p, ep, near = locked_at_two_metres(monkeypatch, max_reacquire_steps=3, stop_on_exhausted_inspection=False,
                                       release_on_failed_inspection=False)
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (40, 25, 300, 404))]
    assert not p.plan(near).stop and p.closing.inspection_sighting == 2
    p.detector.detect = lambda rgb: []
    p.plan(replace(near, step=3))
    p.plan(replace(near, step=4))
    with pytest.raises(RuntimeError, match="terminal visual confirmation"):
        p.plan(replace(near, step=5))
    with pytest.raises(ValueError):
        TargetClosingSettings(stop_on_exhausted_inspection="yes")
    # With the release on, an in-range sighting still ends in a release rather than an error when STOP is off:
    # the target WAS seen from here, so the spot is remembered and the search goes on.
    p, ep, near = locked_at_two_metres(monkeypatch, max_reacquire_steps=3, stop_on_exhausted_inspection=False)
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (40, 25, 300, 404))]
    assert not p.plan(near).stop
    p.detector.detect = lambda rgb: []
    p.plan(replace(near, step=3))
    p.plan(replace(near, step=4))
    assert not p.plan(replace(near, step=5)).stop and p.closing.phase == "RELEASED"


def test_a_border_clipped_box_feeds_no_legacy_target_evidence_either():
    """The Ranchester upstairs run: a 48 px sliver at the left edge the takeover refused, which the
    older target-evidence path then pursued for nine actions. One gate for both."""
    p, ep = setup_policy()
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (0, 273, 48, 479))]
    for step in range(3):
        command = p.plan(observation(ep, step, depth=3))
        assert not command.stop and command.info.get("kind") != "target"
    assert not p.closing.active and p._target_xy is None and p._target_id is None
    assert p.perception.counts["border_clipped_evidence"] >= 1
    assert [row["target_evidence"] for row in p.perception.projections if row.get("status") == "fused"] == ["border_clipped"]
    assert p.landmarks.all_landmarks(), "the object is still mapped -- it is only not pursued"


def test_a_released_spot_does_not_block_a_close_view_of_the_same_object():
    """Ranchester attempt 3: the couch released at 3.8 m from the stair head was never re-verified
    from 1 m, twice, and the episode was lost on that. A close view is new evidence."""
    p, ep = setup_policy(target_closing={"max_verify_steps": 2})
    p.plan(observation(ep, 0, depth=3.5))                              # a far candidate, 3.5 m ahead
    assert p.closing.active and not p.closing.locked
    p.detector.detect = lambda rgb: []
    p.plan(observation(ep, 1, depth=3.5))
    p.plan(observation(ep, 2, depth=3.5))
    assert not p.closing.active and p.closing.releases == 1
    spot = p.closing.rejected[0][0]
    # Seen again from the same place, under the override confidence: the memory holds.
    p.detector.detect = lambda rgb: [DetectionWire("chair", .7, (280, 200, 360, 280))]
    p.plan(observation(ep, 3, depth=3.5))
    assert not p.closing.active
    # Seen from 1.2 m, standing 2.3 m closer: the same spot, a new view -- the takeover starts.
    near = replace(observation(ep, 4, depth=1.2), pose=AgentPose(2.3, 0, 0, 0))
    p.plan(near)
    assert p.closing.active and math.dist(p.closing.anchor[:2], spot[:2]) < p.settings.target_closing.rejection_radius_m


def test_an_overwhelming_box_starts_a_takeover_past_an_unverified_release_but_not_past_a_failed_inspection(monkeypatch):
    """Allensville toilet, actions 45-53: a 0.95-0.97 toilet in plain view at 2.4 m, refused nine frames
    running because a wrongly outvoted lock was remembered 2 m around it. The detector's word at 0.80 and
    up is not a flicker; the memory of a close look that saw nothing, or a stalled approach, still holds."""
    p, ep = setup_policy(target_closing={"max_verify_steps": 2, "release_cooldown_actions": 10})
    p.plan(observation(ep, 0, depth=3.5))
    p.detector.detect = lambda rgb: []
    p.plan(observation(ep, 1, depth=3.5))
    p.plan(observation(ep, 2, depth=3.5))
    assert not p.closing.active and p.closing.last_release == "unverified" and p.closing.rejected[0][4] == "unverified"
    # The same spot at 0.7 during the cooldown: refused, as before.
    p.detector.detect = lambda rgb: [DetectionWire("chair", .7, (280, 200, 360, 280))]
    p.plan(observation(ep, 3, depth=3.5))
    assert not p.closing.active and p.closing.overrides == 0
    # At 0.9: the override starts the takeover past the memory AND the cooldown.
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (280, 200, 360, 280))]
    p.plan(observation(ep, 4, depth=3.5))
    assert p.closing.active and p.closing.overrides == 1
    # A spot released by a failed inspection is never overridden.
    p, ep, near = locked_at_two_metres(monkeypatch, max_reacquire_steps=2)
    p.detector.detect = lambda rgb: []
    for step in (2, 3, 4):
        p.plan(replace(near, step=step))
    assert not p.closing.active and p.closing.last_release.startswith("inspection saw nothing")
    p.detector.detect = lambda rgb: [DetectionWire("chair", .95, (280, 200, 360, 280))]
    far = replace(observation(ep, 5, depth=3.0), pose=AgentPose(-1.0, 0, 0, 0))     # the same spot, from 3 m
    try:
        p.plan(far)
    except AssertionError:
        pass                                   # the frozen search was handed the action: no takeover started
    assert not p.closing.active and p.closing.overrides == 0
    with pytest.raises(ValueError):
        TargetClosingSettings(override_confidence=0.4)
    off = TargetClosingSettings(override_confidence=1.0)
    assert off.override_confidence == 1.0


def test_the_association_radius_grows_with_range_for_a_far_object():
    p, ep = setup_policy()
    s = p.settings.target_closing
    p.plan(observation(ep, 0, depth=4.0))                              # anchor 4 m ahead
    assert p.closing.active and p.closing.anchor is not None
    # Next frame the visible part of the long object is 0.7 m to the side: beyond the plain radius,
    # within the range-grown one (0.5 + 0.15 * 2 = 0.8 m).
    shifted_column = 320 + int(0.7 / 4.0 * ep.camera.intrinsics.fx)
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (shifted_column - 40, 200, shifted_column + 40, 280))]
    p.plan(observation(ep, 1, depth=4.0))
    assert p.closing.locked, "two consecutive frames of one long object at four metres"
    assert 0.5 < math.dist(p.closing.anchor[:2], p.closing.observed_xyz[:2]) < 0.8
    with pytest.raises(ValueError):
        TargetClosingSettings(association_range_gain=-0.1)
    with pytest.raises(ValueError):
        TargetClosingSettings(rejection_min_range_m=0.0)
    assert s.rejection_min_range_m == 2.0 and s.association_range_gain == 0.15


def test_verification_faces_a_visible_candidate_and_steps_towards_it_instead_of_sweeping():
    """Ranchester attempt 4: the +-30-degree sweep put a candidate three metres away at the frame's edge
    on every other frame, the consecutive count never reached two, and twelve-step verifications
    repeated from one spot six times. A visible candidate is faced; a centred one is approached."""
    p, ep = setup_policy()
    # Well off-centre to the left (34 degrees, beyond one turn) and far: the first action turns to face the box.
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (20, 200, 100, 280))]
    turning = p.plan(observation(ep, 0, depth=3.5))
    assert p.closing.active and not p.closing.locked and turning.info["target_visible"]
    assert turning.waypoints == () and turning.final_yaw is not None and turning.final_yaw > 0.1
    # Within one turn of the centre (18 degrees; Hanson 2026-10-05, a chair seen at one yaw only): a step,
    # not a turn -- the box stays in the frame and the next frame can be the consecutive one.
    p1, _ = setup_policy()
    p1.detector.detect = lambda rgb: [DetectionWire("chair", .9, (150, 200, 230, 280))]
    nudging = p1.plan(observation(ep, 0, depth=3.5))
    assert nudging.info.get("verify_step") == "towards the candidate" and nudging.waypoints
    # Centred and far: one step towards it, no sweep away from it.
    p2, _ = setup_policy()
    p2.detector.detect = lambda rgb: [DetectionWire("chair", .9, (280, 200, 360, 280))]
    stepping = p2.plan(observation(ep, 0, depth=3.5))
    assert stepping.info.get("verify_step") == "towards the candidate" and stepping.waypoints
    assert math.dist(stepping.waypoints[-1], (0.0, 0.0)) == pytest.approx(2 * ep.action_spec.forward_step_m)
    assert DiscreteActionConverter(ep.action_spec).step(AgentPose(0, 0, 0, 0), stepping).action == DiscreteAction.MOVE_FORWARD
    # Out of view: the sweep around the bearing, as before.
    p2.detector.detect = lambda rgb: []
    sweeping = p2.plan(observation(ep, 1, depth=3.5))
    assert sweeping.info["target_visible"] is False and sweeping.waypoints == () and sweeping.final_yaw is not None


def test_a_release_cools_far_candidates_but_not_close_ones():
    p, ep = setup_policy(target_closing={"max_verify_steps": 2, "release_cooldown_actions": 10})
    p.plan(observation(ep, 0, depth=3.5))
    p.detector.detect = lambda rgb: []
    p.plan(observation(ep, 1, depth=3.5))
    p.plan(observation(ep, 2, depth=3.5))
    assert not p.closing.active and p.closing.released_step == 2
    # A DIFFERENT far object two actions later, under the override confidence: cooling, no takeover.
    far_elsewhere = replace(observation(ep, 4, depth=3.5), pose=AgentPose(0, 0, 0, math.pi / 2))
    p.detector.detect = lambda rgb: [DetectionWire("chair", .7, (280, 200, 360, 280))]
    p.plan(far_elsewhere)
    assert not p.closing.active, "the search moves before it spends two more frames on a far candidate"
    # A close object during the cooldown: taken.
    close = replace(observation(ep, 5, depth=1.2), pose=AgentPose(0, 0, 0, math.pi))
    p.plan(close)
    assert p.closing.active
    with pytest.raises(ValueError):
        TargetClosingSettings(release_cooldown_actions=-1)



# -- a lock with no safe path: the footing sweep, then the spot is given up (2026-10-05) ----------
def tilt_episode(p, original):
    ep = replace(original, action_spec=MULTIFLOOR_PROTOCOL.actions())
    p.reset(ep, p.target)
    return ep


def drive(p, ep, converter, pose, step, boxes):
    """One action through the real converter (an idle step is the headless agent's TURN_LEFT); returns
    ``(command, action, pose after it)``."""
    p.detector.detect = lambda rgb: boxes
    obs = replace(observation(ep, step, depth=3.0), pose=pose)
    command = p.plan(obs)
    result = converter.step(pose, command)
    action = DiscreteAction.TURN_LEFT if result.idle else result.action      # STOP is 0: never read it as idle
    if p.closing.active:                                       # after a release the action is the frozen search's
        p.notify_action(obs, action)
    return command, action, apply_action(pose, action, ep.action_spec)


def test_a_locked_target_with_no_safe_path_maps_its_footing_then_gives_the_spot_up_without_a_rejection(monkeypatch):
    """Hanson/000002: spawned beside a bed with the plant in view 4.4 m away, 159 actions turning on
    'no safe target path'. The floor under the camera was unknown, and unknown is impassable."""
    p, original = setup_policy()
    ep = tilt_episode(p, original)
    freeze_global(monkeypatch, p)
    world = OccupancyGrid2D(np.full((200, 200), -1, np.int8), OccupancyGrid2DParams(.1, -10, -10))   # the blind disk: unknown
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    p.target_projector = lambda point: None                     # no standoff goal exists from here: no path
    converter, pose = DiscreteActionConverter(ep.action_spec), AgentPose(0, 0, 0, 0)
    box = [DetectionWire("chair", .9, (280, 200, 360, 280))]
    actions, phases = [], []
    for step in range(60):
        command, action, pose = drive(p, ep, converter, pose, step, box)
        actions.append(action)
        phases.append(p.closing.phase)
        if not p.closing.active:
            break
    assert p.closing.locked is False and p.closing.phase == "RELEASED"
    assert p.closing.footing_sweeps == 1 and "FOOTING" in phases
    assert actions.count(DiscreteAction.LOOK_DOWN) == 1 and actions.count(DiscreteAction.LOOK_UP) == 1
    turns = sum(1 for a in actions if a in (DiscreteAction.TURN_LEFT, DiscreteAction.TURN_RIGHT))
    assert turns >= 12, "a full circle at the footing pitch"
    assert pose.camera_pitch == pytest.approx(0.0), "the camera is level again when the lock is given up"
    assert len(actions) < 40, "not the 160-action closing bound"
    assert p.closing.boxed_releases == 1 and p.closing.rejected == [], "the target was not disproved, the spot was"
    assert p.closing.last_release == "no safe path from here"
    assert command.info["kind"] == "target_released" and "no safe path" in command.info["reason"]
    [spot] = p.closing.boxed_in
    assert math.dist(spot[0], (pose.x, pose.y)) < 0.5 and spot[1] == 0
    assert p.closing.diagnostics()["boxed_releases"] == 1 and p.closing.diagnostics()["boxed_in"][0]["floor_id"] == 0
    # From the same spot the same far candidate starts no new takeover; a metre away, past the cooldown, it does.
    step += 1
    for later in range(step, step + 20):
        p.closing.observe(replace(observation(ep, later, depth=3.0), pose=pose))
        assert not p.closing.active
    moved = replace(pose, x=pose.x + 1.5, y=pose.y + 1.5)
    p.closing.observe(replace(observation(ep, step + 25, depth=3.0), pose=moved))
    assert p.closing.active
    # On a floor already known around the feet there is nothing to sweep: the spot is given up without one.
    p, original = setup_policy()
    ep = tilt_episode(p, original)
    freeze_global(monkeypatch, p)
    known = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: known)
    p.target_projector = lambda point: None
    converter, pose = DiscreteActionConverter(ep.action_spec), AgentPose(0, 0, 0, 0)
    for step in range(30):
        command, action, pose = drive(p, ep, converter, pose, step, box)
        if not p.closing.active:
            break
    assert p.closing.footing_sweeps == 0 and p.closing.boxed_releases == 1 and step <= 10


def test_without_the_footing_sweep_a_pathless_lock_is_still_given_up_after_its_bound(monkeypatch):
    """``footing_sweep=False`` drops the sweep alone; the spot is given up after
    ``release_after_footing_steps`` pathless actions all the same. The no-tilt Gibson protocol
    (no LOOK actions) takes this path by construction."""
    p, original = setup_policy(target_closing={"footing_sweep": False})
    ep = tilt_episode(p, original)
    freeze_global(monkeypatch, p)
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    p.target_projector = lambda point: None
    converter, pose = DiscreteActionConverter(ep.action_spec), AgentPose(0, 0, 0, 0)
    box = [DetectionWire("chair", .9, (280, 200, 360, 280))]
    actions = []
    for step in range(30):
        command, action, pose = drive(p, ep, converter, pose, step, box)
        actions.append(action)
        if not p.closing.active:
            break
    assert DiscreteAction.LOOK_DOWN not in actions and p.closing.footing_sweeps == 0
    assert not p.closing.active and p.closing.boxed_releases == 1 and len(actions) <= 10
    # The Gibson protocol has no LOOK actions: the same release, no sweep asked for.
    p, ep = setup_policy()
    freeze_global(monkeypatch, p)
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    p.target_projector = lambda point: None
    converter, pose = DiscreteActionConverter(ep.action_spec), AgentPose(0, 0, 0, 0)
    for step in range(30):
        command, action, pose = drive(p, ep, converter, pose, step, box)
        if not p.closing.active:
            break
    assert p.closing.boxed_releases == 1 and p.closing.footing_sweeps == 0 and step <= 10


@pytest.mark.parametrize("kwargs", [{"footing_after_steps": 0}, {"release_after_footing_steps": 0},
                                    {"boxed_in_radius_m": 0.0}, {"footing_sweep": "yes"},
                                    {"footing_unknown_fraction": 1.5}, {"footing_radius_m": 0.0}])
def test_footing_settings_are_validated(kwargs):
    with pytest.raises(ValueError):
        TargetClosingSettings(**kwargs)


# -- the sweep ends when a path appears; the pitch is the height's; a sighting STOPs at any pitch (2026-10-05) ------
def test_the_footing_sweep_is_cut_short_the_action_a_path_exists_and_the_approach_is_level(monkeypatch):
    """Hanson/000002, second fly: the sweep found the path at its third turn, but its pitch stood for the
    whole approach and the first inspection (owner FOOTING, 30 degrees down, actions 5-63)."""
    p, original = setup_policy()
    ep = tilt_episode(p, original)
    freeze_global(monkeypatch, p)
    unknown = OccupancyGrid2D(np.full((200, 200), -1, np.int8), OccupancyGrid2DParams(.1, -10, -10))
    known = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    state = {"world": unknown, "path": False}
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: state["world"])
    p.target_projector = lambda point: point if state["path"] else None
    converter, pose = DiscreteActionConverter(ep.action_spec), AgentPose(0, 0, 0, 0)
    box = [DetectionWire("chair", .9, (280, 200, 360, 280))]
    actions, owners = [], []
    for step in range(40):
        command, action, pose = drive(p, ep, converter, pose, step, box)
        actions.append(action)
        owners.append(command.info.get("camera", {}).get("owner"))
        if p.closing.phase == "FOOTING" and step >= 5 and not state["path"]:
            state["world"], state["path"] = known, True          # the floor is mapped and a standoff exists now
        if p.closing.phase in ("CLOSE", "CLOSE_OCCLUDED") and actions.count(DiscreteAction.MOVE_FORWARD) >= 2:
            break
    assert p.closing.footing_sweeps == 1 and p.closing.phase in ("CLOSE", "CLOSE_OCCLUDED")
    assert actions.count(DiscreteAction.LOOK_UP) == 1, "restored the action the path appeared, not after the circle"
    assert actions.count(DiscreteAction.TURN_LEFT) < 12, "the circle was cut short"
    assert pose.camera_pitch == pytest.approx(0.0) and owners[-1] == "TARGET_CLOSING", "the approach is level"
    assert not p.camera_control.footing and p.camera_control.inspection is None


def test_the_inspection_pitch_follows_the_targets_height_not_its_label(monkeypatch):
    """A potted plant in a metre-tall planter is at camera height; the camera stays level."""
    p, original = setup_policy()
    ep = replace(tilt_episode(p, original), target_category="potted plant")
    p.reset(ep, gibson_label_mapper().target_labels("potted plant"))
    freeze_global(monkeypatch, p)
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    p.detector.detect = lambda rgb: [DetectionWire("potted plant", .9, (280, 200, 360, 280))]
    p.plan(replace(observation(ep, 0, depth=2), target_category="potted plant"))
    p.plan(replace(observation(ep, 1, depth=2), target_category="potted plant"))
    assert p.closing.locked and abs(p.closing.xyz[2] - ep.camera.height_m) < 0.2, "a box at the image centre is at camera height"
    near = replace(observation(ep, 2, depth=0.9), pose=AgentPose(1.1, 0, 0, 0), target_category="potted plant")
    command = p.plan(near)
    assert command.stop and command.info["reason"] == "fresh terminal target confirmation"
    assert p.closing._pitch(near, 0.9) == pytest.approx(0.0), "no look-down for an object at eye level"
    # A toilet 0.45 m up with the camera at 0.88 m is low by its height: the look-down stands.
    p, original = setup_policy()
    ep = replace(tilt_episode(p, original), target_category="toilet")
    p.reset(ep, gibson_label_mapper().target_labels("toilet"))
    p.closing.xyz = (1.0, 0.0, 0.45)
    assert p.closing._pitch(replace(observation(ep, 0), target_category="toilet"), 0.9) == pytest.approx(ep.action_spec.tilt_angle_rad)


def test_a_fresh_centred_in_range_sighting_stops_at_whatever_pitch_it_came(monkeypatch):
    """The Hanson toilet projected at 60 degrees down where its height predicted 30: twenty actions of
    LOOK_UP / LOOK_DOWN after a fresh, centred sighting at 0.91 m, until the budget STOPped."""
    # The synthetic toilet of this fixture stands at camera height, which the context check would
    # doubt (tested in test_target_context); the pitch mechanics are the point here.
    p, original = setup_policy(target_closing={"context_check": False})
    ep = replace(tilt_episode(p, original), target_category="toilet")
    p.reset(ep, gibson_label_mapper().target_labels("toilet"))
    freeze_global(monkeypatch, p)
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    p.detector.detect = lambda rgb: [DetectionWire("toilet", .9, (280, 200, 360, 280))]
    p.plan(replace(observation(ep, 0, depth=2), target_category="toilet"))
    p.plan(replace(observation(ep, 1, depth=2), target_category="toilet"))
    assert p.closing.locked
    # A low target 0.8 m ahead: its height predicts 30 degrees down. The camera stands at 60, and a flat
    # 0.8 m depth along a ray pitched 60 degrees lands 0.42 m from the anchor: associated, fresh, in range.
    p.closing.xyz = p.closing.anchor = (1.9, 0.0, 0.3)
    pitched = replace(observation(ep, 2, depth=0.8), pose=AgentPose(1.1, 0, 0, 0, camera_pitch=2 * ep.action_spec.tilt_angle_rad),
                      target_category="toilet")
    assert p.closing._pitch(pitched, 0.8) == pytest.approx(ep.action_spec.tilt_angle_rad), "the prediction says 30"
    command = p.plan(pitched)
    assert command.stop and command.info["reason"] == "fresh terminal target confirmation"
    assert command.info["target_visible"] and command.info["measured_m"] <= 1.05


def test_the_legacy_target_evidence_does_not_walk_after_what_the_takeover_refused(monkeypatch):
    """Hanson/000001, second fly: the takeover released a far chair as unverified at action 18 and the
    legacy pursuit walked nine actions toward the same landmark, moving the agent off its warm-up spot."""
    p, ep = setup_policy(target_closing={"max_verify_steps": 2})
    p.detector.detect = lambda rgb: [DetectionWire("chair", .7, (280, 200, 360, 280))]
    p.plan(observation(ep, 0, depth=3.5))
    assert p.closing.active and not p.closing.locked
    p.detector.detect = lambda rgb: []
    p.plan(observation(ep, 1, depth=3.5))
    p.plan(observation(ep, 2, depth=3.5))
    assert not p.closing.active and p.closing.last_release == "unverified" and p.closing.rejected
    p.detector.detect = lambda rgb: [DetectionWire("chair", .7, (280, 200, 360, 280))]
    for step in (3, 4, 5):
        p.plan(observation(ep, step, depth=3.5))
    assert p._target_xy is None, "the released spot is not a legacy target either"
    assert p.perception.counts["refused_evidence"] >= 1
    assert any(row.get("target_evidence") == "refused_by_takeover" for row in p.perception.projections)
    assert not p.closing.active, "and the takeover itself still refuses it from here"


# -- the end-to-end review of 2026-10-05 (evening) ------------------------------------------------
def test_a_fresh_off_centre_sighting_at_another_pitch_is_centred_not_tilted_away(monkeypatch):
    """A toilet in view at 0 degrees, 20 degrees left of the centre column, where its height predicts a
    30-degree look-down: the converter tilts before it faces, so asking for the predicted pitch LOOKed
    away from the target and never turned -- LOOK_DOWN / LOOK_UP for the whole 24-action budget. The
    first move is the TURN. Since 2026-10-11 the centred, bottom-cut box then earns ONE centring
    LOOK_DOWN toward it before the STOP; this synthetic detector shows nothing once pitched, and the
    STOP is taken on the previous frame's evidence the very next action -- never a sweep."""
    p, original = setup_policy(target_closing={"context_check": False})      # an eye-level synthetic toilet, see above
    ep = replace(tilt_episode(p, original), target_category="toilet")
    p.reset(ep, gibson_label_mapper().target_labels("toilet"))
    freeze_global(monkeypatch, p)
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    p.detector.detect = lambda rgb: [DetectionWire("toilet", .9, (280, 200, 360, 280))]
    p.plan(replace(observation(ep, 0, depth=2), target_category="toilet"))
    p.plan(replace(observation(ep, 1, depth=2), target_category="toilet"))
    assert p.closing.locked
    p.closing.xyz = p.closing.anchor = (1.9, 0.0, 0.3)                 # low: the prediction says 30 down
    converter, pose = DiscreteActionConverter(ep.action_spec), AgentPose(1.1, 0, 0, 0)
    k = ep.camera.intrinsics
    bearing = math.radians(20)                                          # the toilet 20 degrees left of the agent's heading

    row = k.cy + k.fy * (ep.camera.height_m - 0.3) / 0.8                 # a 0.3 m high object 0.8 m away, level camera

    def boxes(pose):
        """The box where a level camera at ``pose`` sees the toilet; nothing when the camera is pitched."""
        if abs(pose.camera_pitch) > 1e-6:
            return []
        column = k.cx - k.fx * math.tan(bearing - pose.yaw)
        return [DetectionWire("toilet", .9, (column - 40, min(row, k.height - 1) - 30, column + 40, min(row, k.height - 1)))]
    actions = []
    for step in range(2, 8):
        p.detector.detect = lambda rgb, pose=pose: boxes(pose)
        obs = replace(observation(ep, step, depth=0.8), pose=pose, target_category="toilet")
        command = p.plan(obs)
        if command.stop:
            break
        action = converter.step(pose, command).action
        actions.append(action)
        p.notify_action(obs, action)
        pose = apply_action(pose, action, ep.action_spec)
    assert command.stop and command.info["reason"].startswith("fresh terminal target confirmation")
    assert DiscreteAction.TURN_LEFT in actions[:2], "the first move centres the target at the pitch it was seen at"
    assert actions.count(DiscreteAction.LOOK_DOWN) <= 1 and DiscreteAction.LOOK_UP not in actions, (
        "at most the one centring tilt toward the bottom-cut box, never a sweep: %r" % (actions,))
    assert len(actions) <= 3 and p.closing.stop_centring_tilts == 1
    assert "the centring tilt lost the box, the sighting stands" in command.info["reason"]


def test_a_glance_in_force_is_aborted_when_a_takeover_starts(monkeypatch):
    """A cue glance toward a box at the frame edge is often what un-clips it; left in force it resumed,
    stale, after the release -- turns toward a yaw the cleared route no longer has."""
    p, ep = setup_policy()
    freeze_global(monkeypatch, p)
    p.glances.active = {"kind": "cue", "target": 1.0, "turns": 0, "started": 0, "class": "chair", "side": "right"}
    p.plan(observation(ep, 0, depth=2))
    assert p.closing.active
    assert p.glances.active is None and p.glances.stats["aborted"] == 1


def test_an_inspection_entered_early_steps_closer_when_the_fresh_range_is_beyond_the_limit(monkeypatch):
    """The filtered estimate read 1.0 m (an older, nearer reading averaged in) while every fresh frame
    measured the surface at 1.3 m: the agent stood still through the budget and released a target in
    plain view as 'inspection saw nothing'. Out of range and seen fresh means walk, not look."""
    p, ep = setup_policy(target_closing={"max_reacquire_steps": 4})
    freeze_global(monkeypatch, p)
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    p.plan(observation(ep, 0, depth=2))
    p.plan(observation(ep, 1, depth=2))
    assert p.closing.locked
    p.closing.xyz = (1.6, 0.0, p.closing.xyz[2])                       # the filtered estimate: 0.5 m ahead of the agent at x=1.1
    here = AgentPose(1.1, 0, 0, 0)
    p.detector.detect = lambda rgb: []
    first = p.plan(replace(observation(ep, 2, depth=1.3), pose=here))
    assert p.closing.inspection_started == 2 and not first.stop, "the filtered estimate entered the inspection, unseen"
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (280, 200, 360, 280))]
    second = p.plan(replace(observation(ep, 3, depth=1.3), pose=here))   # the fresh surface: 1.3 m ahead
    assert p.closing.inspection_started is None and p.closing.inspection_resumptions == 1
    assert second.waypoints, "the approach resumed: a route toward the target, not another look"
    assert p.closing.active and p.closing.locked and not p.closing.rejected, "nothing was released"
    third = p.plan(replace(observation(ep, 4, depth=1.3), pose=here))
    assert p.closing.inspection_started is None and third.waypoints, "and a fresh out-of-range frame does not re-enter it"


def test_the_map_keeps_mapping_during_a_takeover_but_another_class_never_outvotes_the_lock(monkeypatch):
    """Until 2026-10-05 nothing fed the landmark map while the takeover walked in. Since 2026-10-07 classes
    never merge: a ``bed`` box at the locked chair's spot opens a bed landmark beside it instead of voting
    the chair into a bed, and the lock stands (the Allensville toilet was lost to exactly such a vote)."""
    p, ep = setup_policy(target_closing={"failed_inspection_radius_factor": 2.0})
    freeze_global(monkeypatch, p)
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    p.plan(observation(ep, 0, depth=2))
    p.plan(observation(ep, 1, depth=2))
    assert p.closing.locked and p.landmarks.all_landmarks(), "the chair is on the map from the takeover's own frames"
    assert all(row.get("target_evidence") == "takeover_active" for row in p.perception.projections
               if row["status"] == "fused"), "the legacy evidence stands down while the takeover owns the target"
    assert p._target_id is None
    p.detector.detect = lambda rgb: [DetectionWire("bed", .95, (280, 200, 360, 280))]
    for step in range(2, 8):
        p.plan(observation(ep, step, depth=2))
        assert p.closing.active and p.closing.locked
    assert p.closing.map_releases == 0 and p.closing.last_release is None
    assert sorted(lm.class_name for lm in p.landmarks.all_landmarks()) == ["bed", "chair"], "two instances, no vote"
    assert p.perception.counts.get("outvoted_detections", 0) == 0 and not p.landmarks.class_votes


def test_the_closings_own_release_turn_is_not_charged_to_the_room_loop(monkeypatch):
    p, ep, near = locked_at_two_metres(monkeypatch, max_reacquire_steps=2)
    p.detector.detect = lambda rgb: []
    for step in (2, 3):
        p.plan(replace(near, step=step))
    released = p.plan(replace(near, step=4))
    assert not p.closing.active and released.info["kind"] == "target_released"
    charged = []
    monkeypatch.setattr(p.loop, "charge", lambda: charged.append(1))
    paused = []
    monkeypatch.setattr(p.supervisor, "pause", lambda seconds: paused.append(seconds))
    p.notify_action(replace(near, step=4), DiscreteAction.TURN_LEFT)
    assert charged == [] and paused == [p.settings.action_time_s]


def test_an_approach_that_goes_nowhere_releases_the_lock_before_the_closing_bound(monkeypatch):
    """Leonardo couch (2026-10-05): locked on a real sofa 4.6 m away, A* found a path every action and the
    follower pushed into an obstacle the map did not show for 160 actions, until the closing bound
    raised an agent error. Twenty CLOSE actions inside a 20 cm circle release the lock with a rejection."""
    p, ep = setup_policy(target_closing={"approach_stall_actions": 6})
    freeze_global(monkeypatch, p)
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    p.plan(observation(ep, 0, depth=3.5))
    p.plan(observation(ep, 1, depth=3.5))
    assert p.closing.locked and p.closing.phase in ("CLOSE", "VERIFY")
    jitter = [(0.00, 0.00), (0.05, 0.02), (-0.03, 0.05), (0.06, -0.04), (0.02, 0.06), (-0.05, 0.01), (0.04, 0.03), (0.0, -0.05)]
    released = None
    for step, (dx, dy) in enumerate(jitter, 2):
        command = p.plan(replace(observation(ep, step, depth=3.5), pose=AgentPose(dx, dy, 0, 0)))
        if not p.closing.active:
            released = command
            break
    assert released is not None, "the lock was released by the stall clock"
    assert released.info["kind"] == "target_released" and "went nowhere" in released.info["reason"]
    assert p.closing.stall_releases == 1 and p.closing.last_release.startswith("approach stalled")
    assert p.closing.rejected, "a rejection: the same far candidate is not taken again from here"
    assert p.closing.failure is None, "no agent error"
    with pytest.raises(ValueError):
        TargetClosingSettings(approach_stall_actions=-1)
    with pytest.raises(ValueError):
        TargetClosingSettings(approach_stall_m=0.0)


def test_the_blocked_clock_survives_collision_jitter():
    """Leonardo toilet (2026-10-05): every blocked MOVE_FORWARD slid the agent 5-10 cm, and that reset the
    clock the BLOCKED verdict waits on; the agent jittered in a wedge for 350 actions."""
    p, ep = setup_policy()
    p.detector.detect = lambda rgb: []                                              # no takeover: the search owns the action
    p.plan(observation(ep, 0))
    p._blocked_since = 3.0
    p._last_pose = (0.0, 0.0)
    p.plan(replace(observation(ep, 1), pose=AgentPose(0.08, 0.05, 0, 0)))          # a collision slide
    assert p._blocked_since == 3.0, "a slide is not progress"
    p.plan(replace(observation(ep, 2), pose=AgentPose(0.30, 0.05, 0, 0)))          # a forward step's worth
    assert p._blocked_since is None, "a step clears the clock"


# -- the review of 2026-10-07: bounded re-takeovers, blind frames, the suspect's step, the locked frame ----
def test_a_near_candidate_released_from_this_spot_is_not_taken_over_again_until_the_agent_moves():
    """Twelve takeovers of one chair 1.5 m away in eighty actions, from one spot: a close view is new
    evidence from a NEW spot, the same evidence from the spot it was released at."""
    p, ep = setup_policy(target_closing={"max_verify_steps": 2})
    p.plan(observation(ep, 0, depth=1.5))
    assert p.closing.active and not p.closing.locked
    anchor = p.closing.anchor
    p.detector.detect = lambda rgb: []
    p.plan(observation(ep, 1, depth=1.5))
    p.plan(observation(ep, 2, depth=1.5))
    assert not p.closing.active and p.closing.releases == 1
    assert p.closing.rejected[0][5] == (0.0, 0.0), "the memory keeps where the agent stood"
    # The same box from the same spot: refused, however close.
    p.detector.detect = lambda rgb: [DetectionWire("chair", .7, (280, 200, 360, 280))]
    for step in (3, 4, 5):
        p.plan(observation(ep, step, depth=1.5))
    assert not p.closing.active and p.closing.same_spot_rejections == 3 and p.closing.releases == 1
    # From 1.2 m away (beyond boxed_in_radius_m) the same object is a new view: a takeover starts.
    moved = replace(observation(ep, 6, depth=1.5), pose=AgentPose(0.0, 1.2, 0, 0))
    p.plan(moved)
    assert p.closing.active and p.closing.same_spot_rejections == 3
    assert math.dist(p.closing.anchor[:2], anchor[:2]) > 1.0, "a different anchor from the new spot"
    assert p.episode_info()["target_closing"]["rejected"][0]["from_xy"] == [0.0, 0.0]


def test_a_detector_frame_in_back_off_is_no_evidence_and_pauses_the_closing_clocks(monkeypatch):
    """A transient detector failure during a correct lock used to read as 'target not seen': the
    inspection cycled blind and released the lock with a 2 m rejection around the real target."""
    p, ep, near = locked_at_two_metres(monkeypatch, max_verify_steps=3)
    assert p.closing.locked
    started = p.closing.started

    def broken(rgb):
        raise RuntimeError("detector away")

    p.detector.detect = broken
    for step in (2, 3, 4, 5):
        command = p.plan(replace(near, step=step))
        assert p.closing.active and p.closing.locked, "no evidence either way: the lock stands"
        assert command.info.get("blind") is True
    assert p.closing.blind_frames >= 1 and p.closing.releases == 0
    assert p.closing.started >= started + 1, "the clocks moved with the blind frames"
    assert p.closing.last_seen == 5, "the consecutive-frame chain is carried across the blind frames"
    # A frame that never had a candidate stays with the search: blind frames start nothing.
    q, eq = setup_policy()
    q.detector.detect = broken
    q.plan(observation(eq, 0, depth=2))
    assert not q.closing.active and q.closing.blind_frames == 0


def test_a_suspect_candidate_may_step_inside_the_terminal_range_to_earn_its_second_viewpoint():
    """A suspect lock needs two viewpoints 0.30 m apart; VERIFY only stepped while more than 1.25 m
    away, so a suspect first seen at 1.2 m held, turned and released for ever (52 frames held)."""
    p, ep = setup_policy()
    p.plan(observation(ep, 0, depth=1.2))
    assert p.closing.active and not p.closing.locked
    p.closing.suspect = "room:kitchen"
    command = p.plan(observation(ep, 1, depth=1.2))
    assert not p.closing.locked, "two frames from one spot do not lock a suspect"
    assert command.waypoints and command.info.get("verify_step") == "towards the candidate", "a step, not a hold"
    assert command.info.get("suspect") == "room:kitchen"
    plain, ep2 = setup_policy()
    plain.plan(observation(ep2, 0, depth=1.2))
    plain.plan(observation(ep2, 1, depth=1.2))
    assert plain.closing.locked, "an unsuspected candidate locks on its second frame"


def test_a_locked_target_without_a_path_is_kept_in_frame_rather_than_turned_away_from(monkeypatch):
    """The pathless wait held the centred target with a satisfied hold, which the headless agent
    executes as a turn, and the next frame turned back: 233 reversals in one Darden episode."""
    p, ep, near = locked_at_two_metres(monkeypatch)
    monkeypatch.setattr(p.closing.path, "command", lambda *a, **kw: None)
    monkeypatch.setattr(p.camera_control, "begin_inspection", lambda *a, **kw: False)
    far = replace(observation(ep, 2, depth=1.6), pose=AgentPose(0.4, 0, 0, 0))    # 1.6 m off: not terminal
    command = p.plan(far)
    assert p.closing.locked and not command.stop
    assert command.info.get("verify_step", "").startswith(("look", "turn")), (
        "a frame that keeps the box whole, never an idle hold: %r" % (command.info,))


# -- the toilets of the 1,000-episode run (2026-10-11): the part of the object a frame shows moves -----------
def test_a_resighting_shifted_along_the_line_of_sight_is_the_same_object_and_one_shifted_across_it_is_not(monkeypatch):
    """Darden/000018 and /000193: from 2 m the depth patch landed on the toilet's bowl, from 1.3 m -- the box cut
    by the frame's lower edge -- on its tank, 0.47-0.49 m back ALONG the ray and 1.6-3.3 cm across it. A 0.50 m
    sphere about the anchor dropped every such frame (0.50-0.53 m) and the agent, standing beside the toilet,
    released it as 'saw nothing in any view'. Along the ray the gate is now the radius plus the object's depth;
    across it, as tight as before."""
    p, ep, near = locked_at_two_metres(monkeypatch)                      # anchor (2.0, 0, 0.88); the agent at (1.1, 0)
    deeper = replace(near, depth_m=np.full(near.depth_m.shape, 1.45, np.float32))   # the same box 0.55 m behind the anchor
    p.plan(deeper)
    assert p.closing.last_seen == 2, "0.55 m along the ray is the far part of the same object"
    assert p.closing.depth_associations == 1, "... and a sphere of 0.50 m would have dropped it"
    sideways = replace(near, step=3)
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (517, 200, 597, 280))]   # 0.55 m to the right at 0.9 m
    p.plan(sideways)
    assert p.closing.last_seen == 2, "0.55 m across the ray is another object"
    assert p.closing.depth_associations == 1
    with pytest.raises(ValueError):
        TargetClosingSettings(association_depth_m=-0.1)


def low_target_locked(monkeypatch, **closing):
    """A toilet-high target locked two metres ahead: rows 300-400 of the frame at 2 m put its centroid 0.31 m up
    (supported points 0.18-0.44 m), on an empty map."""
    p, ep = setup_policy(target_closing=closing) if closing else setup_policy()
    freeze_global(monkeypatch, p)
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (280, 300, 360, 400))]
    p.plan(observation(ep, 0, depth=2))
    p.plan(observation(ep, 1, depth=2))
    assert p.closing.locked and abs(p.closing.xyz[0] - 2.0) < 0.1 and 0.25 < p.closing.xyz[2] < 0.37
    assert p.closing.lowest_m < 0.25 and 0.40 < p.closing.top_m < 0.50
    return p, ep


def floor_cut_frame(ep, step, bottom, near_rows_from=372):
    """From 1.1 m: a box reaching the lower edge whose depth patch is the object's UPPER part at 1.37 m (0.42-0.80 m
    up) with the nearer, lower part (0.6 m, under the patch) filling the frame's bottom rows -- a tank over a bowl."""
    frame = replace(observation(ep, step, depth=1.37), pose=AgentPose(1.1, 0, 0, 0))
    frame.depth_m[near_rows_from:, :] = 0.6
    return frame, [DetectionWire("chair", .9, (250, 190, 390, bottom))]


def test_a_box_cut_by_the_frames_lower_edge_reads_its_terminal_range_from_the_memory_and_stops(monkeypatch):
    """The Darden toilet at 0.89 m: the fresh box's near edge was the TANK's, 1.35 m; the standoff the approach had
    reached was measured from the bowl's memory. Arrived by the memory, out of range by the measurement, released."""
    p, ep = low_target_locked(monkeypatch)
    frame, boxes = floor_cut_frame(ep, 2, bottom=480)
    p.detector.detect = lambda rgb: boxes
    before = p.closing.xyz
    command = p.plan(frame)
    assert p.closing.last_seen == 2 and p.closing.depth_associations == 1, "the tank is the toilet: +0.47 m along the ray"
    assert p.closing.visible_floor_m > p.closing.lowest_m + 0.05, "the part in view stands above what the lock has seen"
    assert p.closing.xyz == before and p.closing.floor_cut_frames == 1, "a sighting of the upper part does not move the memory"
    assert command.stop and p.closing.phase == "STOP" and command.info["reason"] == "fresh terminal target confirmation"
    assert command.info["floor_cut"] is True and command.info["measured_m"] <= 1.05
    assert command.info["measured_m"] == pytest.approx(math.dist((1.1, 0), p.closing.xyz[:2]), abs=1e-6)
    assert p.closing.observed_near_m > 1.3, "... while the visible near edge still said 'too far'"
    assert p.closing.floor_cut_measurements >= 1


def test_the_same_box_whole_in_the_frame_keeps_the_visible_near_edge_as_its_range(monkeypatch):
    """A box that ends above the lower margin hides nothing: the near edge of what is in view is the near edge,
    and the frame is fused into the memory like any other."""
    p, ep = low_target_locked(monkeypatch)
    frame, boxes = floor_cut_frame(ep, 2, bottom=460)
    p.detector.detect = lambda rgb: boxes
    before = p.closing.xyz
    command = p.plan(frame)
    assert p.closing.last_seen == 2 and not command.stop
    assert p.closing.xyz != before and p.closing.floor_cut_frames == 0, "fused: the memory moved toward the new centroid"
    assert p.closing.observed_near_m > 1.3 and p.closing.floor_cut_measurements == 0
    assert p.closing.phase in ("CLOSE", "CLOSE_OCCLUDED")
    off, ep2 = low_target_locked(monkeypatch, floor_cut_terminal=False)
    frame, boxes = floor_cut_frame(ep2, 2, bottom=480)
    off.detector.detect = lambda rgb: boxes
    assert not off.plan(frame).stop, "the switch restores the former measurement"
    with pytest.raises(ValueError):
        TargetClosingSettings(floor_cut_terminal="yes")


def test_a_low_target_under_the_frame_is_stopped_on_from_the_memory_only_without_look_actions(monkeypatch):
    """Wiconisco/000031: the 0.47 m toilet from 0.9 m shows under 40 px of its rim, the detector read 0.12-0.18,
    the yaw sweep saw 'nothing in any view' and the lock was released with the agent inside the success region.
    No LOOK_DOWN exists in this protocol; the geometry says the camera cannot show the target from here."""
    p, ep = low_target_locked(monkeypatch)
    p.detector.detect = lambda rgb: []
    gone = replace(observation(ep, 2, depth=1.6), pose=AgentPose(1.1, 0, 0, 0))     # 0.9 m from the memory
    below = p.closing._below_the_frame(gone, 0.9)
    assert below is not None and 0 < below < 0.15 * 480, "its top, 0.44 m up, would show under 72 px from 0.9 m"
    command = p.plan(gone)
    assert command.stop and p.closing.phase == "STOP" and p.closing.blind_cone_stops == 1
    assert command.info["reason"].startswith("target below the camera's field of view") and "1 action(s) ago" in command.info["reason"]
    assert command.info["below_frame_px"] == pytest.approx(below, abs=0.1)
    # The same geometry with the LOOK pair: the pitch controller's business, not this rule's.
    tilted, original = setup_policy()
    ep_tilt = tilt_episode(tilted, original)
    freeze_global(monkeypatch, tilted)
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(tilted.mapping.__dict__, "update", lambda *a, **kw: world)
    tilted.detector.detect = lambda rgb: [DetectionWire("chair", .9, (280, 300, 360, 400))]
    tilted.plan(observation(ep_tilt, 0, depth=2))
    tilted.plan(observation(ep_tilt, 1, depth=2))
    assert tilted.closing.locked
    tilted.detector.detect = lambda rgb: []
    gone_tilt = replace(observation(ep_tilt, 2, depth=1.6), pose=AgentPose(1.1, 0, 0, 0))
    assert tilted.closing._below_the_frame(gone_tilt, 0.9) is None
    command = tilted.plan(gone_tilt)
    assert not command.stop and tilted.closing.blind_cone_stops == 0
    assert command.camera_pitch is not None and command.camera_pitch > 0, "with LOOK actions the inspection looks down"


def test_the_below_the_frame_stop_needs_a_recent_sighting_an_in_range_memory_and_a_target_in_the_horizontal_field(monkeypatch):
    p, ep = low_target_locked(monkeypatch)
    p.detector.detect = lambda rgb: []
    gone = replace(observation(ep, 2, depth=1.6), pose=AgentPose(1.1, 0, 0, 0))
    assert p.closing._below_the_frame(replace(gone, pose=AgentPose(1.1, 0, 0, math.radians(60))), 0.9) is None, (
        "a target 60 degrees off the heading is a yaw matter")
    assert p.closing._below_the_frame(gone, 2.0) is None, "from 2 m the whole object is in the frame"
    stale = replace(gone, step=1 + p.settings.target_closing.blind_cone_max_missing + 1)
    command = p.plan(stale)
    assert not command.stop and p.closing.blind_cone_stops == 0, "seen too long ago: the sweep, as before"
    off, ep2 = low_target_locked(monkeypatch, blind_cone_stop=False)
    off.detector.detect = lambda rgb: []
    assert not off.plan(replace(observation(ep2, 2, depth=1.6), pose=AgentPose(1.1, 0, 0, 0))).stop
    assert off.closing.blind_cone_stops == 0
    for kwargs in ({"blind_cone_stop": 1}, {"blind_cone_max_missing": 0}, {"blind_cone_min_fraction": 1.0}):
        with pytest.raises(ValueError):
            TargetClosingSettings(**kwargs)

# -- the image-space pitch servo and the whole-box, confident STOP (2026-10-11) ---------------------------------
def low_target_locked_with_tilt(monkeypatch, **closing):
    """``low_target_locked`` under the multi-floor action spec (the LOOK pair, pitch limits -30 / +60)."""
    p, original = setup_policy(target_closing=closing) if closing else setup_policy()
    ep = tilt_episode(p, original)
    freeze_global(monkeypatch, p)
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (280, 300, 360, 400))]
    p.plan(observation(ep, 0, depth=2))
    p.plan(observation(ep, 1, depth=2))
    assert p.closing.locked and 0.25 < p.closing.xyz[2] < 0.37
    return p, ep
def test_the_approach_pitch_inside_the_servo_band_is_the_boxs_not_the_memorys(monkeypatch):
    """Within 1.5 m with a fresh box the image-space servo has the camera. A box in the lower region of the frame
    is looked DOWN at; a box in the centre band leaves the camera level even where the geometric controller
    would force a tilt (the memory 0.24 m below the camera -- outside ``elevation_band_m`` -- inside
    ``look_down_distance_m``) -- the two would otherwise hunt. Without a fresh box the geometry decides as before."""
    def locked(**closing):
        p, original = setup_policy(target_closing=closing) if closing else setup_policy()
        ep = tilt_episode(p, original)
        freeze_global(monkeypatch, p)
        world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
        monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
        p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (280, 240, 360, 330))]      # 0.24 m under the camera at 2 m
        p.plan(observation(ep, 0, depth=2))
        p.plan(observation(ep, 1, depth=2))
        assert p.closing.locked and 0.60 < p.closing.xyz[2] < 0.72
        return p, ep
    p, ep = locked()
    tilt = ep.action_spec.tilt_angle_rad
    # 1.4 m from the memory: the box of a low object sits in the lower quarter of the frame.
    low = replace(observation(ep, 2, depth=1.4), pose=AgentPose(0.6, 0, 0, 0))
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (280, 380, 360, 450))]
    command = p.plan(low)
    assert p.closing.phase == "CLOSE" and command.camera_pitch == pytest.approx(tilt)
    # Read on the frame the step will produce: already the lower region now, cut by the lower edge after it.
    assert p.closing.bbox_pitch_corrections == 1 and p.closing.last_fix.region in ("lower", "cut_bottom")
    assert p.closing.diagnostics()["bbox_region"] == p.closing.last_fix.region and p.closing.diagnostics()["bbox_pitch_steps"] == 1
    # The same memory, the box drawn in the centre band: the geometry alone would say LOOK_DOWN ...
    assert p.closing._pitch(low, 1.15) == pytest.approx(tilt), "the geometric controller forces one tilt here"
    centred = replace(observation(ep, 3, depth=1.4), pose=AgentPose(0.6, 0, 0, 0))
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (280, 200, 360, 280))]
    command = p.plan(centred)
    assert p.closing.last_seen == 3, "the centred box projects 0.24 m above the memory: a re-sighting"
    assert p.closing.phase == "CLOSE" and command.camera_pitch == pytest.approx(0.0), "... but the box is centred: the camera stays"
    assert p.closing.bbox_pitch_corrections == 1 and p.closing.last_fix.region == "centre"
    # No fresh box: the geometry's prediction, as before.
    p.detector.detect = lambda rgb: []
    occluded = replace(observation(ep, 4, depth=1.4), pose=AgentPose(0.6, 0, 0, 0))
    command = p.plan(occluded)
    assert p.closing.phase == "CLOSE_OCCLUDED" and command.camera_pitch == pytest.approx(tilt)
    # The servo switched off: the geometric controller alone decides, and the servo's counter never moves.
    off, ep_off = locked(bbox_pitch=False)
    off.detector.detect = lambda rgb: [DetectionWire("chair", .9, (280, 380, 360, 450))]
    command = off.plan(replace(observation(ep_off, 2, depth=1.4), pose=AgentPose(0.6, 0, 0, 0)))
    assert off.closing.last_seen == 2 and command.camera_pitch == pytest.approx(tilt)
    assert off.closing.bbox_pitch_corrections == 0 and off.closing.last_fix is None


def test_the_geometric_pitch_keeps_the_present_step_when_the_wanted_angle_sits_at_half_a_step(monkeypatch):
    """Darden/000034 (2026-10-14): a 0.39 m toilet 2.19 m away, the camera 0.9 m up -- 15.0 degrees down wanted
    against a 30-degree step. Rounding gave 0 on one frame and 30 on the next as the memory's height moved by
    millimetres, and nine LOOK_DOWN / LOOK_UP actions were spent without a step."""
    p, original = setup_policy()
    ep = tilt_episode(p, original)
    tilt = ep.action_spec.tilt_angle_rad
    camera_z = ep.camera.height_m                                   # the pose's z is 0 in this fixture
    ahead = 1.94
    half = camera_z - ahead * math.tan(0.5 * tilt)                 # the height at which exactly half a step is wanted
    level = replace(observation(ep, 0), pose=AgentPose(0, 0, 0, 0, camera_pitch=0.0))
    down = replace(observation(ep, 0), pose=AgentPose(0, 0, 0, 0, camera_pitch=tilt))
    for dz in (-0.005, 0.0, 0.005):                                # the memory flickers about the boundary
        p.closing.xyz = (ahead, 0.0, half + dz)
        assert p.closing._pitch(level, ahead) == pytest.approx(0.0), "level camera: stays level"
        assert p.closing._pitch(down, ahead) == pytest.approx(tilt), "camera a step down: stays a step down"
    # A change wanted by more than the band still happens, in either direction.
    p.closing.xyz = (ahead, 0.0, camera_z - ahead * math.tan(math.radians(40)))      # 40 degrees down wanted
    assert p.closing._pitch(level, ahead) == pytest.approx(tilt)
    p.closing.xyz = (ahead, 0.0, camera_z)                                           # level wanted
    assert p.closing._pitch(down, ahead) == pytest.approx(0.0)
    # Without the band the old rounding is back, and the half-step case chatters with the frame.
    p.closing.settings = replace(p.closing.settings, pitch_hysteresis=0.0)
    p.closing.xyz = (ahead, 0.0, half - 0.005)
    assert p.closing._pitch(level, ahead) == pytest.approx(tilt)
    p.closing.xyz = (ahead, 0.0, half + 0.005)
    assert p.closing._pitch(down, ahead) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        replace(p.closing.settings, pitch_hysteresis=0.5)


def test_the_servo_predicts_the_step_and_looks_down_before_the_box_leaves_through_the_bottom(monkeypatch):
    """From 1.45 m a low box sits whole at rows 300-440; one 0.25 m step would cut it at the lower edge. The step
    carries LOOK_DOWN -- read off the predicted frame, not the memory."""
    p, ep = low_target_locked_with_tilt(monkeypatch)
    tilt = ep.action_spec.tilt_angle_rad
    frame = replace(observation(ep, 2, depth=1.45), pose=AgentPose(0.55, 0, 0, 0))
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (280, 300, 360, 440))]
    command = p.plan(frame)
    assert p.closing.phase == "CLOSE" and command.camera_pitch == pytest.approx(tilt)
    assert p.closing.last_fix.region == "cut_bottom" and "after the step" in p.closing.last_fix.reason
def test_the_terminal_stop_centres_a_cut_box_with_one_tilt_and_stops_on_the_whole_box(monkeypatch):
    """A fresh in-range box cut by the lower edge: one LOOK_DOWN, then the STOP on the whole box the next frame
    (``stop_centring_tilts`` 1, the STOP's ``bbox_region`` centre). The tilt is spent once per inspection, and
    ``stop_centring=False`` restores the STOP at whatever pitch the box came."""
    p, ep = low_target_locked_with_tilt(monkeypatch)
    tilt = ep.action_spec.tilt_angle_rad
    near = replace(observation(ep, 2, depth=0.9), pose=AgentPose(1.1, 0, 0, 0))
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (250, 300, 390, 480))]
    command = p.plan(near)
    assert not command.stop and p.closing.phase == "INSPECT" and p.closing.stop_centring_tilts == 1
    assert command.camera_pitch == pytest.approx(tilt) and command.final_yaw == pytest.approx(0.0)
    assert "one centring tilt before the terminal STOP" in command.info["reason"] and command.info["bbox_region"] == "cut_bottom"
    assert DiscreteActionConverter(ep.action_spec).step(near.pose, command).action == DiscreteAction.LOOK_DOWN
    pitched = replace(observation(ep, 3, depth=0.9), pose=AgentPose(1.1, 0, 0, 0, camera_pitch=tilt))
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (250, 150, 390, 330))]
    command = p.plan(pitched)
    assert command.stop and command.info["reason"] == "fresh terminal target confirmation"
    assert command.info["bbox_region"] == "centre" and command.info["confidence"] == pytest.approx(0.9)
    assert command.info["depth_valid"] is True and command.info["measured_m"] is not None
    assert p.closing.stop_centring_tilts == 1
    off, ep_off = low_target_locked_with_tilt(monkeypatch, stop_centring=False)
    off.detector.detect = lambda rgb: [DetectionWire("chair", .9, (250, 300, 390, 480))]
    command = off.plan(replace(observation(ep_off, 2, depth=0.9), pose=AgentPose(1.1, 0, 0, 0)))
    assert command.stop and off.closing.stop_centring_tilts == 0
def test_the_centring_tilt_is_spent_once_and_a_still_cut_box_stops_the_next_frame(monkeypatch):
    """If the box is still cut after the one centring tilt (a taller object than predicted), the STOP is not
    deferred again: the tilt is once per inspection."""
    p, ep = low_target_locked_with_tilt(monkeypatch)
    tilt = ep.action_spec.tilt_angle_rad
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (250, 300, 390, 480))]
    first = p.plan(replace(observation(ep, 2, depth=0.9), pose=AgentPose(1.1, 0, 0, 0)))
    assert not first.stop and p.closing.stop_centring_tilts == 1
    second = p.plan(replace(observation(ep, 3, depth=0.9), pose=AgentPose(1.1, 0, 0, 0, camera_pitch=tilt)))
    assert second.stop and second.info["reason"] == "fresh terminal target confirmation" and p.closing.stop_centring_tilts == 1
def test_a_fresh_in_range_box_under_stop_confidence_is_held_for_a_bounded_number_of_frames(monkeypatch):
    """Detect high, associate low, STOP high: a 0.40 re-sighting of the locked target at terminal range does not
    end the episode on its own. The closing takes the frame that keeps the box whole for two frames
    (``weak_stop_max_holds``), then STOPs on the evidence there is; a 0.9 box STOPs at once."""
    p, ep = low_target_locked_with_tilt(monkeypatch)
    tilt = ep.action_spec.tilt_angle_rad
    weak = lambda rgb: [DetectionWire("chair", .4, (250, 150, 390, 330))]      # whole and centred: nothing to re-frame
    p.detector.detect = weak
    first = p.plan(replace(observation(ep, 2, depth=0.9), pose=AgentPose(1.1, 0, 0, 0, camera_pitch=tilt)))
    assert not first.stop and p.closing.weak_stop_holds == 1 and "under stop_confidence" in first.info["reason"]
    assert first.info["confidence"] == pytest.approx(0.4) and first.info.get("verify_step")
    second = p.plan(replace(observation(ep, 3, depth=0.9), pose=AgentPose(1.1, 0, 0, 0, camera_pitch=tilt)))
    assert not second.stop and p.closing.weak_stop_holds == 2
    third = p.plan(replace(observation(ep, 4, depth=0.9), pose=AgentPose(1.1, 0, 0, 0, camera_pitch=tilt)))
    assert third.stop and third.info["reason"] == "fresh terminal target confirmation" and p.closing.weak_stop_holds == 2
    strong, ep2 = low_target_locked_with_tilt(monkeypatch)
    strong.detector.detect = lambda rgb: [DetectionWire("chair", .9, (250, 150, 390, 330))]
    command = strong.plan(replace(observation(ep2, 2, depth=0.9), pose=AgentPose(1.1, 0, 0, 0, camera_pitch=tilt)))
    assert command.stop and strong.closing.weak_stop_holds == 0
    with pytest.raises(ValueError):
        TargetClosingSettings(stop_confidence=0.2)                           # under track_confidence
    for kwargs in ({"weak_stop_max_holds": -1}, {"bbox_pitch": 1}, {"bbox_pitch_region": 0.5},
                   {"bbox_pitch_distance_m": 0.0}, {"stop_centring": "yes"}):
        with pytest.raises(ValueError):
            TargetClosingSettings(**kwargs)
def test_without_look_actions_the_servo_has_no_say_and_a_weak_box_with_no_frame_to_keep_it_stops_at_once(monkeypatch):
    """The published Gibson protocol: no LOOK pair, so neither the servo nor the centring tilt exist; the fresh
    bottom-cut toilet frame STOPs exactly as the floor-cut rule says. A weak box that no TURN keeps whole is
    not held either -- there is no better frame to wait for."""
    p, ep = low_target_locked(monkeypatch)
    assert not ep.action_spec.has_camera_tilt
    frame, boxes = floor_cut_frame(ep, 2, bottom=480)
    p.detector.detect = lambda rgb: boxes
    command = p.plan(frame)
    assert command.stop and command.info["reason"] == "fresh terminal target confirmation"
    assert p.closing.last_fix is None and p.closing.stop_centring_tilts == 0 and p.closing.bbox_pitch_corrections == 0
    q, ep2 = low_target_locked(monkeypatch)
    frame, _ = floor_cut_frame(ep2, 2, bottom=480)
    q.detector.detect = lambda rgb: [DetectionWire("chair", .4, (0, 190, 640, 480))]          # fills the frame: no turn keeps it
    command = q.plan(frame)
    assert command.stop and q.closing.weak_stop_holds == 0 and command.info["confidence"] == pytest.approx(0.4)
# -- an arrived standoff is terminal (Collierville/000194 and /000043, 2026-10-11) ---------------------------------
def arrived_at_a_kept_standoff(monkeypatch, **closing):
    """A chair locked two metres ahead (standoff goal at x = 1.35); the memory then refined to 2.3 m -- the goal,
    0.95 m from it, is kept -- and the agent stands at (1.15, 0): 0.20 m from the goal (inside the executor's
    0.25 m arrival tolerance) and 1.15 m from the memory, beyond ``terminal_distance_m``."""
    from sparx_agency.core.mapping.tracking.target_memory_3d import TargetMemory3D
    p, ep, _ = locked_at_two_metres(monkeypatch, **closing) if closing else locked_at_two_metres(monkeypatch)
    goal = p.closing.path.goal_xyz
    assert goal is not None and abs(goal[0] - 1.35) < 0.05
    s = p.settings.target_closing
    p.closing.memory = TargetMemory3D((2.3, 0.0, 0.88), 2.3, 1, s.memory_noise_floor_m, s.memory_noise_per_m)
    p.closing.xyz = p.closing.memory.xyz
    p.detector.detect = lambda rgb: []
    return p, ep, goal
def test_an_arrived_standoff_begins_the_inspection_instead_of_spinning_on_the_idle_result(monkeypatch):
    """Collierville/000194: at a kept standoff 1.15 m from the memory the route was walked, the executor said
    'arrived', and the headless agent turned left twenty times until the stall clock released the lock with the
    television in front of it. The arrival is read with the route memory's own rule the action the route completes,
    and the inspection begins from the standoff: no idle turn, no 'stalled' release."""
    p, ep, goal = arrived_at_a_kept_standoff(monkeypatch)
    at_standoff = replace(observation(ep, 2, depth=1.6), pose=AgentPose(1.15, 0, 0, 0))
    command = p.plan(at_standoff)
    assert p.closing.path.kept == 1 and p.closing.path.goal_xyz == goal, "the refined memory kept the standoff"
    assert p.closing.path.arrived and p.closing.path.arrivals == 1, "the route to it is walked"
    assert p.closing.phase == "INSPECT" and not command.waypoints and not command.stop, "the inspection, not the route"
    assert p.closing.diagnostics()["standoff_arrived"] is True and p.closing.stall_releases == 0
    for step in range(3, 8):
        command = p.plan(replace(observation(ep, step, depth=1.6), pose=AgentPose(1.15, 0, 0, 0)))
        assert p.closing.phase == "INSPECT" and not command.stop and p.closing.active, "the sweep, not a release"
    assert p.closing.stall_releases == 0 and len(p.closing.close_trail) <= 1, (
        "the stall clock never ran at the standoff (the one entry is the fixture's own CLOSE step)")
def test_an_arrived_standoff_lets_the_approach_history_stop_within_the_arrival_tolerance(monkeypatch):
    """The memory STOP of a tracked target measured ``terminal + range_tolerance`` from the memory; from an
    arrived standoff the executor's ``goal_tolerance_m`` is part of where the agent stands, so 1.15 m completes
    the trajectory where 1.05 m did not -- the agent of Collierville/000194 stood inside the success region."""
    p, ep, goal = arrived_at_a_kept_standoff(monkeypatch, approach_history_m=0.5, approach_history_closed_m=0.5)
    command = None
    for step in range(2, 7):
        command = p.plan(replace(observation(ep, step, depth=1.6), pose=AgentPose(1.15, 0, 0, 0)))
        assert p.closing._history_qualifies(), "1.15 m walked, 0.85 m closed (recorded by the first frame here)"
        if command.stop:
            break
    assert command.stop and command.info["standoff_reached"] is True and p.closing.history_stops == 1
    assert "the standoff is reached" in command.info["reason"] and command.info["range_m"] == pytest.approx(1.15, abs=0.02)
    # The same frames without the arrival: 1.15 m is beyond the memory STOP's reach, as before.
    q, ep2, _ = arrived_at_a_kept_standoff(monkeypatch, approach_history_m=0.5, approach_history_closed_m=0.5)
    q.closing.path.drop()
    monkeypatch.setattr(q.closing.path, "command", lambda *a, **kw: None)          # no standoff at all: the pathless wait
    monkeypatch.setattr(q.camera_control, "begin_inspection", lambda *a, **kw: False)
    for step in range(2, 7):
        command = q.plan(replace(observation(ep2, step, depth=1.6), pose=AgentPose(1.15, 0, 0, 0)))
    assert not command.stop and q.closing.history_stops == 0
def test_a_fresh_range_still_too_far_from_an_arrived_standoff_drops_it_for_a_closer_one(monkeypatch):
    """From the standoff the fresh box measures 1.3 m: the inspection resumes the approach -- and walking the arrived
    route again would arrive again, so the standoff is dropped and the next selection is a closer ring about the
    memory: a route with waypoints, not an idle turn."""
    p, ep, goal = arrived_at_a_kept_standoff(monkeypatch)
    p.plan(replace(observation(ep, 2, depth=1.6), pose=AgentPose(1.15, 0, 0, 0)))
    assert p.closing.phase == "INSPECT" and p.closing.path.arrived
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (280, 200, 360, 280))]       # the box, 1.3 m off
    command = p.plan(replace(observation(ep, 3, depth=1.3), pose=AgentPose(1.15, 0, 0, 0)))
    assert p.closing.inspection_resumptions == 1 and p.closing.path.dropped == 1 and not p.closing.path.arrived
    assert p.closing.path.goal_xyz is not None and p.closing.path.goal_xyz != goal, "a closer standoff was selected"
    assert math.dist(p.closing.path.goal_xyz[:2], p.closing.xyz[:2]) < math.dist(goal[:2], p.closing.xyz[:2])
    assert command.waypoints and p.closing.phase in ("CLOSE", "CLOSE_OCCLUDED"), "the approach walks on"
    assert p.closing.diagnostics()["standoff_dropped"] == 1


# -- the toilet re-run of 2026-10-13: four ways a seen target was not locked or not STOPped on (2026-10-14) ----
def test_a_resighting_of_the_active_candidate_counts_past_the_rejection_memory():
    """Markleeville/000188, actions 34-45: a takeover begun on a 0.84 override past an 'unverified' memory had
    every 0.60-0.79 re-sighting of the same toilet dropped BY that memory, and the consecutive count never
    reached two. The memory judges whether a takeover may start, not whether the one in force counts its frame."""
    p, ep = setup_policy(target_closing={"max_verify_steps": 2, "release_cooldown_actions": 10})
    p.plan(observation(ep, 0, depth=3.5))
    p.detector.detect = lambda rgb: []
    p.plan(observation(ep, 1, depth=3.5))
    p.plan(observation(ep, 2, depth=3.5))
    assert not p.closing.active and p.closing.last_release == "unverified"
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (280, 200, 360, 280))]
    p.plan(observation(ep, 3, depth=3.5))
    assert p.closing.active and p.closing.overrides == 1 and p.closing.count == 1
    p.detector.detect = lambda rgb: [DetectionWire("chair", .7, (280, 200, 360, 280))]     # the same spot, under the override
    p.plan(observation(ep, 4, depth=3.5))
    assert p.closing.locked and p.closing.count == 2
    assert p.closing.resighting_exemptions == 1 and p.episode_info()["target_closing"]["resighting_exemptions"] == 1


def test_a_depthless_box_on_the_candidates_bearing_is_approached_and_carries_the_chain():
    """Markleeville/000188, actions 20-31: the toilet at 5 m read 0.62 with depth at the frame's edge and 0.49
    without depth once centred; the sweep turned away from the centred frame every other action, twice through
    the budget. A box of the class on the anchor's bearing without a usable depth is the candidate in view:
    face it, step toward it, carry the chain across the frame, and do not count the step against the clock."""
    p, ep = setup_policy()
    p.plan(observation(ep, 0, depth=3.5))                               # the anchor 3.5 m ahead, count 1
    assert p.closing.active and p.closing.count == 1
    started = p.closing.started
    glimpse = p.plan(observation(ep, 1, depth=float("nan")))            # the same box, no usable depth
    assert glimpse.info.get("verify_step") == "towards the glimpsed candidate" and glimpse.waypoints
    assert glimpse.info["target_visible"] is True and glimpse.info["target_measured"] is False
    assert p.closing.count == 1 and p.closing.glimpse_steps == 1 and p.closing.started == started + 1
    assert DiscreteActionConverter(ep.action_spec).step(AgentPose(0, 0, 0, 0), glimpse).action == DiscreteAction.MOVE_FORWARD
    p.plan(observation(ep, 2, depth=3.25))                              # measured again: the second consecutive frame
    assert p.closing.locked and p.closing.count == 2 and p.closing.verify_pauses == 1
    # A box of the class elsewhere in the frame, without depth, is not a glimpse of this candidate.
    q, _ = setup_policy()
    q.plan(observation(ep, 0, depth=3.5))
    q.detector.detect = lambda rgb: [DetectionWire("chair", .9, (20, 200, 100, 280))]    # 34 degrees off the bearing
    sweep = q.plan(observation(ep, 1, depth=float("nan")))
    assert sweep.info["target_visible"] is False and q.closing.count == 0 and q.closing.glimpse_steps == 0
    with pytest.raises(ValueError):
        TargetClosingSettings(max_verify_approach_steps=-1)


def test_the_inspection_steps_when_the_fresh_range_is_a_step_beyond_the_limit_and_does_not_oscillate(monkeypatch):
    """Wiconisco/000193, actions 476-499: the toilet's near edge at 1.0505 m against the 1.05 m limit, its box
    16-17 degrees off the centre on alternating sides, and twenty-four TURN_LEFT/TURN_RIGHT at the step limit.
    A turn back that would undo the last turn is the dead band, not a correction; and a fresh range one step
    out, faced within a turn, is one MOVE_FORWARD from the STOP."""
    p, ep, goal = arrived_at_a_kept_standoff(monkeypatch)
    here = AgentPose(1.15, 0, 0, 0)
    k = ep.camera.intrinsics
    right = int(k.cx + k.fx * math.tan(math.radians(17)))             # 17 degrees right of the centre
    left = int(k.cx - k.fx * math.tan(math.radians(17)))
    boxes = {"right": [DetectionWire("chair", .9, (right - 40, 200, right + 40, 280))],
             "left": [DetectionWire("chair", .9, (left - 40, 200, left + 40, 280))]}

    def frame(step, depth, side=None):
        p.detector.detect = lambda rgb: boxes[side] if side else []
        return replace(observation(ep, step, depth=depth), pose=here)
    converter = DiscreteActionConverter(ep.action_spec)
    p.plan(frame(2, 1.6))
    assert p.closing.phase == "INSPECT", "the arrived standoff: the inspection begins"
    # In range, 17 degrees right: the centring turn.
    turning = p.plan(frame(3, 0.97, "right"))
    assert not turning.stop and turning.waypoints == () and converter.step(here, turning).action == DiscreteAction.TURN_RIGHT
    # A step out of range, 17 degrees LEFT: a turn back would recreate the frame before (the dead band), and the
    # range says one step -- not another turn.
    stepping = p.plan(frame(4, 1.10, "left"))
    assert not stepping.stop and stepping.waypoints and "one step" in stepping.info["inspect_step"]
    assert p.closing.dead_band_alignments == 1 and p.closing.inspection_steps == 1
    assert converter.step(here, stepping).action == DiscreteAction.MOVE_FORWARD
    assert p.closing.inspection_started is not None and p.closing.inspection_resumptions == 0, "still the inspection"
    # In range, 17 degrees right again (the chain of turns broke at the step): a turn ...
    turning = p.plan(frame(5, 0.97, "right"))
    assert not turning.stop and converter.step(here, turning).action == DiscreteAction.TURN_RIGHT
    # ... and 17 degrees left, in range: aligned by the dead band, the STOP.
    stopping = p.plan(frame(6, 0.97, "left"))
    assert stopping.stop and p.closing.dead_band_alignments == 2
    assert p.episode_info()["target_closing"]["dead_band_alignments"] == 2
    with pytest.raises(ValueError):
        TargetClosingSettings(inspection_max_steps=-1)


def test_a_locked_target_with_no_route_on_the_known_floor_is_walked_to_through_the_unknown(monkeypatch):
    """Collierville/000023, action 375: the toilet locked at 2.8 m in plain view, the floor between the corridor
    and its doorway unknown (a level camera sees the floor from 1.4 m out), no standoff route, a footing sweep,
    and the lock released as boxed in -- the spot then refused the 0.82-0.88 re-sightings. With no route on
    the known floor the ring is planned through the unknown; the committed route is re-checked as the map fills."""
    p, ep = setup_policy()
    freeze_global(monkeypatch, p)
    # A band of unknown floor across the whole map between the agent (x=0) and the target (x=2).
    data = np.zeros((200, 200), np.int8)
    params = OccupancyGrid2DParams(.1, -10, -10)
    gx0, _ = OccupancyGrid2D(data, params).world_to_grid(0.6, 0.0)
    gx1, _ = OccupancyGrid2D(data, params).world_to_grid(1.3, 0.0)
    data[:, gx0:gx1] = -1
    world = OccupancyGrid2D(data, params)
    assert int((world.grid == world.values.unknown).sum()) == 200 * (gx1 - gx0)
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    p.plan(observation(ep, 0, depth=2))
    command = p.plan(observation(ep, 1, depth=2))
    assert p.closing.locked and abs(p.closing.xyz[0] - 2.0) < 0.1
    assert command.waypoints and command.info.get("through_unknown") is True
    assert p.closing.phase in ("CLOSE", "CLOSE_OCCLUDED") and p.closing.through_unknown_routes == 1
    assert p.closing.path.through_unknown == 1 and p.closing.no_path_steps == 0 and p.closing.footing_sweeps == 0
    assert any(gx0 <= world.world_to_grid(x, y)[0] < gx1 for x, y in command.waypoints), "the route crosses the band"
    assert p.episode_info()["target_closing"]["through_unknown_routes"] == 1
    # Switched off: the pathless wait as before.
    q, ep2 = setup_policy(target_closing={"through_unknown": False})
    freeze_global(monkeypatch, q)
    monkeypatch.setitem(q.mapping.__dict__, "update", lambda *a, **kw: world)
    q.plan(observation(ep2, 0, depth=2))
    pathless = q.plan(observation(ep2, 1, depth=2))
    assert q.closing.locked and pathless.waypoints == () and q.closing.no_path_steps == 1
    assert q.closing.through_unknown_routes == 0
    with pytest.raises(ValueError):
        TargetClosingSettings(through_unknown_cost=0.5)


def test_a_fruitless_sweep_cycle_releases_the_verification_before_its_budget():
    """Wiconisco/000014, actions 231-241: the toilet seen once at 3.2 m, one step toward it, and eleven in-place
    turns at the bearing and one turn to each side -- the step had put the toilet behind a jamb, and the same three
    headings cannot bring back what none of them shows. After ``verify_sweep_max_actions`` (6) sweep actions without
    a frame of the candidate the verification is released as at its budget; the rest of the budget is the search's."""
    p, ep = setup_policy()
    p.plan(observation(ep, 0, depth=3.5))
    assert p.closing.active and p.closing.count == 1
    p.detector.detect = lambda rgb: []
    step = 1
    while p.closing.active and step < 12:
        command = p.plan(observation(ep, step, depth=3.5))
        if p.closing.active:
            assert command.info["target_visible"] is False and command.info["sweep_action"] == step
        step += 1
    assert not p.closing.active and p.closing.last_release == "unverified" and p.closing.sweep_releases == 1
    assert step == 8, "six sweep actions, the release on the seventh frame: five actions short of the twelve"
    assert p.episode_info()["target_closing"]["sweep_releases"] == 1
    # Off: the budget alone ends the sweep, as before.
    q, _ = setup_policy(target_closing={"verify_sweep_max_actions": 0})
    q.plan(observation(ep, 0, depth=3.5))
    q.detector.detect = lambda rgb: []
    for step in range(1, 12):
        q.plan(observation(ep, step, depth=3.5))
        assert q.closing.active
    q.plan(observation(ep, 12, depth=3.5))
    assert not q.closing.active and q.closing.sweep_releases == 0
    with pytest.raises(ValueError):
        TargetClosingSettings(verify_sweep_max_actions=-1)
# ── the standoff of a big object (Collierville/000009, 2026-10-15) ───────────────────────────────────
def locked_at_three_metres(monkeypatch, **closing):
    """A chair locked three metres ahead on an empty map; the standoff the lock frames chose is dropped, so the
    next action selects the ring anew under the test's own projector."""
    p, ep = setup_policy(target_closing=closing) if closing else setup_policy()
    freeze_global(monkeypatch, p)
    world = OccupancyGrid2D(np.zeros((200, 200), np.int8), OccupancyGrid2DParams(.1, -10, -10))
    monkeypatch.setitem(p.mapping.__dict__, "update", lambda *a, **kw: world)
    p.plan(observation(ep, 0, depth=3))
    p.plan(observation(ep, 1, depth=3))
    assert p.closing.locked and abs(p.closing.xyz[0] - 3.0) < 0.1 and abs(p.closing.xyz[1]) < 0.1
    p.closing.path.drop()
    p.closing.path.dropped = 0
    return p, ep, world
def test_the_standoff_ring_widens_to_the_terminal_radius_when_no_standoff_point_has_a_route(monkeypatch):
    """A 1.5 m television on its cabinet: every 0.70 m ring point inside the cabinet, against it or behind it, in
    both planners, from 3 m away -- three correct locks released as 'no safe path from here' with the television
    in view. Wider rings in the front arc, out to the terminal radius, find the floor the standoff ring has not."""
    p, ep, world = locked_at_three_metres(monkeypatch)
    memory = tuple(p.closing.xyz)
    refused = []
    def cabinet(point):                                       # no floor within 0.9 m of the memory
        if math.dist(point[:2], memory[:2]) < 0.9:
            refused.append(point)
            return None
        return point
    p.target_projector = cabinet
    command = p.plan(observation(ep, 2, depth=3))
    assert command.waypoints and not command.stop and p.closing.phase == "CLOSE" and p.closing.no_path_steps == 0
    path = p.closing.path
    assert path.widened and path.widenings == 1 and p.closing.widened_standoffs == 1
    assert 0.9 <= math.dist(path.goal_xyz[:2], memory[:2]) <= 1.0 + world.resolution
    assert len(refused) >= 16, "the standoff ring was tried first, at every bearing, with both planners"
    assert math.dist(command.waypoints[-1], memory[:2]) <= 1.0 + world.resolution
    assert p.closing.diagnostics()["standoff_widened"] is True and p.closing.diagnostics()["widened_standoffs"] == 1
    # A standoff ring point with floor is still preferred: the widening is only for when it has none.
    q, ep, _ = locked_at_three_metres(monkeypatch)
    q.target_projector = lambda point: point
    q.plan(observation(ep, 2, depth=3))
    assert not q.closing.path.widened and q.closing.widened_standoffs == 0
    assert math.dist(q.closing.path.goal_xyz[:2], q.closing.xyz[:2]) <= 0.75
def test_a_goal_a_refinement_makes_stale_is_kept_when_no_ring_point_has_a_route(monkeypatch):
    """Collierville/000009, action 15: the standoff stood 1.01 m from the refined memory with a seventeen-waypoint
    route to it; it was dropped as stale for a ring that had no route at all, and the lock was released six
    pathless actions later. Kept over no goal, while it lies within the terminal radius plus the tolerance."""
    p, ep, _ = locked_at_three_metres(monkeypatch)
    memory = tuple(p.closing.xyz)
    path = p.closing.path
    path.goal_xyz, path.reference_xyz = (memory[0] - 1.01, memory[1], 0.0), memory
    p.target_projector = lambda point: None                   # no ring point has floor, at any radius
    command = p.plan(observation(ep, 2, depth=3))
    assert command.waypoints and p.closing.phase == "CLOSE" and p.closing.no_path_steps == 0
    assert path.goal_xyz[:2] == pytest.approx((memory[0] - 1.01, memory[1]), abs=0.02) and path.widened
    assert path.stale_kept == 1 and p.closing.stale_standoffs_kept == 1 and path.refinements == 1
    assert p.closing.diagnostics()["stale_standoffs_kept"] == 1
    # Beyond the terminal radius plus the arrival tolerance the stale goal is not kept: no path, as before.
    path.goal_xyz, path.reference_xyz, path.widened = (memory[0] - 1.3, memory[1], 0.0), memory, False
    command = p.plan(observation(ep, 3, depth=3))
    assert p.closing.no_path_steps == 1 and p.closing.phase == "REACQUIRE" and not command.waypoints
    assert command.info["kind"] == "target_closing" and command.info["target_visible"] is True
    assert path.goal_xyz is None and path.stale_kept == 1 and not path.widened
def arrived_widened_standoff(monkeypatch, widened=True, **closing):
    """The agent at an ARRIVED standoff about 0.95 m from the memory -- found on the widened ring unless told
    otherwise, no closer ring point with floor -- standing 1.1 m from the chair, which fills the frame fresh,
    centred, at 1.2 m (a centred box keeps the refinement on the line of sight, so the standoff stays inside the
    terminal radius and is not made stale by it)."""
    p, ep, _ = locked_at_three_metres(monkeypatch, **closing)
    path = p.closing.path
    path.goal_xyz, path.reference_xyz, path.widened = (2.2, 0.0, 0.0), tuple(p.closing.xyz), widened
    p.target_projector = lambda point: None
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (124, 25, 516, 404))]
    return p, ep, replace(observation(ep, 2, depth=1.2), pose=AgentPose(2.05, 0, 0, 0))
def test_from_a_widened_standoff_the_inspection_steps_and_stops_when_the_step_is_blocked(monkeypatch):
    """The cabinet: the inspection's step toward a fresh range 1.2 m against a 1.05 m limit does not move the agent.
    As close as the floor allows, the target fresh in view: STOP, not 'the inspection saw nothing'."""
    p, ep, frame = arrived_widened_standoff(monkeypatch)
    command = p.plan(frame)
    assert p.closing.path.arrived and p.closing.phase == "INSPECT" and p.closing.inspection_resumptions == 0
    assert not command.stop and command.waypoints and "one step" in command.info["inspect_step"]
    assert p.closing.inspection_stepped == 1 and p.closing.inspection_step_pending
    p.notify_action(frame, DiscreteAction.MOVE_FORWARD)
    blocked = replace(observation(ep, 3, depth=1.2), pose=frame.pose)
    p.notify_blocked(blocked)
    assert p.closing.inspection_blocked and not p.closing.inspection_step_pending
    command = p.plan(blocked)
    assert command.stop and p.closing.closest_approach_stops == 1 and p.closing.inspection_releases == 0
    assert "closest reachable approach" in command.info["reason"] and "blocked" in command.info["reason"]
    assert command.info["measured_m"] == pytest.approx(1.2, abs=0.05) and command.info["target_visible"] is True
    assert p.closing.diagnostics()["closest_approach_stops"] == 1 and p.closing.rejected == []
def test_from_a_widened_standoff_the_inspection_stops_once_its_steps_are_spent(monkeypatch):
    p, ep, frame = arrived_widened_standoff(monkeypatch)
    assert not p.plan(frame).stop and p.closing.inspection_stepped == 1
    p.notify_action(frame, DiscreteAction.MOVE_FORWARD)                      # the step moved the agent: not blocked
    p.closing.inspection_stepped = TargetClosingSettings().inspection_max_steps
    command = p.plan(replace(observation(ep, 3, depth=1.2), pose=frame.pose))
    assert not p.closing.inspection_blocked
    assert command.stop and p.closing.closest_approach_stops == 1 and "steps are spent" in command.info["reason"]
def test_the_closest_approach_stop_needs_a_widened_standoff_and_can_be_switched_off(monkeypatch):
    """From a standoff on the ring itself a fresh range beyond the limit is what it always was: a reason to step
    closer, never a STOP; and the switch restores that for a widened standoff too."""
    p, ep, frame = arrived_widened_standoff(monkeypatch, widened=False)
    command = p.plan(frame)
    assert not command.stop and p.closing.phase in ("CLOSE", "CLOSE_OCCLUDED") and p.closing.closest_approach_stops == 0
    assert p.closing.inspection_started is None, "a standoff on the ring, out of range: no inspection from here"
    q, ep, frame = arrived_widened_standoff(monkeypatch, closest_approach_stop=False)
    command = q.plan(frame)
    assert not command.stop and q.closing.inspection_started is None and q.closing.closest_approach_stops == 0
    with pytest.raises(ValueError):
        TargetClosingSettings(closest_approach_stop=1)
def test_a_pending_inspection_step_does_not_outlive_its_frame(monkeypatch):
    """The blocked hook speaks before the next frame; a block reported later belongs to another command."""
    p, ep, frame = arrived_widened_standoff(monkeypatch)
    p.plan(frame)
    assert p.closing.inspection_step_pending
    p.notify_action(frame, DiscreteAction.TURN_LEFT)                         # the converter turned: no step was tried
    assert not p.closing.inspection_step_pending
    again = replace(observation(ep, 3, depth=1.2), pose=frame.pose)
    p.plan(again)
    assert p.closing.inspection_step_pending and p.closing.inspection_stepped == 2, "the next frame asks for the step again"
    p.notify_action(again, DiscreteAction.MOVE_FORWARD)
    later = replace(observation(ep, 4, depth=1.2), pose=frame.pose)
    p.closing.observe(later)
    assert not p.closing.inspection_step_pending
    p.closing.note_blocked()
    assert not p.closing.inspection_blocked, "a block after the frame changed is not the inspection step's"
