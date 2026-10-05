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
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.target_closing import TargetClosingSettings
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
    p.plan(observation(ep, 3, depth=3.4))  # still associated, meaningful refinement
    assert p.closing.path.goal_xyz != goal and p.closing.path.refinements == 1


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
    sim = HabitatRGBDSimulator(None, None, .18, True)
    sim._sim = SimpleNamespace(pathfinder=SimpleNamespace(snap_point=lambda point: snap))
    assert sim.project_target((1, 2, 3)) is None


def test_navmesh_round_trip_enu_native_coordinates():
    received = []
    sim = HabitatRGBDSimulator(None, None, .18, True)
    sim._sim = SimpleNamespace(pathfinder=SimpleNamespace(snap_point=lambda point: received.append(point) or point))
    assert sim.project_target((1, 2, 3)) == (1, 2, 3)
    np.testing.assert_allclose(received[0], [-2, 3, -1])


@pytest.mark.parametrize("kwargs", [{"confirmation_frames": 1}, {"confirmation_frames": True},
                                    {"confidence": float("nan")}, {"confidence": 0}, {"max_verify_steps": 0},
                                    {"border_margin_px": -1}, {"border_margin_px": 2.0},
                                    {"release_unverified": 1}, {"rejection_radius_m": 0}])
def test_invalid_closing_settings(kwargs):
    with pytest.raises(ValueError):
        TargetClosingSettings(**kwargs)


@pytest.mark.parametrize("box", [(0, 269, 84, 479), (300, 0, 360, 80), (560, 200, 640, 280), (300, 400, 360, 480)])
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
    # The same spot flickering again must not start a second takeover...
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (280, 200, 360, 280))]
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


def test_a_target_box_on_a_confirmed_other_object_is_outvoted_and_starts_nothing():
    """Five frames of a bed where the box now says chair: the map says bed, the takeover stays off."""
    p, ep = setup_policy()
    p.detector.detect = lambda rgb: [DetectionWire("bed", .9, (280, 200, 360, 280))]
    for step in range(5):
        p.plan(observation(ep, step, depth=3))
    [bed] = p.landmarks.confirmed()
    assert bed.class_name == "bed" and bed.votes == {"bed": 5}
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (280, 200, 360, 280))]
    p.plan(observation(ep, 5, depth=3))
    assert not p.closing.active and p.closing.map_rejections >= 1
    assert p.landmarks.confirmed()[0].votes == {"bed": 5, "chair": 1}, "the box voted, and lost"
    assert p._target_xy is None, "no target evidence from an outvoted box"
    assert p.perception.counts["outvoted_detections"] == 1


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
    [(spot, floor, step, radius)] = p.closing.rejected
    assert spot == anchor and step == 4 and radius == pytest.approx(2.0 * p.settings.target_closing.rejection_radius_m)
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


def test_a_lock_the_map_outvotes_is_released_at_once(monkeypatch):
    """A sofa from four metres, confirmed as a bed by the map's vote from two: the lock goes, the spot is
    remembered, and no inspection is spent on it."""
    p, ep, near = locked_at_two_metres(monkeypatch)
    anchor = p.closing.anchor
    for frame in range(10, 14):                                     # the map confirms a bed where the sofa was
        p.landmarks.observe("bed", anchor[:2], frame_id=frame, radius_m=0.8)
    bed = next(lm for lm in p.landmarks.confirmed() if lm.class_name == "bed")
    assert p.landmarks.is_confirmed(bed) and not p.target.accepts("bed")
    p.detector.detect = lambda rgb: []
    obs = replace(near, step=2)
    p.perception.observe(obs)
    p.closing.observe(obs)                                          # the policy's own order: perceive, then the closing
    assert not p.closing.active and p.closing.phase == "RELEASED", "released before any plan: this action is the search's"
    assert p.closing.map_releases == 1 and p.closing.last_release == "contradicted by the map"
    assert p.closing.rejected[0][3] == pytest.approx(2.0 * p.settings.target_closing.rejection_radius_m)
    assert p.closing.inspection_releases == 0 and p._target_xy is None


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
    # Seen again from the same place: the memory holds.
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (280, 200, 360, 280))]
    p.plan(observation(ep, 3, depth=3.5))
    assert not p.closing.active
    # Seen from 1.2 m, standing 2.3 m closer: the same spot, a new view -- the takeover starts.
    near = replace(observation(ep, 4, depth=1.2), pose=AgentPose(2.3, 0, 0, 0))
    p.plan(near)
    assert p.closing.active and math.dist(p.closing.anchor[:2], spot[:2]) < p.settings.target_closing.rejection_radius_m


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
    # A DIFFERENT far object two actions later: cooling, no takeover.
    far_elsewhere = replace(observation(ep, 4, depth=3.5), pose=AgentPose(0, 0, 0, math.pi / 2))
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (280, 200, 360, 280))]
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
    action = converter.step(pose, command).action or DiscreteAction.TURN_LEFT
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
    p, original = setup_policy()
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
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (280, 200, 360, 280))]
    p.plan(observation(ep, 0, depth=3.5))
    assert p.closing.active and not p.closing.locked
    p.detector.detect = lambda rgb: []
    p.plan(observation(ep, 1, depth=3.5))
    p.plan(observation(ep, 2, depth=3.5))
    assert not p.closing.active and p.closing.last_release == "unverified" and p.closing.rejected
    p.detector.detect = lambda rgb: [DetectionWire("chair", .9, (280, 200, 360, 280))]
    for step in (3, 4, 5):
        p.plan(observation(ep, step, depth=3.5))
    assert p._target_xy is None, "the released spot is not a legacy target either"
    assert p.perception.counts["refused_evidence"] >= 1
    assert any(row.get("target_evidence") == "refused_by_takeover" for row in p.perception.projections)
    assert not p.closing.active, "and the takeover itself still refuses it from here"
