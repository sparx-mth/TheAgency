"""The visual arm's first-floor exception must not become general stair traversal."""
import pytest

from sparx_agency.core.planning.object_search.tests.fixtures import FakeDetector, body_actions, frame, make_policy
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.pose import AgentPose


def spawn_policy():
    return make_policy(detector=FakeDetector("bed"), actions=body_actions(tilt=True),
                       multifloor={"spawn_leave_after_actions": 3, "spawn_departure_attempts": 1})


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("level", [.18, .52])
def test_stair_spawn_owns_motion_and_views_until_first_floor(monkeypatch, sign, level):
    p, ep = spawn_policy()
    b, atlas = p.building, p.mapping.atlas
    flights = [{"rise_m": rise, "direction": direction, "path": [(1., 1., 0.), (1.5, 1., .4 * direction)]}
               for rise, direction in ((2.1, -sign), (.5, sign))]
    monkeypatch.setattr(b.terrain, "_candidate_paths", lambda pose: setattr(b.terrain, "candidates", flights))
    b.terrain.plateau_area = atlas.plateau_observer = lambda pose: 4.0 if abs(pose.z) >= level else 1.0
    monkeypatch.setattr(b.terrain, "stair_distance", lambda pose: 1.0)
    old = p.frontiers
    monkeypatch.setattr(old, "refresh", lambda *args, **kw: pytest.fail("frontier scored on the spawn flight"))
    monkeypatch.setattr(p.closing, "observe", lambda obs: pytest.fail("target acquired on provisional treads"))
    for step in range(4):
        obs = frame(ep, step, depth=3, pose=AgentPose(1, 1))
        cmd = p.plan(obs)
    assert b.transition is not None and b.active["direction"] == sign and b._departures == 1
    assert cmd.info["kind"] == "TRAVERSE" and not p.closing.active
    assert b.departure is None and b.switch is None and atlas.in_transition
    for permit in (True, False):
        monkeypatch.setattr(b.terrain, "permits", lambda *args: permit)
        assert p.filter_action(obs, DiscreteAction.MOVE_FORWARD) == (
            DiscreteAction.MOVE_FORWARD if permit else DiscreteAction.TURN_LEFT)
    for step, z in enumerate((.18, .35, .52, .52, .52, .52, .52), 4):
        obs = frame(ep, step, depth=3, pose=AgentPose(1 + (step - 3) * .25, 1, sign * min(level, z)))
        cmd = p.plan(obs)
        if b.storey_confirmed:
            break
    assert b.storey_confirmed and not b.traversing and b.transition is None
    assert cmd.info["kind"] == "spawn_handoff" and p.floor_guard.spawn_z == pytest.approx(sign * level)
    monkeypatch.undo()
    cmd = p.plan(frame(ep, step + 1, depth=3, pose=AgentPose(obs.pose.x, 1, sign * level)))
    assert cmd.info["kind"] == "warmup" and cmd.info["warmup_step"] == 1
    assert p.frontiers is not old and p.frontiers.first_frame == step + 1
    assert p.mapping.floor_id == atlas.active_id and not p.floor_guard.off_plane
    assert len(p.frame_log.records) == step + 2  # archival frames retained, not admitted to the new search


def test_unplanned_departure_needs_no_room_gate_and_cannot_retry_forever():
    p, ep = spawn_policy()
    b = p.building
    p.plan(frame(ep, 0, depth=3))
    obs = frame(ep, 1, depth=3, pose=AgentPose(.25, 0, -.52))
    assert p.plan(obs).info["kind"] == "TRAVERSE"
    assert b.active["direction"] == -1 and b._departures == 1 and b._retreat is None
    # Back on the source tread without an arrival on a floor.
    assert p.mapping.atlas.settle(AgentPose(0, 0), 0.0)
    p.plan(frame(ep, 2, depth=3))
    assert b.transition is None
    p.plan(frame(ep, 3, depth=3, pose=AgentPose(.25, 0, -.52)))
    assert b.retreating and b._departures == 1 and b.transition is None


def test_exhausted_spawn_on_observed_treads_does_not_handoff_to_frontiers(monkeypatch):
    p, ep = spawn_policy()
    b, atlas = p.building, p.mapping.atlas
    b.terrain.plateau_area = atlas.plateau_observer = lambda pose: 1.0
    flight = {"direction": 1, "rise_m": .6, "path": [(0., 0., 0.), (.5, 0., .3)]}
    monkeypatch.setattr(b.terrain, "_candidate_paths", lambda pose: setattr(b.terrain, "candidates", [flight]))
    for i in range(4):
        p.plan(frame(ep, i, depth=3))
    assert b.transition is not None and atlas.settle(AgentPose(0, 0), 0.0)
    cmd = p.plan(frame(ep, 4, depth=3))
    assert cmd.stop and cmd.info["kind"] == "SAFE_HALT" and not cmd.info["target_confirmed"]
    assert not b.storey_confirmed and b._departures == 1 and p.frontiers.calls == 0


def test_same_floor_step_then_confirmed_departure_is_not_a_stair_spawn():
    p, ep = spawn_policy()
    b = p.building
    b.terrain.plateau_area = p.mapping.atlas.plateau_observer = lambda pose: 6.0
    for i, z in enumerate((0., 0., 0., .15, .15, .15)):
        obs = frame(ep, i, depth=3, pose=AgentPose(i*.25, 0, z))
        p.plan(obs)
    assert b.storey_confirmed and b._departures == 0 and p.mapping.floor_id == 0
    assert p.floor_guard.stats["plane_remeasured"] == 1
    p.plan(frame(ep, 6, depth=3, pose=AgentPose(1.5, 0, -.52)))
    assert b.retreating and b.transition is None and b._departures == 0
