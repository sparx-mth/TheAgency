"""Unsafe forward vetoes must not become turn/undo/retry loops."""
from sparx_agency.core.planning.exploration.falcon.bursts import RECOVER, TRANSIT, VERIFY
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction as A
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.test_falcon_orchestration import rig


def test_transit_veto_commits_recovery_before_resuming(monkeypatch):
    p, agent, world, make = rig(monkeypatch)
    agent.act(make(0))
    h, m = p.hierarchy, p.hierarchy.machine
    m.transition(TRANSIT, "selected room")
    monkeypatch.setattr(h.motion, "permits", lambda *a, **kw: False)
    assert h.filter_action(make(1), A.MOVE_FORWARD) == A.TURN_LEFT
    assert m.phase == RECOVER and h.recovery_phase == TRANSIT
    assert p.route_memory.path is None
    for step in range(1, 4):
        m.charge(step)
    monkeypatch.setattr(h, "_transit", lambda *a: NavigationCommand.hold(info={"transit_resumed": True}))
    command = h.plan(make(4), world, False)
    assert m.phase == TRANSIT and command.info["transit_resumed"]
    assert m.actions == 4


def test_target_recovery_consumes_original_interrupt_allowance(monkeypatch):
    p, agent, world, make = rig(monkeypatch, target_actions=3)
    agent.act(make(0))
    p._target_id, p._target_xy, p._target_step = 42, (4.0, 4.0), 1
    monkeypatch.setattr(p, "_approach", lambda *a: NavigationCommand.hold(final_yaw=1.0))
    agent.act(make(1))
    h, m = p.hierarchy, p.hierarchy.machine
    assert m.phase == VERIFY
    monkeypatch.setattr(h.motion, "permits", lambda *a, **kw: False)
    assert h.filter_action(make(2), A.MOVE_FORWARD) == A.TURN_LEFT
    assert m.phase == RECOVER and h.recovery_phase == VERIFY
    m.charge(2)
    m.charge(3)
    h.plan(make(4), world, False)
    assert p._target_xy is None
    assert len(m.history) == 1 and m.actions == 4

