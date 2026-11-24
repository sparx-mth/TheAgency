"""Standing on a frontier node starts a short in-place spin, so the frames after it can show what lies beyond."""
import pytest

from sparx_agency.core.planning.object_search.frontier_views.tests.rig import pose
from sparx_agency.core.planning.object_search.frontier_views.tests.test_live_search import two_gaps
from sparx_agency.core.planning.object_search.tests.fixtures import frame

SCORES = [0.2, 0.8] * 4


def arrive(monkeypatch, **settings):
    p, ep, _ = two_gaps(monkeypatch, SCORES, **settings)
    p.plan(frame(ep, 0, depth=float("nan")))
    return p, ep, pose(*p._goal)


def test_arriving_at_a_node_starts_the_configured_spin(monkeypatch):
    p, ep, at_goal = arrive(monkeypatch, arrival_spin_turns=3)
    command = p.plan(frame(ep, 2, depth=float("nan"), pose=at_goal))
    assert command.info["kind"] == "warmup" and command.info["warmup_step"] == 1 and command.info["warmup_budget"] == 3


def test_no_spin_by_default(monkeypatch):
    p, ep, at_goal = arrive(monkeypatch)
    assert p.plan(frame(ep, 2, depth=float("nan"), pose=at_goal)).info.get("kind") != "warmup"


def test_a_node_spins_once(monkeypatch):
    p, ep, at_goal = arrive(monkeypatch, arrival_spin_turns=3)
    p.plan(frame(ep, 2, depth=float("nan"), pose=at_goal))
    p.warmup_actions = p.warmup_turns
    assert p.plan(frame(ep, 3, depth=float("nan"), pose=at_goal)).info.get("kind") != "warmup"


def test_a_new_floor_still_gets_the_full_warmup(monkeypatch):
    p, ep, at_goal = arrive(monkeypatch, arrival_spin_turns=3, multifloor={"enabled": False})
    p.plan(frame(ep, 2, depth=float("nan"), pose=at_goal))
    p.warmup_actions = p.warmup_turns
    command = p.plan(frame(ep, 3, depth=float("nan"), pose=pose(0, 0, z=3)))
    assert command.info["kind"] == "warmup" and command.info["warmup_budget"] == p.rotation_turns()


def test_the_end_of_the_route_starts_the_spin_even_when_the_node_is_already_gone(monkeypatch):
    p, ep, at_goal = arrive(monkeypatch, arrival_spin_turns=3)
    observe = p.frontiers.observe
    monkeypatch.setattr(p.frontiers, "observe", lambda obs, world, clusters: observe(obs, world, ()))
    command = p.plan(frame(ep, 2, depth=float("nan"), pose=at_goal))
    assert command.info["kind"] == "warmup" and command.info["warmup_budget"] == 3


def test_no_spin_while_the_route_is_still_under_way(monkeypatch):
    p, ep, _ = arrive(monkeypatch, arrival_spin_turns=3)
    assert p.plan(frame(ep, 2, depth=float("nan"))).info.get("kind") != "warmup"


def test_one_route_spins_once_however_long_the_agent_stays_at_its_end(monkeypatch):
    p, ep, at_goal = arrive(monkeypatch, arrival_spin_turns=3)
    p.plan(frame(ep, 2, depth=float("nan"), pose=at_goal))
    p.warmup_actions = p.warmup_turns
    assert p.plan(frame(ep, 3, depth=float("nan"), pose=at_goal)).info.get("kind") != "warmup"


def test_standing_on_a_node_during_the_opening_warmup_leaves_that_warmup_whole(monkeypatch):
    p, ep, _ = two_gaps(monkeypatch, SCORES, arrival_spin_turns=3, warmed_up=False)
    p.plan(frame(ep, 0, depth=float("nan")))
    on_a_node = pose(*next(iter(p.frontiers.nodes.values())).xy)
    command = p.plan(frame(ep, 1, depth=float("nan"), pose=on_a_node))
    assert command.info["kind"] == "warmup" and command.info["warmup_budget"] == p.rotation_turns()


@pytest.mark.parametrize("bad", [-1, 1.5, True, "3"])
def test_the_spin_length_is_a_non_negative_integer(bad):
    from sparx_agency.core.planning.object_search.settings import RPTSettings
    with pytest.raises(ValueError, match="arrival_spin_turns"):
        RPTSettings(arrival_spin_turns=bad)
