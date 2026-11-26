"""A live policy on an observed floor with two equally distant gaps in the same image, no model or GPU."""
import numpy as np
import pytest

from sparx_agency.core.planning.object_search.frame_log.codec import FrameCodec
from sparx_agency.core.planning.object_search.frontier_views.tests.rig import Plan, pose
from sparx_agency.core.planning.object_search.tests.fixtures import FakeDetector, frame, make_policy
from sparx_agency.core.planning.objnav.errors import ObjNavInternalError


class VisualModel:
    def __init__(self, probabilities):
        self.probabilities = iter(probabilities)
        self.questions = []

    def chat_json(self, system, user, *, image_jpeg):
        self.questions.append((user, FrameCodec().decode(image_jpeg)))
        value = next(self.probabilities)
        if isinstance(value, Exception):
            raise value
        return {"probability": value}


def two_gaps(monkeypatch, probabilities, **settings):
    model = VisualModel(probabilities)
    p, ep = make_policy(FakeDetector("bed"), model, **settings)
    world = (Plan().free(-1, -2.5, 5, 2.5)
             .wall(-1.1, -2.6, -1, 2.6).wall(-1, -2.6, 5, -2.5).wall(-1, 2.5, 5, 2.6)
             .wall(4.9, -2.5, 5, -1.8).wall(4.9, -0.8, 5, 0.8).wall(4.9, 1.8, 5, 2.5)
             .free(5, -1.8, 5.8, -0.8).free(5, 0.8, 5.8, 1.8).world())
    update = p.mapping.update

    def mapped(observation, **kwargs):
        update(observation, **kwargs)  # real atlas; only substitute the observed floor plan
        return world

    monkeypatch.setattr(p.mapping, "update", mapped)
    return p, ep, model


@pytest.mark.parametrize("probabilities", [(0.01, 0.97), (0.97, 0.01)])
def test_different_windows_on_one_frame_mean_different_probabilities_rpt_orders_and_astar_goals(monkeypatch, probabilities):
    p, ep, model = two_gaps(monkeypatch, probabilities)
    obs = frame(ep, 0, depth=np.nan)
    command = p.plan(obs)
    assert len(model.questions) == 2 and all("chair" in prompt for prompt, _ in model.questions)
    red = []
    for _, image in model.questions:
        red.append((image[:, :, 0] > 150) & (image[:, :, 1] < 70) & (image[:, :, 2] < 70))
    assert all(mask.any() for mask in red) and not np.any(red[0] & red[1])
    assert not p.frame_log.rgb("step_000000.jpg").any(), "the source image is never marked"
    assert list(p.frontiers.scores.values()) == list(probabilities)
    assert all(r.frames == ("step_000000.jpg",) for r in p.frontiers.ledger.nodes.values())
    winner = p.solver.last.order[0]
    assert p.solver.last.source == "rpt_star" and p.frontiers.scores[winner] == 0.97
    assert command.info["frontier_id"] == winner and command.waypoints and p._goal == p.frontiers.nodes[winner].xy
    assert p._plan_calls == 1 and p.fallback.stats["invocations"] == 0
    assert p.plan(obs) is command
    assert p.plan(frame(ep, 1, depth=np.nan)).info["frontier_id"] == winner
    assert len(model.questions) == 2 and p.solver.calls == 1 and p._plan_calls == 1

    at_goal = pose(*p._goal)
    p.plan(frame(ep, 2, depth=np.nan, pose=at_goal))
    assert p.frontiers.ledger.nodes[winner].visited_step == 2 and p.frontiers.current != winner
    p.plan(frame(ep, 14, depth=np.nan, pose=at_goal))
    assert p.frontiers.ledger.nodes[winner].state == "blocked" and len(model.questions) == 2
    eligible = {g.cell for g in p.fallback._unblocked(p.frontiers.inventory.goals)}
    assert eligible.isdisjoint(p.frontiers.nodes[winner].goal_cells) and eligible
    p.reset(ep, p.target)
    assert not p.frontiers.scores and not p.frame_log.records


@pytest.mark.parametrize("bad", [True, "0.9", float("nan"), -1, 1.01, OSError("service away")])
def test_bad_scores_back_off_without_spending_the_node_and_fallback_keeps_moving(monkeypatch, bad):
    p, ep, model = two_gaps(monkeypatch, [bad, 0.2, 0.7])
    command = p.plan(frame(ep, 0, depth=np.nan))
    assert command.info["fallback_stage"] and not command.stop
    assert p.frontiers.ledger.counts()["pending"] == 2 and not p.frontiers.scores
    retry = p.fallback.retry_step["frontier_vlm"]
    p.plan(frame(ep, 1, depth=np.nan))
    assert len(model.questions) == 1
    p.plan(frame(ep, retry, depth=np.nan))
    assert len(model.questions) == 3 and p.frontiers.ledger.counts()["scored"] == 2


def test_identity_drift_is_not_an_exploration_fallback(monkeypatch):
    p, ep, _ = two_gaps(monkeypatch, [ObjNavInternalError("model changed")])
    with pytest.raises(ObjNavInternalError, match="model changed"):
        p.plan(frame(ep, 0, depth=np.nan))


def test_a_frontier_that_appears_after_turning_away_reads_the_earlier_camera_frame(monkeypatch):
    p, ep, model = two_gaps(monkeypatch, [0.4, 0.7])
    # The first action sees the gaps; their clusters only enter the inventory on the next action.
    observe = p.frontiers.observe
    monkeypatch.setattr(p.frontiers, "observe", lambda obs, world, clusters: observe(obs, world, () if obs.step == 0 else clusters))
    p.plan(frame(ep, 0, depth=np.nan))
    assert not model.questions
    p.plan(frame(ep, 1, depth=np.nan, pose=pose(0, 0, yaw_deg=180)))
    assert len(model.questions) == 2
    assert all(r.frames == ("step_000000.jpg",) for r in p.frontiers.ledger.nodes.values())


def test_closing_keeps_motion_even_when_a_due_score_fails(monkeypatch):
    p, ep, model = two_gaps(monkeypatch, [OSError("offline")])
    p.detector = FakeDetector("chair", duplicate=True)
    monkeypatch.setattr(p, "_explore", lambda *args: pytest.fail("closing must own these actions"))
    # Remove only the depth outside the target box, so the synthetic depth plane does not mask both frontiers.
    for step in (0, 1):
        obs = frame(ep, step, depth=np.nan)
        obs.depth_m[200:280, 280:360] = 0.7
        command = p.plan(obs)
        assert p._action_owner == "target_closing"
        assert command.info["kind"] == "target_closing" and "fallback" not in command.info
    assert p.closing.active and p.closing.count == 2
    assert len(model.questions) == 1 and p.frontiers.ledger.counts()["pending"] > 0 and p.solver.calls == 0


def test_one_request_preserves_all_selected_views_without_exceeding_the_pixel_budget():
    from sparx_agency.core.planning.object_search.frontier_views.scoring import MAX_VIEW_SIDE, ask_frontier, marked_view
    model = VisualModel([0.5])
    rgb = np.zeros((800, 1200, 3), np.uint8)
    images = [marked_view(rgb, box) for box in ((0, 0, 100, 800), (300, 200, 500, 400), (1050, 300, 1199, 500))]
    assert ask_frontier(model, "chair", images) == 0.5 and not rgb.any()
    sheet = model.questions[0][1]
    assert sheet.shape == (round(800 * MAX_VIEW_SIDE / 1200), MAX_VIEW_SIDE * 3, 3)
    for image in np.split(sheet, 3, axis=1):
        assert ((image[:, :, 0] > 160) & (image[:, :, 1] < 50)).any()


def test_first_frame_on_a_new_floor_starts_a_new_rotation_and_scores_only_that_floor(monkeypatch):
    p, ep, model = two_gaps(monkeypatch, [0.3] * 4, multifloor={"enabled": False})
    assert p.plan(frame(ep, 0, depth=np.nan)).waypoints
    old_ids = set(p.frontiers.scores)
    command = p.plan(frame(ep, 1, depth=np.nan, pose=pose(0, 0, z=3)))
    assert command.info["kind"] == "warmup" and command.info["warmup_step"] == 1
    assert set(p.frontiers.nodes).isdisjoint(old_ids)
    assert all(p.frontiers.ledger.nodes[i].frames == ("step_000001.jpg",) for i in p.frontiers.nodes)
    assert len(model.questions) == 4 and p.solver.calls == 1


def test_stair_handoff_keeps_calls_but_cannot_score_from_pre_landing_frames(monkeypatch):
    p, ep, model = two_gaps(monkeypatch, [0.3] * 4)
    p.plan(frame(ep, 0, depth=np.nan))
    assert p.frontiers.calls == 2
    monkeypatch.setattr(p.building, "visual_spawn_pending", lambda obs: obs.step == 1)
    p.plan(frame(ep, 1, depth=np.nan))
    # Even at the same elevation, the pre-landing frame facing the gaps is ineligible.
    p.plan(frame(ep, 2, depth=np.nan, pose=pose(0, 0, yaw_deg=180)))
    assert p.frontiers.first_frame == 2 and p.frontiers.calls == len(model.questions) == 2
    assert not p.frontiers.scores
    p.plan(frame(ep, 3, depth=np.nan))
    assert p.frontiers.calls == len(model.questions) == 4
    assert {r.frames for r in p.frontiers.ledger.nodes.values()} == {("step_000003.jpg",)}
