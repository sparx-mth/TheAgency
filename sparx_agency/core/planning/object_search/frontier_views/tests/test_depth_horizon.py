"""Long-range RGB is evidence on a frontier bearing, not a range measurement or a new navigation goal."""
import numpy as np
import pytest

from sparx_agency.core.planning.objnav.camera_geometry import project_to_image
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.object_search.frame_log.codec import FrameCodec
from sparx_agency.core.planning.object_search.frontier_views.scoring import ask_frontier, marked_view
from sparx_agency.core.planning.object_search.frontier_views.selection import select_views
from sparx_agency.core.planning.object_search.frontier_views.tests.rig import camera, node, pose, record, room_with_doorway
from sparx_agency.core.planning.object_search.frontier_views.tests.test_live_search import two_gaps
from sparx_agency.core.planning.object_search.tests.fixtures import FakeDetector, frame


def test_rgb_cue_at_nine_metres_is_in_the_window_of_the_five_metre_frontier():
    plan, cam, at = room_with_doorway(), camera(), pose(2.05, 0.05)
    boundary = node(plan, 6.9, -0.4, 7.0, 0.4)
    world = plan.world()
    original = world.grid.copy()
    view, = select_views(world, [boundary], [record(0, at)], cam)[boundary.node_id].views
    assert 4.8 < view.range_m < cam.max_depth_m == 5
    pixels, depth = project_to_image(np.array([[11.05, 0.05, 0.90]]), cam, at)
    (u, v), = pixels.round().astype(int)
    x0, y0, x1, y1 = view.window
    assert depth[0] == pytest.approx(9) and x0 < u < x1 and y0 < v < y1
    rgb = np.zeros((480, 640, 3), np.uint8)
    rgb[v-6:v+6, u-6:u+6, 1] = 255

    class Visual:
        def chat_json(self, system, user, *, image_jpeg):
            assert "5 m" in user and FrameCodec().decode(image_jpeg)[v, u, 1] > 240
            return {"probability": 0.8}

    assert ask_frontier(Visual(), "chair", [marked_view(rgb, view.window)], depth_limit_m=cam.max_depth_m) == 0.8
    np.testing.assert_array_equal(world.grid, original)


@pytest.mark.parametrize("far", [5.0, 7.0, np.inf])
def test_rgb_bearing_never_creates_a_target_position_or_stop_until_metric_depth(monkeypatch, far):
    p, ep, _ = two_gaps(monkeypatch, [0.25] * 10)
    p.detector = FakeDetector("chair")
    for step in (0, 1):
        obs = frame(ep, step, depth=np.nan)
        obs.depth_m[200:280, 280:360] = far
        command = p.plan(obs)
        assert not command.stop and not p.closing.locked and p.closing.anchor is None
    assert p.closing.phase == "BEARING" and not command.info["target_measured"]
    # An unplanned direct forward step of closing cannot override the measured stair/drop mask.
    p.floor_guard.mask = np.zeros(p.last_world.grid.shape, bool)
    x, y = p.last_world.world_to_grid(ep.action_spec.forward_step_m, 0)
    p.floor_guard.mask[y, x] = True
    assert p.filter_action(obs, DiscreteAction.MOVE_FORWARD) == DiscreteAction.TURN_LEFT
    for step in (2, 3):
        command = p.plan(frame(ep, step, depth=2.0))
    assert p.closing.bearing_resolved == 1 and p.closing.anchor is not None and p.closing.locked
    assert command.info["kind"] == "target_closing" and not hasattr(p, "_target_xy")


def test_only_the_frontier_aligned_with_distant_rgb_is_valued(monkeypatch):
    p, ep, model = two_gaps(monkeypatch, [])
    at = pose(0.8, 0)
    obs = frame(ep, 0, depth=np.nan, pose=at)
    # The right and left exits are both ~5 m away. The cue is 9 optical metres
    # away, on the ray through the north exit. Only its RGB pixels reach the policy.
    pixels, z = project_to_image(np.array([[9.8, 1.25 * 9 / 4.95, 0.9]]), ep.camera, at)
    (u, v), = pixels.round().astype(int)
    assert z[0] > ep.camera.max_depth_m
    obs.rgb[v-5:v+5, u-5:u+5, 1] = 255

    def prob_from_image(system, user, *, image_jpeg):
        img = FrameCodec().decode(image_jpeg)
        red = (img[:, :, 0] > 160) & (img[:, :, 1] < 70)
        green = (img[:, :, 1] > 170) & (img[:, :, 0] < 70)
        rows, cols = np.nonzero(red)
        return {"probability": 0.96 if green[rows.min():rows.max()+1, cols.min():cols.max()+1].any() else 0.01}

    monkeypatch.setattr(model, "chat_json", prob_from_image)
    command = p.plan(obs)
    assert sorted(p.frontiers.scores.values()) == [0.01, 0.96]
    assert command.info["probability"] == 0.96 and p._goal[1] > 0 and p._goal[0] - at.x <= ep.camera.max_depth_m
