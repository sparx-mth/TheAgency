"""The selection: the best frame first, a further frame only for what it adds, and a reason when there is none.

One plan throughout -- a room, a doorway in its east wall, and the two
metres of corridor seen through it -- and frames taken from named places in
the room. The node is the corridor's boundary: its far end (which a frame
looking down the corridor sees PAST) and its two sides (which no frame in
the room can see past: the corridor's floor lies behind them).
"""
from __future__ import annotations

import math
import types

import numpy as np
import pytest

from sparx_agency.core.planning.object_search.frontier_views.nodes import FrontierNode
from sparx_agency.core.planning.object_search.frontier_views.selection import (
    HIDDEN, NO_FRAMES, NOT_FRAMED, TOO_FAR, TOO_THIN, select_views, storey_frames, views_by_frame)
from sparx_agency.core.planning.object_search.frontier_views.settings import ViewSettings
from sparx_agency.core.planning.object_search.frontier_views.sight import OCCLUDED, SEEN, SightScene
from sparx_agency.core.planning.object_search.frontier_views.tests.rig import (
    Plan, camera, node, pose, record, room_with_doorway)

CAMERA = camera()


def corridor_end(plan, node_id=0):
    """The corridor's far end: the column of cells at ``x`` just under 7, across its 0.8 m."""
    return node(plan, 6.9, -0.4, 7.0, 0.4, node_id)


def corridor_boundary(plan, node_id=0) -> FrontierNode:
    """The whole boundary of the corridor seen through the doorway: both sides and the far end, 44 cells."""
    parts = [plan.cells(5.1, 0.3, 6.9, 0.4), plan.cells(5.1, -0.4, 6.9, -0.3), plan.cells(6.9, -0.4, 7.0, 0.4)]
    rows = np.concatenate([rows for rows, _ in parts])
    cols = np.concatenate([cols for _, cols in parts])
    return FrontierNode(node_id, 0, (6.95, 0.05), (int(cols[-4]), int(rows[-4])), rows, cols, 3)


def choose(plan, nodes, records, settings=None, **kwargs):
    return select_views(plan.world(), nodes, records, CAMERA, settings, plan.settled, **kwargs)


def test_the_frame_that_looks_down_the_corridor_is_chosen_over_one_that_glimpses_it():
    plan = room_with_doorway()
    end = corridor_end(plan)
    square = record(3, pose(2.05, 0.05))                        # in line with the doorway, 4.9 m from the end
    glimpse = record(7, pose(1.05, 1.05, yaw_deg=-12.0))        # from the side: the north jamb hides part of the end
    away = record(9, pose(2.05, 0.05, yaw_deg=180.0))
    chosen = choose(plan, [end], [away, glimpse, square])[end.node_id]
    assert chosen.usable and chosen.names == ("step_000003.jpg",) and chosen.reason == ""
    view = chosen.views[0]
    assert view.seen_fraction == 1.0 and view.through_fraction == 1.0
    assert view.range_m == pytest.approx(4.9, abs=0.05)
    assert chosen.score == pytest.approx(ViewSettings().good_range_m / 4.9, rel=0.02)
    assert view.gain == pytest.approx(chosen.score)
    assert chosen.best_score == pytest.approx(chosen.score) and chosen.frames == 3
    # The window: the 0.8 m of corridor end, centred, floor to the top of the band.
    k = CAMERA.intrinsics
    left, top, right, bottom = view.window
    assert left < k.cx < right and right - left == pytest.approx(k.fx * 0.7 / 4.9, rel=0.1)
    assert top < k.cy < bottom and 0 <= top and bottom <= k.height - 1


def test_a_second_frame_is_kept_only_for_what_it_adds_and_three_is_the_most():
    plan = room_with_doorway()
    boundary = corridor_boundary(plan)
    # From the middle of the room, in line with the doorway, the far end and both sides are in view.
    centre = record(1, pose(2.05, 0.05))
    back = record(2, pose(0.55, 0.05))                          # the same line of sight from a metre and a half back
    alone = choose(plan, [boundary], [centre])[boundary.node_id]
    assert alone.names == ("step_000001.jpg",) and alone.seen_fraction == 1.0
    # The farther frame shows the same cells, each from farther: it adds nothing and is not kept.
    assert choose(plan, [boundary], [back, centre])[boundary.node_id].names == ("step_000001.jpg",)
    # From against the south wall the corridor's north side is seen ACROSS (past it lies the unknown) and its
    # south side is behind the jamb; from the north wall the reverse. Each shows what the other cannot.
    south = record(3, pose(3.05, -1.55, yaw_deg=25.0))
    north = record(4, pose(3.05, 1.55, yaw_deg=-25.0))
    pair = choose(plan, [boundary], [south, north])[boundary.node_id]
    assert len(pair.views) == 2 and set(pair.names) == {"step_000003.jpg", "step_000004.jpg"}
    assert pair.views[1].gain >= 0.2 * pair.views[0].gain
    assert pair.seen_fraction > max(view.seen_fraction for view in pair.views)
    assert pair.score == pytest.approx(sum(view.gain for view in pair.views))
    # A bar no second frame clears, and a budget of one, both leave the best frame alone.
    assert len(choose(plan, [boundary], [south, north], ViewSettings(min_gain_ratio=1.0))[boundary.node_id].views) == 1
    assert len(choose(plan, [boundary], [south, north], ViewSettings(max_frames=1))[boundary.node_id].views) == 1
    many = [record(10 + i, pose(1.05 + 0.3 * i, -1.5 + 0.3 * i, yaw_deg=20.0 - 4.0 * i)) for i in range(9)]
    crowd = choose(plan, [boundary], many, ViewSettings(min_gain_ratio=0.0, min_gain=0.0))[boundary.node_id]
    assert 1 <= len(crowd.views) <= 3 and len(set(crowd.names)) == len(crowd.views)
    assert [view.gain for view in crowd.views] == sorted((view.gain for view in crowd.views), reverse=True)


def test_under_the_strict_rule_only_a_frame_that_sees_past_the_boundary_is_a_view():
    plan = room_with_doorway()
    sides = FrontierNode(1, 0, (6.05, 0.35), (160, 103), *plan.cells(5.1, 0.3, 6.9, 0.4), 1)
    end = corridor_end(plan)
    frames = [record(1, pose(2.05, 0.05))]
    relaxed = choose(plan, [sides, end], frames)
    assert relaxed[1].usable and relaxed[1].through_fraction == 0.0           # seen, along the boundary
    strict = choose(plan, [sides, end], frames, ViewSettings(at_weight=0.0))
    assert strict[end.node_id].usable and strict[end.node_id].through_fraction == 1.0
    assert not strict[1].usable and strict[1].reason == TOO_THIN and strict[1].views == ()
    assert strict[1].score == 0.0 and strict[1].best_score == 0.0


def test_a_node_without_a_view_says_which_gate_every_frame_failed():
    plan = room_with_doorway()
    end = corridor_end(plan)
    assert choose(plan, [end], [])[end.node_id].reason == NO_FRAMES
    far = choose(plan, [end], [record(1, pose(0.55, 0.05))], ViewSettings(max_range_m=4.0))[end.node_id]
    assert not far.usable and far.reason == TOO_FAR and far.frames == 1
    turned = choose(plan, [end], [record(1, pose(2.05, 0.05, yaw_deg=120.0))])[end.node_id]
    assert turned.reason == NOT_FRAMED
    hidden = choose(plan, [end], [record(1, pose(4.05, 1.55, yaw_deg=-28.0))])[end.node_id]      # the north jamb is in the way
    assert hidden.reason == HIDDEN
    # Every cell seen, at a worth under the bar: the bar is the score's, not only the count's.
    whole = choose(plan, [end], [record(1, pose(2.05, 0.05))], ViewSettings(min_score=0.7))[end.node_id]
    assert whole.reason == TOO_THIN and whole.views == () and whole.best_score == pytest.approx(3.0 / 4.9, rel=0.02)
    # Seen, but one cell of eight at eight metres is not a picture of it.
    plan = Plan().free(0.0, -2.0, 9.0, 2.0)
    wide = node(plan, 8.0, -2.0, 8.1, 2.0)
    thin = choose(plan, [wide], [record(1, pose(7.55, -1.95, yaw_deg=0.0))])[wide.node_id]
    assert thin.reason == TOO_THIN and 0.0 < thin.best_score < ViewSettings().min_score


def test_frames_of_another_storey_are_not_asked():
    plan = room_with_doorway()
    end = corridor_end(plan)
    upstairs = record(1, pose(2.05, 0.05, z=2.9))
    here = record(2, pose(2.05, 0.05, z=0.1))
    assert choose(plan, [end], [upstairs], reference_z=0.0)[end.node_id].reason == NO_FRAMES
    assert choose(plan, [end], [upstairs, here], reference_z=0.0)[end.node_id].names == ("step_000002.jpg",)
    assert choose(plan, [end], [upstairs])[end.node_id].usable                 # no reference: every frame is asked
    assert [r.step for r in storey_frames([upstairs, here], 0.0, 0.6)] == [2]


def test_one_pose_is_one_picture_and_the_latest_frame_of_it_is_the_one_named():
    plan = room_with_doorway()
    end = corridor_end(plan)
    same = pose(2.05, 0.05)
    frames = [record(4, same), record(9, same), record(6, same), record(2, pose(2.05, 0.05, yaw_deg=180.0))]
    assert [r.step for r in storey_frames(frames, None, 0.6)] == [2, 9]
    chosen = choose(plan, [end], frames)[end.node_id]
    assert chosen.names == ("step_000009.jpg",) and chosen.frames == 2
    # Between two frames that show the same -- mirror images of each other about the corridor's axis -- the later.
    for first, second in ((4, 8), (8, 4)):
        twins = [record(first, pose(2.05, 0.5, yaw_deg=-5.0)), record(second, pose(2.05, -0.5, yaw_deg=5.0))]
        assert choose(plan, [end], twins)[end.node_id].names == ("step_000008.jpg",)


def test_the_frame_logs_own_lookup_is_accepted_as_the_frames_to_choose_from():
    """``EpisodeFrameLog.records`` maps a frame's name to its record: handing it over whole chooses from its records."""
    plan = room_with_doorway()
    end = corridor_end(plan)
    frames = [record(3, pose(2.05, 0.05)), record(7, pose(2.05, 0.05, yaw_deg=180.0))]
    lookup = types.MappingProxyType({frame.name: frame for frame in frames})
    assert choose(plan, [end], lookup)[end.node_id] == choose(plan, [end], frames)[end.node_id]
    assert choose(plan, [end], lookup)[end.node_id].names == ("step_000003.jpg",)
    assert storey_frames(dict(lookup), None, 0.6) == frames and storey_frames(iter(frames), None, 0.6) == frames


def test_of_the_frames_of_a_turn_on_one_spot_the_one_with_the_node_in_the_middle_is_taken():
    """A look-around's frames see a doorway from one place: worth the same, and not the same picture."""
    plan = room_with_doorway()
    end = corridor_end(plan)
    turn = [record(step, pose(2.05, 0.05, yaw_deg=yaw)) for step, yaw in enumerate((-30.0, 0.0, 30.0, 60.0))]
    chosen = choose(plan, [end], turn)[end.node_id]
    # Three of the four have the corridor's end in the picture, at one worth; the last of them has it at the edge.
    assert chosen.names == ("step_000001.jpg",) and chosen.frames == 4
    k = CAMERA.intrinsics
    assert abs(0.5 * (chosen.views[0].window[0] + chosen.views[0].window[2]) - k.cx) < 5.0
    # Nearer the middle beats later: 20 degrees off the axis at action 2, 30 degrees off at action 5.
    edge = choose(plan, [end], [record(2, pose(2.05, 0.05, yaw_deg=-20.0)), record(5, pose(2.05, 0.05, yaw_deg=30.0))])
    assert edge[end.node_id].names == ("step_000002.jpg",) and edge[end.node_id].score == pytest.approx(chosen.score)
    # A better view beats a better-centred one: centring only breaks ties.
    nearer = record(9, pose(3.05, 0.05, yaw_deg=25.0))
    assert choose(plan, [end], turn + [nearer])[end.node_id].names == ("step_000009.jpg",)


def test_each_node_is_chosen_for_on_its_own_and_two_may_be_handed_one_frame():
    plan = room_with_doorway()
    plan.unknown(3.5, 1.0, 5.0, 2.0)                             # the room's north-east corner has not been seen
    corner = node(plan, 3.5, 0.9, 5.0, 1.0, node_id=5)           # ... so the floor along it is boundary
    end = corridor_end(plan, node_id=2)
    east = record(1, pose(2.05, 0.05, yaw_deg=-25.0))            # the doorway, the corner out of the picture
    up = record(2, pose(4.25, -1.05, yaw_deg=90.0))              # the corner square on, the doorway out of it
    both = record(3, pose(2.05, 0.05))                           # both in one picture
    selection = choose(plan, [end, corner], [east, up])
    assert selection[2].names == ("step_000001.jpg",) and selection[5].names == ("step_000002.jpg",)
    # Square on, the unknown corner is seen into -- but for the row's two end cells, whose lines leave it sideways.
    assert selection[5].through_fraction == pytest.approx(13.0 / 15.0)
    assert views_by_frame(selection) == {"step_000001.jpg": [2], "step_000002.jpg": [5]}
    shared = choose(plan, [end, corner], [both])
    assert shared[2].usable and shared[5].usable
    assert views_by_frame(shared) == {"step_000003.jpg": [2, 5]}
    # The corner lies to the camera's left of the doorway: its window ends before the doorway's begins.
    assert shared[5].views[0].window[2] < shared[2].views[0].window[0]
    # Adding a node changes nothing for the others.
    assert choose(plan, [end], [east, up])[2] == selection[2]


def test_the_map_of_now_decides_what_an_old_frame_shows():
    plan = room_with_doorway()
    end = corridor_end(plan)
    frames = [record(1, pose(2.05, 0.05))]
    assert choose(plan, [end], frames)[end.node_id].usable
    plan.wall(5.0, -0.4, 5.1, 0.4)                               # the doorway turns out to be a closed door
    assert choose(plan, [end], frames)[end.node_id].reason == HIDDEN


def test_the_selection_does_not_depend_on_the_order_the_frames_or_the_nodes_are_given_in():
    plan = room_with_doorway()
    boundary, end = corridor_boundary(plan, 1), corridor_end(plan, 2)
    frames = [record(i, pose(0.55 + 0.4 * i, -1.5 + 0.35 * i, yaw_deg=30.0 - 7.0 * i)) for i in range(8)]
    forward = choose(plan, [boundary, end], frames)
    backward = choose(plan, [end, boundary], frames[::-1])
    assert forward == backward
    assert choose(plan, [], frames) == {}


def test_a_window_is_the_box_the_seen_cells_land_in():
    plan = room_with_doorway()
    boundary = corridor_boundary(plan)
    view = choose(plan, [boundary], [record(1, pose(2.05, 0.05))])[boundary.node_id].views[0]
    k = CAMERA.intrinsics
    left, top, right, bottom = view.window
    # The corridor's sides run from the doorway to its end, so the box is as wide as their nearest cells, 3.1 m
    # off: the north side 0.30 m to the camera's left (the camera stands at y = 0.05), the south side 0.40 m right.
    assert left == pytest.approx(k.cx - k.fx * 0.30 / 3.1, abs=1.0)
    assert right == pytest.approx(k.cx + k.fx * 0.40 / 3.1, abs=1.0)
    assert bottom == pytest.approx(k.cy + k.fy * 0.88 / 3.1, abs=1.0)             # the floor at the nearest cell
    assert top == pytest.approx(k.cy - k.fy * 0.62 / 3.1, abs=1.0)                # 1.5 m up at the nearest cell
    assert math.isfinite(top) and 0 <= top < k.cy < bottom <= k.height - 1
    # From against the north wall the south jamb hides part of the boundary: the box is of what is SEEN, and the
    # cells the frame faces without reaching lie outside it.
    side = pose(3.05, 1.55, yaw_deg=-25.0)
    partial = choose(plan, [boundary], [record(2, side)])[boundary.node_id].views[0]
    sight = SightScene(plan.world(), boundary.rows, boundary.cols, CAMERA, None, plan.settled).sight(side)
    seen, hidden = sight.stage == SEEN, sight.stage == OCCLUDED
    assert 0.2 < partial.seen_fraction < 0.9 and seen.sum() == round(partial.seen_fraction * boundary.size_cells)
    assert partial.window == pytest.approx((sight.u[seen].min(), sight.v_top[seen].min(), sight.u[seen].max(),
                                            sight.v_bottom[seen].max()))
    assert hidden.any() and (sight.u[hidden].min() < partial.window[0] or sight.u[hidden].max() > partial.window[2])
