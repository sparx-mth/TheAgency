"""The image-space vertical servo (``methods/bbox_pitch``): rows, tilts and the regions of the frame.

Gibson's camera throughout: 640 x 480, 79-degree HFOV (fx = fy = 388.2 px), 30-degree tilt steps,
the multi-floor protocol's pitch limits of -30 / +60 degrees where limits matter.
"""
import math

import pytest

from sparx_agency.core.planning.objnav.camera_intrinsics import intrinsics_from_hfov
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.bbox_pitch import (
    REGION_CENTRE, REGION_CUT_BOTH, REGION_CUT_BOTTOM, REGION_CUT_TOP, REGION_LOWER, REGION_UPPER,
    box_after_advance, box_after_tilt, box_fits, region_compatible, row_after_tilt, vertical_fix)

K = intrinsics_from_hfov(640, 480, 79.0)
TILT = math.radians(30.0)
LIMITS = dict(min_pitch_rad=math.radians(-30.0), max_pitch_rad=math.radians(60.0))


def fix(box, pitch=0.0, **kwargs):
    return vertical_fix(box, K.width, K.height, K.cy, K.fy, pitch, TILT, region=0.25, margin_px=8, **kwargs)


def test_a_look_down_moves_the_image_content_up_by_the_tangent_of_the_tilt():
    """A row on the axis lands fy * tan(30) = 224 px higher; the principal row itself is exact."""
    assert row_after_tilt(K.cy, K.cy, K.fy, TILT) == pytest.approx(K.cy - K.fy * math.tan(TILT))
    assert row_after_tilt(K.cy + 100, K.cy, K.fy, 0.0) == pytest.approx(K.cy + 100)
    assert row_after_tilt(400.0, K.cy, K.fy, -TILT) > 400.0, "LOOK_UP moves the content down"
    x1, y1, x2, y2 = box_after_tilt((100, 300, 200, 400), K.cy, K.fy, TILT)
    assert (x1, x2) == (100, 200), "columns do not move"
    assert y1 < 300 and y2 < 400 and y2 - y1 < 100, "a box below the axis shrinks as it rises toward it"
    assert row_after_tilt(1e9, K.cy, K.fy, -math.pi / 2) == float("inf"), "past the edge saturates, never wraps"


def test_the_advance_prediction_pushes_a_low_box_down_and_a_high_box_up_and_grows_both():
    """Stepping 0.25 m toward a 0.4 m toilet 1.3 m away: its rows move down (deeper depression) and the
    box grows; a wall-mounted screen above the camera moves up. Standing still returns the box as it is."""
    low = (250, 300, 390, 420)                         # below the axis: the toilet from 1.3 m, level camera
    x1, y1, x2, y2 = box_after_advance(low, K.cy, K.fy, 0.0, 1.3, 0.25)
    assert (x1, x2) == (250, 390)
    assert y1 > 300 and y2 > 420 and (y2 - y1) > 120, "lower and taller after the step"
    high = (250, 60, 390, 180)                         # above the axis
    _, hy1, _, hy2 = box_after_advance(high, K.cy, K.fy, 0.0, 1.3, 0.25)
    assert hy1 < 60 and hy2 < 180 and (hy2 - hy1) > 120
    assert box_after_advance(low, K.cy, K.fy, 0.0, 1.3, 0.0) == tuple(float(v) for v in low)
    floored = box_after_advance(low, K.cy, K.fy, 0.0, 0.4, 0.25)
    assert floored[1] > 300 and floored == box_after_advance(low, K.cy, K.fy, 0.0, 0.4, 0.5), (
        "the range after the step never goes below the floor of 0.3 m")
    assert box_after_advance(low, K.cy, K.fy, 0.0, float("nan"), 0.25) == tuple(float(v) for v in low), "no range: no prediction"


def test_a_box_cut_by_the_lower_edge_asks_for_one_look_down_when_the_top_stays_in():
    """The Darden toilet from 1.36 m, level camera: box rows 285-480, cut by the lower edge. One LOOK_DOWN
    puts its top at row 72 -- whole, with the bowl now in view."""
    verdict = fix((196, 285, 325, 480))
    assert verdict.steps == 1 and verdict.region == REGION_CUT_BOTTOM and verdict.wanted
    after = box_after_tilt((196, 285, 325, 480), K.cy, K.fy, TILT)
    assert 60 < after[1] < 90, "the top edge after the tilt is well inside the frame"
    assert "LOOK_DOWN by 1" in verdict.reason


def test_a_box_cut_by_the_lower_edge_whose_top_would_leave_through_the_top_is_left_alone():
    """A tall object cut at the bottom with its top already near the upper margin: a LOOK_DOWN trades a bottom
    cut for a top cut, which is no gain -- nothing is asked for."""
    verdict = fix((100, 20, 540, 480))
    assert verdict.steps == 0 and verdict.region == REGION_CUT_BOTTOM
    assert "no tilt within the limits keeps the upper edge in" in verdict.reason


def test_a_box_cut_by_both_edges_fills_the_frame_and_no_tilt_is_asked_for():
    verdict = fix((100, 0, 540, 480))
    assert verdict.steps == 0 and verdict.region == REGION_CUT_BOTH


def test_a_box_cut_by_the_upper_edge_asks_for_a_look_up():
    """A wall-mounted television from close: rows 0-180, cut by the top. LOOK_UP brings it down whole."""
    verdict = fix((200, 0, 440, 180))
    assert verdict.steps == -1 and verdict.region == REGION_CUT_TOP
    assert "LOOK_UP by 1" in verdict.reason


def test_a_whole_box_in_the_lower_region_asks_for_the_rounded_look_down_and_one_in_the_centre_band_for_nothing():
    """Centre at row 400 (22 degrees below the axis): one step. Centre at row 300: the centre band; nothing."""
    verdict = fix((300, 370, 360, 430))
    assert verdict.steps == 1 and verdict.region == REGION_LOWER and verdict.centre_row == pytest.approx(400.0)
    centred = fix((300, 270, 360, 330))
    assert centred.steps == 0 and centred.region == REGION_CENTRE and not centred.wanted
    upper = fix((300, 40, 360, 100))
    assert upper.steps == -1 and upper.region == REGION_UPPER


def test_a_whole_box_in_the_lower_region_that_no_tilt_keeps_whole_is_left_alone():
    """A box spanning rows 100-470 with its centre at 285 is in the band; one spanning 250-470 (centre 360.0,
    on the edge) is not asked for a tilt that would push its top out through the upper margin."""
    tall = fix((100, 200, 540, 470))                       # centre 335: the band
    assert tall.steps == 0 and tall.region == REGION_CENTRE
    low_tall = fix((100, 290, 540, 470))                   # centre 380: lower region; a tilt lifts the top to row 2
    assert low_tall.region == REGION_LOWER
    after = box_after_tilt((100, 290, 540, 470), K.cy, K.fy, TILT)
    assert (after[1] >= 8) == (low_tall.steps == 1), "asked for exactly when the whole box stays inside the margins"


def test_the_servo_predicts_the_box_after_the_step_toward_a_low_target():
    """From 1.5 m a toilet's box sits whole at rows 310-445 (centre 378 -- the lower region already in this
    frame); from 1.25 m, after one 0.25 m step, its lower edge is predicted at row 486: cut. The verdict is
    read on the predicted frame: LOOK_DOWN now, so the step does not push the object out through the bottom."""
    standing = fix((250, 310, 390, 445))
    stepping = fix((250, 310, 390, 445), range_m=1.5, advance_m=0.25)
    assert standing.region == REGION_LOWER and standing.steps == 1
    assert stepping.region == REGION_CUT_BOTTOM and stepping.steps == 1 and "after the step" in stepping.reason
    centred = fix((250, 200, 390, 300), range_m=2.0, advance_m=0.25)
    assert centred.steps == 0, "a centred box that stays centred after the step asks for nothing"


def test_the_pitch_limits_cut_a_tilt_back_to_what_they_allow():
    """At the 60-degree limit a bottom-cut box cannot be looked further down at; at -30 a top cut cannot be
    looked further up at. Unclamped, both are allowed."""
    at_bottom = fix((196, 285, 325, 480), pitch=math.radians(60.0), **LIMITS)
    assert at_bottom.steps == 0 and at_bottom.region == REGION_CUT_BOTTOM
    at_top = fix((200, 0, 440, 180), pitch=math.radians(-30.0), **LIMITS)
    assert at_top.steps == 0 and at_top.region == REGION_CUT_TOP
    assert fix((196, 285, 325, 480), pitch=math.radians(60.0)).steps == 1, "no limits: allowed"


def test_the_servo_never_asks_for_a_tilt_that_lands_the_centre_in_the_opposite_region():
    """Rounding puts a centre within fy * tan(tilt / 2) = 104 rows of the principal row after a tilt -- inside
    the 0.25 band with these intrinsics, as the compatibility test says. A 60-degree tilt's dead band is 224
    rows, wider than any sensible region: the servo then widens its band to the dead band by construction,
    so even there a tilt never lands the centre in the opposite region."""
    assert region_compatible(K.cy, K.fy, K.height, TILT, 0.25)
    assert not region_compatible(K.cy, K.fy, K.height, math.radians(60.0), 0.25)
    assert region_compatible(K.cy, K.fy, K.height, math.radians(60.0), 0.03)
    for tilt in (TILT, math.radians(60.0)):
        for row in range(int(K.height * 0.5), K.height - 40, 5):
            verdict = vertical_fix((300, row - 30, 360, row + 30), K.width, K.height, K.cy, K.fy, 0.0, tilt,
                                   region=0.25, margin_px=8)
            if verdict.steps:
                after = row_after_tilt(verdict.centre_row, K.cy, K.fy, verdict.steps * tilt)
                assert after >= min(K.height * 0.25, K.cy - K.fy * math.tan(tilt / 2)) - 1e-6, (
                    "after the tilt the centre never lands in the upper region")
                assert after <= K.cy + 1e-6, "... and lies at or above the principal row"
    wide = vertical_fix((300, 400, 360, 460), K.width, K.height, K.cy, K.fy, 0.0, math.radians(60.0), region=0.25, margin_px=8)
    assert wide.steps == 0 and wide.region == REGION_CENTRE, "row 430 is inside a 60-degree tilt's dead band: nothing"


def test_box_fits_judges_the_rows_against_the_margins_only():
    assert box_fits((0, 8, 640, 472), 480, 8) and not box_fits((0, 7, 640, 472), 480, 8) and not box_fits((0, 8, 640, 473), 480, 8)


@pytest.mark.parametrize("kwargs", [dict(tilt_rad=0.0), dict(tilt_rad=float("nan")), dict(region=0.0), dict(region=0.5),
                                    dict(height=0)])
def test_bad_servo_inputs_raise(kwargs):
    args = dict(box=(300, 370, 360, 430), width=K.width, height=K.height, cy=K.cy, fy=K.fy, pitch_rad=0.0, tilt_rad=TILT,
                region=0.25, margin_px=8)
    args.update(kwargs)
    with pytest.raises(ValueError):
        vertical_fix(**args)




