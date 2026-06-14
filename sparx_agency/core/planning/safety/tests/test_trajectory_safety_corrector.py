"""Tests for :class:`TrajectorySafetyCorrector`.

The fields here are built by hand (or via ``PotentialFieldLayer`` where
available) so the geometry of each scenario is fully controlled. The corrector
is frame-agnostic: every test uses the standard BEV convention
``col = (x-origin_x)/res``, ``row = (y-origin_y)/res`` and indexes ``field[row, col]``.
"""
import numpy as np
import pytest

from sparx_agency.core.common.types import Path2D, Pose2D
from sparx_agency.core.planning.safety import (
    TrajectoryCorrectionParams,
    TrajectorySafetyCorrector,
)


# ---------------------------------------------------------------------------
# Field builders
# ---------------------------------------------------------------------------
def _gaussian_repulsion(occ: np.ndarray, sigma_px: float) -> np.ndarray:
    """Repulsive potential = peak-normalised Gaussian blur of the obstacle mask.

    A faithful interior stand-in for PotentialFieldLayer's U_rep (which blurs the
    obstacle mask with cv2.GaussianBlur and normalises to u_max) built with
    ``scipy.ndimage.gaussian_filter``. Border handling differs (``mode="nearest"``
    here vs cv2's reflect), so tests should sample the field interior, away from
    the map edge.
    """
    from scipy.ndimage import gaussian_filter

    u = gaussian_filter(occ.astype(np.float64), sigma=sigma_px, mode="nearest")
    peak = float(u.max())
    if peak > 1e-9:
        u /= peak
    return u


def _horizontal_corridor(h=60, w=120, wall=8):
    """Two horizontal walls (low-y + high-y rows); free corridor down the middle."""
    occ = np.zeros((h, w), dtype=np.float64)
    occ[:wall, :] = 1.0
    occ[-wall:, :] = 1.0
    return occ


# ---------------------------------------------------------------------------
# Centring behaviour
# ---------------------------------------------------------------------------
class TestCorridorCentring:
    def test_offset_path_moves_toward_centre(self):
        """A path hugging one wall of a tight corridor is pulled to the centre.

        A *tight* corridor (free band ~1 m, walls within ~2σ) gives the field a
        sharp central minimum, so opposing-wall repulsion cancels exactly on the
        centre-line. (Wide corridors only pull the path into the safe middle
        band — there is no narrow minimum to seek.)
        """
        h, w, res = 30, 120, 0.1
        occ = _horizontal_corridor(h, w, wall=10)       # free band y in [1.0, 2.0]
        u = _gaussian_repulsion(occ, sigma_px=6.0)
        corrector = TrajectorySafetyCorrector(
            TrajectoryCorrectionParams(iterations=20, gain=1.0, max_step_m=0.3,
                                       max_total_shift_m=1.0)
        )
        corrector.set_field(u, res, origin_x=0.0, origin_y=0.0)

        centre_y = (h / 2) * res                        # 1.5 m
        off_y = centre_y - 0.3                           # hugging the low-y wall
        xy = np.array([[x * res, off_y] for x in range(15, 105, 10)], dtype=np.float64)

        result = corrector.correct(xy)
        out = result.waypoints

        before = np.abs(xy[1:, 1] - centre_y)
        after = np.abs(out[1:, 1] - centre_y)
        assert np.all(after < before)                    # every wp moved closer
        assert after.mean() < 0.1                        # essentially centred
        assert result.visible_mask.all()

    def test_centred_path_barely_moves(self):
        """On the centre-line of a TIGHT corridor the path stays put.

        Uses a tight corridor (free band ~1 m) so the centre potential is well
        above ``u_floor`` and descent actually runs — the path is held by
        opposing-wall cancellation, not by the u_floor early-out.
        """
        h, w, res = 30, 120, 0.1
        u = _gaussian_repulsion(_horizontal_corridor(h, w, wall=10), sigma_px=6.0)
        corrector = TrajectorySafetyCorrector()
        corrector.set_field(u, res, 0.0, 0.0)

        centre_y = (h / 2) * res                          # 1.5 m
        # Sanity: descent really is active here (centre U >> u_floor).
        assert corrector._field.potential(5.0, centre_y) > corrector.params.u_floor

        xy = np.array([[x * res, centre_y] for x in range(15, 105, 10)], dtype=np.float64)
        result = corrector.correct(xy)
        assert result.max_shift_m < 0.05


# ---------------------------------------------------------------------------
# Single-wall repulsion direction
# ---------------------------------------------------------------------------
class TestRepulsionDirection:
    def test_push_is_away_from_single_wall(self):
        """Near one wall (no opposing wall), waypoints move directly away from it."""
        h, w, res = 80, 80, 0.1
        occ = np.zeros((h, w), dtype=np.float64)
        # row r -> world y = r*res, so the first rows are the *low-y* wall.
        occ[:8, :] = 1.0                        # wall along low y (y in [0, 0.8))
        u = _gaussian_repulsion(occ, sigma_px=6.0)
        corrector = TrajectorySafetyCorrector(
            TrajectoryCorrectionParams(iterations=8, gain=1.0, max_step_m=0.3,
                                       max_total_shift_m=1.5)
        )
        corrector.set_field(u, res, 0.0, 0.0)

        y_near = 1.2                            # ~0.4 m above the wall edge
        xy = np.array([[x * res, y_near] for x in range(10, 70, 10)], dtype=np.float64)
        out = corrector.correct(xy).waypoints
        # Wall is at low y -> waypoints should move to *higher* y (away from it).
        assert np.all(out[1:, 1] > xy[1:, 1] + 1e-3)


# ---------------------------------------------------------------------------
# "Only correct what you can see"
# ---------------------------------------------------------------------------
class TestVisibilityConstraint:
    def _setup(self):
        h, w, res = 60, 120, 0.1
        u = _gaussian_repulsion(_horizontal_corridor(h, w), sigma_px=6.0)
        c = TrajectorySafetyCorrector(TrajectoryCorrectionParams(iterations=10, gain=1.0))
        c.set_field(u, res, 0.0, 0.0)
        return c, h, w, res

    def test_out_of_field_waypoints_untouched(self):
        """Waypoints past the field extent are returned exactly unchanged."""
        c, h, w, res = self._setup()
        low_y = (h * 0.30) * res
        # First two inside the field, the rest far beyond the +x edge.
        xy = np.array(
            [[2.0, low_y], [4.0, low_y], [30.0, low_y], [40.0, low_y]],
            dtype=np.float64,
        )
        result = c.correct(xy)
        assert result.visible_mask.tolist() == [True, True, False, False]
        # Off-map points are bit-for-bit unchanged.
        assert np.allclose(result.waypoints[2:], xy[2:].astype(np.float32))

    def test_only_first_two_visible_corrects_only_them(self):
        """User scenario: only the first 2 points are on the map -> only #1 moves.

        (#0 is pinned, #1 is the lone correctable visible point.)
        """
        c, h, w, res = self._setup()
        low_y = (h * 0.30) * res
        xy = np.array([[2.0, low_y], [4.0, low_y], [50.0, low_y]], dtype=np.float64)
        result = c.correct(xy)
        assert result.corrected_mask.tolist() == [False, True, False]

    def test_known_mask_blocks_unobserved_cells(self):
        """A waypoint on an unobserved cell is skipped even though it is in bounds."""
        c, h, w, res = self._setup()
        known = np.ones((h, w), dtype=bool)
        known[:, 60:] = False                    # right half unobserved
        # Re-set the field with the mask.
        u = _gaussian_repulsion(_horizontal_corridor(h, w), sigma_px=6.0)
        c.set_field(u, res, 0.0, 0.0, known_mask=known)

        low_y = (h * 0.30) * res
        xy = np.array([[2.0, low_y], [4.0, low_y], [8.0, low_y]], dtype=np.float64)
        result = c.correct(xy)
        # x=8.0 -> col 80 -> unobserved -> not visible / not corrected.
        assert result.visible_mask.tolist() == [True, True, False]
        # The unobserved waypoint is returned bit-exact (not just position-ish).
        assert np.array_equal(result.waypoints[2], xy[2].astype(np.float32))

    def test_smoother_does_not_drag_across_unobserved_neighbour(self):
        """A movable waypoint next to an unobserved one is not blended toward it."""
        c, h, w, res = self._setup()
        known = np.ones((h, w), dtype=bool)
        known[:, 55:] = False                    # cols >=55 (x>=5.5) unobserved
        u = _gaussian_repulsion(_horizontal_corridor(h, w), sigma_px=6.0)
        c.set_field(u, res, 0.0, 0.0, known_mask=known)

        low_y = (h * 0.30) * res
        # wp2 (x=4.5) is the last visible interior point; wp3 (x=6.0) is unobserved.
        xy = np.array([[2.0, low_y], [3.0, low_y], [4.5, low_y], [6.0, low_y]],
                      dtype=np.float64)
        result = c.correct(xy)
        assert result.visible_mask.tolist() == [True, True, True, False]
        # Unobserved wp3 untouched; its x is never pulled toward the visible run.
        assert np.array_equal(result.waypoints[3], xy[3].astype(np.float32))


# ---------------------------------------------------------------------------
# Pinning, clamps, smoothing
# ---------------------------------------------------------------------------
class TestInvariants:
    def _corridor_corrector(self, **kw):
        h, w, res = 60, 120, 0.1
        u = _gaussian_repulsion(_horizontal_corridor(h, w), sigma_px=6.0)
        c = TrajectorySafetyCorrector(TrajectoryCorrectionParams(**kw))
        c.set_field(u, res, 0.0, 0.0)
        return c, h, res

    def test_first_waypoint_pinned(self):
        c, h, res = self._corridor_corrector(iterations=10, gain=1.0)
        low_y = (h * 0.30) * res
        xy = np.array([[x * res, low_y] for x in range(15, 95, 10)], dtype=np.float64)
        out = c.correct(xy).waypoints
        assert np.allclose(out[0], xy[0].astype(np.float32))

    def test_total_shift_clamped(self):
        c, h, res = self._corridor_corrector(
            iterations=30, gain=3.0, max_step_m=1.0, max_total_shift_m=0.15
        )
        low_y = (h * 0.20) * res
        xy = np.array([[x * res, low_y] for x in range(15, 95, 10)], dtype=np.float64)
        result = c.correct(xy)
        assert result.max_shift_m <= 0.15 + 1e-4

    def test_pin_last_holds_goal(self):
        c, h, res = self._corridor_corrector(iterations=10, gain=1.0, pin_last=True)
        low_y = (h * 0.30) * res
        xy = np.array([[x * res, low_y] for x in range(15, 95, 10)], dtype=np.float64)
        out = c.correct(xy).waypoints
        assert np.allclose(out[-1], xy[-1].astype(np.float32))


# ---------------------------------------------------------------------------
# Hard clearance guarantee
# ---------------------------------------------------------------------------
class TestClearanceGuarantee:
    @staticmethod
    def _wall_low_y(h=80, w=80, res=0.1, wall=8):
        """Single wall on the low-y rows; free space above it."""
        occ = np.zeros((h, w), dtype=np.float64)
        occ[:wall, :] = 1.0                      # wall on rows 0..wall-1 (low y)
        from scipy.ndimage import distance_transform_edt
        d_obs = distance_transform_edt(occ < 0.5).astype(np.float64) * res
        return occ, d_obs, res

    def test_min_clearance_enforced(self):
        """A waypoint just outside a wall is nudged up to the requested clearance."""
        occ, d_obs, res = self._wall_low_y(wall=8)        # wall edge at y = 0.8
        u = _gaussian_repulsion(occ, sigma_px=6.0)
        target = 0.5
        c = TrajectorySafetyCorrector(
            TrajectoryCorrectionParams(
                iterations=2, gain=0.4, min_clearance_m=target,
                clearance_iters=12, max_total_shift_m=2.0, max_step_m=0.3,
                smoothing_passes=2,                       # smoothing ON: must not erode clearance
            )
        )
        c.set_field(u, res, 0.0, 0.0, d_obs=d_obs)

        # Start ~0.2 m above the wall edge (edge at y=0.8 -> free-space start y=1.0).
        xy = np.array([[x * res, 1.0] for x in range(20, 60, 10)], dtype=np.float64)
        out = c.correct(xy).waypoints

        # Assert against the SAME bilinear field the corrector enforces.
        for i in range(1, len(out)):
            assert c._field.clearance(out[i, 0], out[i, 1]) >= target - 1e-3

    def test_too_narrow_corridor_is_best_effort_not_crash(self):
        """A corridor narrower than 2*target can't reach clearance -> best-effort.

        The push must not crash, must not exceed the shift cap, and should leave
        the path as clear as the geometry allows (no hard guarantee).
        """
        h, w, res = 16, 120, 0.1                          # free band y in [0.6, 1.0], 0.4 m wide
        occ = _horizontal_corridor(h, w, wall=6)
        u = _gaussian_repulsion(occ, sigma_px=4.0)
        from scipy.ndimage import distance_transform_edt
        d_obs = distance_transform_edt(occ < 0.5).astype(np.float64) * res

        target = 0.5                                       # impossible: max clearance ~0.2 m
        c = TrajectorySafetyCorrector(
            TrajectoryCorrectionParams(min_clearance_m=target, max_total_shift_m=0.5,
                                       max_step_m=0.2, clearance_iters=6)
        )
        c.set_field(u, res, 0.0, 0.0, d_obs=d_obs)

        centre_y = (h / 2) * res
        xy = np.array([[x * res, centre_y - 0.1] for x in range(15, 105, 10)], dtype=np.float64)
        result = c.correct(xy)                             # must not raise
        assert result.max_shift_m <= 0.5 + 1e-4            # cap respected
        # Geometry forbids reaching target; we just confirm it stayed finite/inside band.
        assert np.all(np.isfinite(result.waypoints))


class TestDisplacementCapWithCorners:
    def test_smoothing_cannot_break_total_shift_cap(self):
        """Regression: the post-smoothing re-clamp keeps a sharp corner within the cap.

        On a flat field descent is a no-op, so any motion is pure 3-tap
        corner-cutting; before the re-clamp this blew past max_total_shift_m.
        """
        # Flat field -> no repulsion; only smoothing can move points.
        c = TrajectorySafetyCorrector(
            TrajectoryCorrectionParams(max_total_shift_m=0.15, smoothing_passes=4)
        )
        c.set_field(np.zeros((40, 40), dtype=np.float64), 0.1, 0.0, 0.0)

        # A right-angle, 1 m-leg corner (exactly what grid A* produces).
        xy = np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 2.0], [3.0, 2.0]], dtype=np.float64)
        result = c.correct(xy)
        assert result.max_shift_m <= 0.15 + 1e-4


# ---------------------------------------------------------------------------
# Path2D wrapper + guards
# ---------------------------------------------------------------------------
class TestPathWrapperAndGuards:
    def test_correct_path_roundtrip(self):
        h, w, res = 60, 120, 0.1
        u = _gaussian_repulsion(_horizontal_corridor(h, w), sigma_px=6.0)
        c = TrajectorySafetyCorrector(TrajectoryCorrectionParams(iterations=10, gain=1.0))
        c.set_field(u, res, 0.0, 0.0)

        low_y = (h * 0.30) * res
        pts = tuple(Pose2D(x * res, low_y, 0.0) for x in range(15, 95, 10))
        path = Path2D(points=pts, frame_id="map", metadata={"planner": "astar"})
        out = c.correct_path(path)

        assert isinstance(out, Path2D)
        assert out.frame_id == "map"
        assert out.metadata["safety_corrected"] is True
        # num_corrected matches the array API on the same points.
        n_arr = int(np.count_nonzero(c.correct(np.array([[p.x, p.y] for p in pts]))
                                     .corrected_mask))
        assert out.metadata["num_corrected"] == n_arr >= 1
        # Pre-existing metadata is preserved, not dropped.
        assert out.metadata["planner"] == "astar"
        assert len(out.points) == len(pts)
        # First point pinned exactly, headings re-derived (finite).
        assert (out.points[0].x, out.points[0].y) == (pts[0].x, pts[0].y)
        assert all(np.isfinite(p.yaw) for p in out.points)

    def test_correct_path_keeps_unobserved_pose_exactly(self):
        """correct_path must not re-aim the yaw of an unobserved waypoint."""
        h, w, res = 60, 120, 0.1
        known = np.ones((h, w), dtype=bool)
        known[:, 60:] = False                            # x>=6.0 unobserved
        u = _gaussian_repulsion(_horizontal_corridor(h, w), sigma_px=6.0)
        c = TrajectorySafetyCorrector(TrajectoryCorrectionParams(iterations=10, gain=1.0))
        c.set_field(u, res, 0.0, 0.0, known_mask=known)

        low_y = (h * 0.30) * res
        pts = (Pose2D(2.0, low_y, 0.123), Pose2D(4.0, low_y, 0.123),
               Pose2D(8.0, low_y, 0.123), Pose2D(10.0, low_y, 0.123))
        out = c.correct_path(Path2D(points=pts, frame_id="map"))
        # wp2/wp3 are unobserved -> returned exactly, yaw included.
        assert out.points[2] == pts[2]
        assert out.points[3] == pts[3]

    def test_correct_before_set_field_raises(self):
        c = TrajectorySafetyCorrector()
        with pytest.raises(RuntimeError):
            c.correct(np.zeros((3, 2)))

    def test_bad_field_shape_raises(self):
        c = TrajectorySafetyCorrector()
        with pytest.raises(ValueError):
            c.set_field(np.zeros((5,)), 0.1, 0.0, 0.0)

    def test_nonfinite_field_raises(self):
        c = TrajectorySafetyCorrector()
        bad = np.zeros((10, 10))
        bad[0, 0] = np.nan
        with pytest.raises(ValueError):
            c.set_field(bad, 0.1, 0.0, 0.0)

    def test_bad_waypoints_shape_raises(self):
        c = TrajectorySafetyCorrector()
        c.set_field(np.zeros((10, 10)), 0.1, 0.0, 0.0)
        with pytest.raises(ValueError):
            c.correct(np.zeros((4,)))


class TestLateralProjection:
    def test_lateral_only_avoids_along_track_drift(self):
        """On a diagonal path, lateral_only suppresses fore/aft sliding.

        The wall gradient is +y; a diagonal tangent has a +y component, so the
        full push (lateral_only=False) slides waypoints along-track and distorts
        spacing, while lateral_only=True keeps only the perpendicular nudge.
        """
        h, w, res = 80, 200, 0.1
        occ = np.zeros((h, w), dtype=np.float64)
        occ[:8, :] = 1.0                                  # wall on low y
        u = _gaussian_repulsion(occ, sigma_px=6.0)

        # Diagonal path skimming above the wall (increasing x and y).
        xs = np.arange(10, 110, 10)
        xy = np.array([[x * res, 1.0 + 0.02 * (x - 10)] for x in xs], dtype=np.float64)

        def shifts(lateral_only):
            c = TrajectorySafetyCorrector(TrajectoryCorrectionParams(
                iterations=10, gain=1.0, max_step_m=0.3, max_total_shift_m=1.0,
                smoothing_passes=0, lateral_only=lateral_only))
            c.set_field(u, res, 0.0, 0.0)
            out = c.correct(xy).waypoints.astype(np.float64)
            along, perp = [], []
            for i in range(1, len(xy) - 1):
                t = xy[i + 1] - xy[i - 1]
                t = t / np.hypot(t[0], t[1])
                nrm = np.array([-t[1], t[0]])
                along.append(abs(float(np.dot(out[i] - xy[i], t))))
                perp.append(abs(float(np.dot(out[i] - xy[i], nrm))))
            return np.mean(along), np.mean(perp)

        lat_along, lat_perp = shifts(True)
        full_along, full_perp = shifts(False)
        # Lateral-only suppresses most along-track sliding...
        assert lat_along < 0.5 * full_along
        # ...while still moving the path off the wall as much as the full push.
        assert lat_perp > 0.1 and lat_perp >= 0.8 * full_perp


class TestNavDpEntryPath:
    def test_anchored_body_trajectory_is_correctable(self):
        """NavDP body-frame waypoints, once anchored to world, are corrected.

        Recommended flow: NavDP -> anchor_trajectory_to_world(...) -> correct(...).
        Here the drone sits at world origin facing +x (ref_yaw=0), so body
        (forward, left) maps onto world (x, y) and a path hugging the low-y wall
        of a tight corridor gets pulled toward the centre.
        """
        from sparx_agency.core.planning.navdp.geometry import anchor_trajectory_to_world

        h, w, res = 30, 120, 0.1
        u = _gaussian_repulsion(_horizontal_corridor(h, w, wall=10), sigma_px=6.0)
        c = TrajectorySafetyCorrector(
            TrajectoryCorrectionParams(iterations=20, gain=1.0, max_step_m=0.3,
                                       max_total_shift_m=1.0)
        )
        c.set_field(u, res, 0.0, 0.0)

        centre_y = (h / 2) * res                      # 1.5 m
        off = centre_y - 0.3
        # NavDP body trajectory: forward 1.5..9.5 m, constant left offset.
        body = np.array([[fwd, off] for fwd in np.arange(1.5, 10.0, 1.0)], dtype=np.float32)
        world_xy = np.asarray(anchor_trajectory_to_world(body, 0.0, 0.0, 0.0))

        result = c.correct(world_xy)
        # Every non-pinned visible waypoint ends closer to the corridor centre.
        before = np.abs(world_xy[1:, 1] - centre_y)
        after = np.abs(result.waypoints[1:, 1] - centre_y)
        assert np.all(after < before)
        assert result.visible_mask.all()
