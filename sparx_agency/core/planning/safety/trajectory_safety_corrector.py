"""
Potential-field trajectory safety correction (ROS-free, numpy-only).

This module nudges an already-planned trajectory (from A*, NavDP, or any other
source) away from walls and toward the centre of free space, using the
*repulsive potential field* produced by
:class:`sparx_agency.core.mapping.costmap.potential_field_layer.PotentialFieldLayer`.

Why a potential field centres the path
---------------------------------------
``U_rep`` is a Gaussian blur of the obstacle mask, so the value at any free cell
is the proximity-weighted sum of *every* nearby wall. Descending ``-∇U_rep``
therefore moves a waypoint away from all surrounding obstacles at once: against a
single wall it pushes perpendicular; inside a tight corridor opposing walls
cancel and the minimum sits on the centre-line. We exploit that to recentre the
path. (In a wide corridor the centre is a broad flat basin, so the field pulls
the path into the safe middle band rather than onto an exact centre-line.)

Improvements over a plain gradient-descent corrector
----------------------------------------------------
* **See-only-what-you-see.** Waypoints outside the field, or on cells flagged
  unobserved via ``known_mask``, are skipped *and held exactly fixed through
  every stage* (descent, clearance, smoothing) — the corrector only ever edits
  the portion of the trajectory the live map can support. ``visible_mask``
  reports which waypoints those were.
* **Lateral-only nudging** (default). Each push is projected perpendicular to
  the local path tangent, so recentring does not slide waypoints fore/aft and
  distort path spacing — it purely moves the path off the wall.
* **Optional clearance push (best-effort).** With a distance field and
  ``min_clearance_m > 0`` the corrector pushes each visible waypoint toward a
  minimum distance-to-obstacle. It is best-effort, not a hard guarantee: it is
  bounded by the total-displacement cap, and a waypoint on a distance-field
  plateau (a corridor narrower than ``2·min_clearance_m``, or deep inside a
  wall) is left as clear as it could get. Treat the dedicated
  :class:`TrajectorySafetyChecker` as the hard collision gate.
* **Two centring strategies** (``centering``). ``"descent"`` (default) is the
  iterative, gain-scaled gradient descent described above. ``"line_search"``
  samples along the path normal and moves each waypoint straight to the point of
  MAXIMUM clearance -- the medial axis, equidistant from the surrounding walls
  (the exact corridor centre and the farthest-from-walls line; it falls back to
  the repulsion minimum when no distance field is supplied). It is
  omnidirectional, single-pass and scale-free, dominates the A* path within the
  search range, cannot push a waypoint into a wall (clearance only drops past the
  centre), swings wide around corners (more so with ``corner_swing``), and needs
  none of the descent's gain tuning.

Frame contract
--------------
Waypoints and the field's ``origin``/``resolution`` must be in the *same* metric
frame; see :class:`PotentialFieldSampler` for the exact BEV convention. A NavDP
body-frame trajectory should first be anchored to that frame with
:func:`sparx_agency.core.planning.navdp.geometry.anchor_trajectory_to_world`.

Python 3.8 compatible (the FALCON Noetic adapter imports ``core`` under 3.8):
no PEP 604 unions, no ``match``/``case``; numpy-only at import time.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from sparx_agency.core.common.types import Path2D, Pose2D

from .potential_field_sampler import PotentialFieldSampler
from .types import TrajectoryCorrectionParams, TrajectoryCorrectionResult


class TrajectorySafetyCorrector:
    """Recentre a BEV trajectory away from walls using a repulsive field.

    Usage::

        corrector = TrajectorySafetyCorrector()
        u_rep, d_obs = potential_layer.compute_from_prob_grid(p_occ, res_m)
        corrector.set_field(u_rep, res_m, origin_x, origin_y, d_obs=d_obs)
        result = corrector.correct(waypoints_xy)        # (N, 2) array, or
        new_path = corrector.correct_path(path2d)       # Path2D -> Path2D
    """

    def __init__(self, params: Optional[TrajectoryCorrectionParams] = None) -> None:
        """Initialise with tuning parameters (defaults if ``params`` is None)."""
        self.p = params or TrajectoryCorrectionParams()
        self._field: Optional[PotentialFieldSampler] = None

    @property
    def params(self) -> TrajectoryCorrectionParams:
        """Active tuning parameters."""
        return self.p

    def set_field(
        self,
        u_rep: np.ndarray,
        resolution_m: float,
        origin_x: float,
        origin_y: float,
        *,
        d_obs: Optional[np.ndarray] = None,
        known_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Install the repulsive field to correct against.

        Thin wrapper around :class:`PotentialFieldSampler`; see it for argument
        and frame details. Call once per fresh map before :meth:`correct`.
        """
        self._field = PotentialFieldSampler(
            u_rep, resolution_m, origin_x, origin_y,
            d_obs=d_obs, known_mask=known_mask,
        )

    def set_sampler(self, field: PotentialFieldSampler) -> None:
        """Install a pre-built :class:`PotentialFieldSampler` directly."""
        self._field = field

    @property
    def field(self) -> Optional[PotentialFieldSampler]:
        """The installed field sampler (``None`` until ``set_field``/``set_sampler``).

        Exposed so callers can sample the same field the corrector uses -- e.g. the
        repulsive force ``F_rep = -grad U_rep`` via ``field.descent(x, y)``, or the
        clearance via ``field.clearance(x, y)`` -- at each waypoint, for diagnostics
        and visualisation.
        """
        return self._field

    # ------------------------------------------------------------------
    # Public correction API
    # ------------------------------------------------------------------
    def correct(self, waypoints: Sequence[Sequence[float]]) -> TrajectoryCorrectionResult:
        """Correct an ``(N, >=2)`` array of ``(x, y[, ...])`` waypoints.

        Extra columns beyond ``(x, y)`` are ignored; the result holds the
        corrected ``(x, y)`` only. Waypoints outside the observed field are
        returned unchanged (see ``visible_mask`` on the result).

        Note: the result is float32, so an "unchanged" waypoint is returned
        float32-exact (not bit-exact for non-float32 inputs). :meth:`correct_path`
        keeps pinned/unobserved waypoints bit-exact.

        Raises:
            RuntimeError: If no field has been installed.
            ValueError: If ``waypoints`` is not ``(N, >=2)``.
        """
        if self._field is None:
            raise RuntimeError("set_field(...) must be called before correct().")

        wp = np.asarray(waypoints, dtype=np.float64)
        if wp.ndim != 2 or wp.shape[1] < 2:
            raise ValueError(f"waypoints must be (N, >=2), got shape {wp.shape}")

        orig = wp[:, :2].copy()
        out, visible, shift = self._correct_xy(orig)
        return TrajectoryCorrectionResult(
            waypoints=out.astype(np.float32),
            corrected_mask=shift > 1e-4,
            visible_mask=visible,
            max_shift_m=float(shift.max()) if shift.size else 0.0,
        )

    def correct_path(self, path: Path2D) -> Path2D:
        """Correct a :class:`Path2D`, preserving frame and re-deriving headings.

        Pinned waypoints (the leading ``pin_first_k`` and, if ``pin_last``, the
        goal) are returned exactly as given; the rest are re-aimed along the
        corrected geometry. ``metadata`` gains ``safety_corrected`` and
        ``num_corrected`` keys.
        """
        pts = path.points                                 # Path2D guarantees >= 2 points
        xy = np.array([[p.x, p.y] for p in pts], dtype=np.float64)
        out, visible, shift = self._correct_xy(xy)
        yaws = self._recompute_yaw([p.yaw for p in pts], out)

        n = len(pts)
        k = self._pin_k()
        new_points: List[Pose2D] = []
        for i in range(n):
            pinned = i < k or (self.p.pin_last and i == n - 1)
            if pinned or not visible[i]:
                new_points.append(pts[i])                 # pinned/unobserved: keep exactly
            else:
                new_points.append(Pose2D(float(out[i, 0]), float(out[i, 1]), yaws[i]))

        meta = dict(path.metadata)
        meta["safety_corrected"] = True
        meta["num_corrected"] = int(np.count_nonzero(shift > 1e-4))
        return Path2D(points=tuple(new_points), frame_id=path.frame_id, metadata=meta)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def _pin_k(self) -> int:
        """Number of pinned leading waypoints (≥1: waypoint 0 is the drone)."""
        return max(1, self.p.pin_first_k)

    def _correct_xy(self, orig: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the full correction on an ``(N, 2)`` float64 array.

        Stage order matters: smoothing is a Laplacian corner-cut that ignores
        the displacement cap, so it must be followed by a re-clamp; clearance is
        enforced *last* (then re-clamped) so smoothing cannot pull a waypoint
        back below ``min_clearance_m``. ``_freeze`` is the final word, keeping
        pinned/unobserved waypoints exactly on input through every stage.

        Returns ``(corrected_xy, visible_mask, per_waypoint_shift)``.
        """
        visible = self._visibility(orig)
        if self.p.centering == "line_search":
            # Direct medial-axis centring. The search is per-waypoint bounded and
            # lands on the smooth maximum-clearance line, so centring itself needs
            # neither the displacement clamp nor the smoothing pass -- both would
            # only drag the path back toward the wall-hugging input.
            out = self._center_line_search(orig.copy(), visible)
            # Clearance FLOOR (best-effort), same step as the descent branch.
            # Centring lands on the NEAREST local clearance maximum, which against
            # a lone wall or in an open room can still sit close to it (the walk
            # stops at the first prominent peak by design). With min_clearance_m>0
            # push each visible waypoint out to that distance-to-wall along
            # +grad D_obs, bounded by max_total_shift_m. Without this the
            # min_clearance_m knob was silently a no-op in line_search mode, so a
            # large value barely moved the path off the walls. ``_freeze`` then
            # re-pins the fixed (pinned / unobserved) waypoints.
            if self.p.min_clearance_m > 0.0 and self._field.has_distance:
                self._enforce_clearance(out, visible)
                self._clamp_total_shift(out, orig)
            self._freeze(out, orig, visible)
            return out, visible, np.linalg.norm(out - orig, axis=1)

        out = self._descend(orig.copy(), visible)
        self._clamp_total_shift(out, orig)
        self._smooth(out, visible)
        self._clamp_total_shift(out, orig)                # cap smoothing's corner-cut
        if self.p.min_clearance_m > 0.0 and self._field.has_distance:
            self._enforce_clearance(out, visible)
            self._clamp_total_shift(out, orig)
        self._freeze(out, orig, visible)
        return out, visible, np.linalg.norm(out - orig, axis=1)

    def _visibility(self, pts: np.ndarray) -> np.ndarray:
        """Mark waypoints inside the observed field (evaluated at input positions).

        Independent of pinning: it answers "what can the live map see right
        now", so a pinned-but-in-field waypoint still reads as visible.
        """
        visible = np.zeros(pts.shape[0], dtype=bool)
        for i in range(pts.shape[0]):
            visible[i] = self._field.is_observed(pts[i, 0], pts[i, 1])
        return visible

    def _descend(self, out: np.ndarray, visible: np.ndarray) -> np.ndarray:
        """Iterative, proximity-weighted gradient descent with a decaying step."""
        n = out.shape[0]
        hi = n - 1 if self.p.pin_last else n
        step = 1.0
        for _ in range(max(0, self.p.iterations)):
            nxt = out.copy()
            max_moved = 0.0
            for i in range(self._pin_k(), hi):
                if not visible[i]:
                    continue                      # not observable -> never touch
                u = self._field.potential(out[i, 0], out[i, 1])
                if u is None or u < self.p.u_floor:
                    continue                      # off-map this pass, or clear of walls
                push = self._field.descent(out[i, 0], out[i, 1]) * (self.p.gain * step)
                if self.p.lateral_only:
                    push = self._project_lateral(push, out, i, n)
                mag = float(np.hypot(push[0], push[1]))
                if mag < 1e-6:
                    continue
                if mag > self.p.max_step_m:
                    push = push * (self.p.max_step_m / mag)
                nxt[i] = out[i] + push
                max_moved = max(max_moved, float(np.hypot(nxt[i, 0] - out[i, 0],
                                                          nxt[i, 1] - out[i, 1])))
            out = nxt
            step *= self.p.step_decay
            if max_moved < 1e-4:
                break
        return out

    @staticmethod
    def _unit_tangent(pts: np.ndarray, i: int, n: int) -> Optional[np.ndarray]:
        """Unit tangent of the path at ``i`` (central difference), or None."""
        if 0 < i < n - 1:
            tan = pts[i + 1] - pts[i - 1]
        elif i + 1 < n:
            tan = pts[i + 1] - pts[i]
        elif i > 0:
            tan = pts[i] - pts[i - 1]
        else:
            return None
        norm = float(np.hypot(tan[0], tan[1]))
        if norm < 1e-9:
            return None
        return tan / norm

    @classmethod
    def _project_lateral(cls, push: np.ndarray, pts: np.ndarray, i: int, n: int) -> np.ndarray:
        """Remove the along-tangent component of ``push`` (keep only sideways)."""
        t = cls._unit_tangent(pts, i, n)
        if t is None:
            return push
        return push - float(np.dot(push, t)) * t

    @classmethod
    def _unit_normal(cls, pts: np.ndarray, i: int, n: int) -> Optional[np.ndarray]:
        """Unit normal (left of travel) to the path tangent at ``i``, or None."""
        t = cls._unit_tangent(pts, i, n)
        if t is None:
            return None
        return np.array([-t[1], t[0]])

    def _center_line_search(self, out: np.ndarray, visible: np.ndarray) -> np.ndarray:
        """Move each visible waypoint laterally to the maximum-clearance point.

        Along the path NORMAL, sample the distance-to-obstacle field and move the
        waypoint to its MAXIMUM -- the medial axis: the point equidistant from the
        surrounding walls (the exact corridor centre, and the farthest-from-walls
        point on that line). A 3-point parabolic fit refines the discrete maximum.
        This centres the path directly -- no gain, no convergence loop -- and, by
        seeking clearance rather than a gradient, it dominates the A* path within
        the search range, cannot push a waypoint into a wall (clearance only drops
        past the centre), and at a corner the medial axis bulges away from the
        inside wall so the path swings wide. With no distance field it falls back
        to minimising the repulsive potential (same maths on ``-U_rep``), and is
        scale-free either way. It walks outward to the NEAREST local maximum -- so
        an already-centred waypoint stays put and a wide search range never makes a
        waypoint jump to a farther, larger opening; an unobserved / off-map cell
        stops the walk. The lateral range is widened at sharp corners
        (``corner_swing``) so corners may swing wider than straight runs.
        """
        n = out.shape[0]
        hi = n - 1 if self.p.pin_last else n
        step = max(float(self.p.center_step_m), 1e-3)
        use_clear = self._field.has_distance
        # Snapshot the input geometry: normals are taken from the ORIGINAL path, not
        # the partially-updated one. Otherwise moving an early waypoint skews the
        # normal of later ones (a straight path turns "diagonal"), sending their
        # lateral search the wrong way -- the main cause of erratic/weak centring.
        ref = out.copy()
        for i in range(self._pin_k(), hi):
            if not visible[i]:
                continue
            nrm = self._unit_normal(ref, i, n)
            if nrm is None:
                continue
            v0 = self._score(out[i, 0], out[i, 1], use_clear)
            if v0 is None:
                continue
            span = float(self.p.max_total_shift_m) * self._corner_span_factor(ref, i, n)
            n_side = max(1, int(span / step))
            # Walk outward each way to the NEAREST local score maximum (the current
            # corridor's medial axis), stopping at the first sustained decrease or
            # an unobserved / off-map cell. Cost then scales with the corridor
            # width, not the (possibly large) search range, and the waypoint never
            # jumps down its normal to a farther, larger opening.
            off_p, v_p = self._walk_local_max(out[i], nrm, +1.0, v0, n_side, step, use_clear)
            off_m, v_m = self._walk_local_max(out[i], nrm, -1.0, v0, n_side, step, use_clear)
            if v_p >= v_m and v_p > v0:
                off = self._refine_peak(out[i], nrm, off_p, step, use_clear)
            elif v_m > v0:
                off = self._refine_peak(out[i], nrm, off_m, step, use_clear)
            else:
                off = 0.0                          # already at a local max -> stay put
            if abs(off) < 1e-4:
                continue
            out[i] = out[i] + nrm * off
        return out

    def _walk_local_max(self, p, nrm, direction, v0, n_side, step, use_clear):
        """Walk from ``p`` along ``direction*nrm`` to the nearest PROMINENT maximum.

        Returns ``(best_offset, best_score)``. Stops once the score has fallen a
        real fraction of its climb below the running peak -- so sub-cell ripples in
        the (approximate) distance transform are ignored and the walk reaches the
        true centre -- or at an unobserved / off-map cell, or the range limit. The
        prominence fraction of the climb is scale-free, and the wall on the far
        side (where clearance dips to ~0) guarantees the walk stops at the current
        corridor's centre rather than running on to a larger opening beyond it.
        """
        best_off, best_v = 0.0, v0
        for k in range(1, n_side + 1):
            off = direction * k * step
            x = p[0] + nrm[0] * off
            y = p[1] + nrm[1] * off
            if not self._field.is_observed(x, y):
                break
            v = self._score(x, y, use_clear)
            if v is None:
                break
            if v > best_v:
                best_v, best_off = v, off
            else:
                climb = best_v - v0
                if climb > 1e-12 and v < best_v - 0.25 * climb:
                    break                          # descended a real amount -> past the peak
        return best_off, best_v

    def _refine_peak(self, p, nrm, off, step, use_clear):
        """3-point parabolic sub-sample refinement of a score maximum at ``off``."""
        def s(o):
            return self._score(p[0] + nrm[0] * o, p[1] + nrm[1] * o, use_clear)
        vm, vc, vp = s(off - step), s(off), s(off + step)
        if vm is None or vc is None or vp is None:
            return off
        denom = vm - 2.0 * vc + vp
        # Concavity guard relative to the local magnitude, so the refinement fires
        # identically at any field scale (the vertex ratio is already scale-free).
        ref = max(abs(vm), abs(vc), abs(vp), 1e-30)
        if denom < -1e-9 * ref:                     # concave (a genuine maximum)
            return off + float(np.clip(0.5 * (vm - vp) / denom, -1.0, 1.0)) * step
        return off

    def _score(self, x: float, y: float, use_clear: bool):
        """Centrality score at world ``(x, y)`` -- higher = more central.

        Distance-to-obstacle (clearance) when a distance field is available, so the
        maximum sits on the medial axis (and is in physical metres, hence scale
        free); otherwise the negated repulsive potential, whose maximum is the
        repulsion minimum. ``None`` outside the field.
        """
        if use_clear:
            return self._field.clearance(x, y)
        u = self._field.potential(x, y)
        return None if u is None else -u

    def _corner_span_factor(self, pts: np.ndarray, i: int, n: int) -> float:
        """Lateral-search multiplier: 1.0 on a straight run, up to ``1+corner_swing``
        at a >=90 deg turn, so sharp corners may swing wider than straights."""
        if self.p.corner_swing <= 0.0 or not (0 < i < n - 1):
            return 1.0
        a = pts[i] - pts[i - 1]
        b = pts[i + 1] - pts[i]
        na = float(np.hypot(a[0], a[1]))
        nb = float(np.hypot(b[0], b[1]))
        if na < 1e-9 or nb < 1e-9:
            return 1.0
        cos_t = float(np.dot(a, b) / (na * nb))
        turn = float(np.arccos(np.clip(cos_t, -1.0, 1.0)))      # 0=straight..pi=U-turn
        return 1.0 + self.p.corner_swing * min(turn / (np.pi / 2.0), 1.0)

    def _enforce_clearance(self, out: np.ndarray, visible: np.ndarray) -> None:
        """Best-effort push of visible waypoints toward ``min_clearance_m``.

        Steps along ``+∇D_obs`` (unit-magnitude eikonal gradient) so a step of
        ``target - d`` closes the deficit in ≈ one iteration. Not a hard
        guarantee: a waypoint on a distance-field plateau (deep inside a wall, or
        on the medial axis of a corridor narrower than ``2·target``) has a
        vanishing gradient and is left as close as it could get, without error.
        """
        target = self.p.min_clearance_m
        n = out.shape[0]
        hi = n - 1 if self.p.pin_last else n
        for i in range(self._pin_k(), hi):
            if not visible[i]:
                continue
            for _ in range(max(0, self.p.clearance_iters)):
                d = self._field.clearance(out[i, 0], out[i, 1])
                if d is None or d >= target:
                    break
                g = self._field.clearance_ascent(out[i, 0], out[i, 1])
                mag = float(np.hypot(g[0], g[1])) if g is not None else 0.0
                if mag < 1e-6:
                    break
                out[i] = out[i] + (g / mag) * min(self.p.max_step_m, target - d)

    def _clamp_total_shift(self, out: np.ndarray, orig: np.ndarray) -> None:
        """Scale back any waypoint that drifted beyond ``max_total_shift_m``."""
        delta = out - orig
        dist = np.linalg.norm(delta, axis=1)
        over = dist > self.p.max_total_shift_m
        if np.any(over):
            scale = self.p.max_total_shift_m / np.maximum(dist[over], 1e-9)
            out[over] = orig[over] + delta[over] * scale[:, None]

    def _smooth(self, out: np.ndarray, visible: np.ndarray) -> None:
        """In-place 3-tap smoothing of de-kinked waypoints.

        Only a waypoint whose entire 3-tap window is observed and that is itself
        free to move is blended, so off-map / unobserved / pinned waypoints are
        never dragged and the smoother never reaches across the map edge.
        (A later ``_clamp_total_shift`` re-bounds the corner-cut, and ``_freeze``
        re-pins frozen waypoints exactly.)
        """
        n = out.shape[0]
        if n < 3 or self.p.smoothing_passes <= 0:
            return
        movable = visible.copy()
        movable[: self._pin_k()] = False
        if self.p.pin_last:
            movable[-1] = False
        sm = (movable[1:-1] & visible[:-2] & visible[2:])
        for _ in range(self.p.smoothing_passes):
            blended = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
            out[1:-1][sm] = blended[sm]

    def _freeze(self, out: np.ndarray, orig: np.ndarray, visible: np.ndarray) -> None:
        """Final hard guarantee: pinned and unobserved waypoints == input exactly."""
        n = out.shape[0]
        if n == 0:
            return
        frozen = ~visible
        frozen[: self._pin_k()] = True
        if self.p.pin_last:
            frozen[-1] = True
        out[frozen] = orig[frozen]

    def _recompute_yaw(self, orig_yaw: List[float], new_xy: np.ndarray) -> List[float]:
        """Keep yaw on the pinned prefix; re-aim the rest along corrected geometry."""
        n = len(orig_yaw)
        yaws = list(orig_yaw)
        k = self._pin_k()
        for i in range(k, n):
            j = i + 1 if i + 1 < n else i
            a = i if i + 1 < n else i - 1
            dx = new_xy[j, 0] - new_xy[a, 0]
            dy = new_xy[j, 1] - new_xy[a, 1]
            if abs(dx) > 1e-9 or abs(dy) > 1e-9:
                yaws[i] = float(np.arctan2(dy, dx))
        return yaws
