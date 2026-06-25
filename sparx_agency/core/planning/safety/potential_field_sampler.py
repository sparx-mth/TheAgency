"""
Bilinear sampling of a repulsive potential field in world coordinates.

This is the *frame + sampling* half of potential-field trajectory correction,
split out from :class:`TrajectorySafetyCorrector` so the corrector holds only
the descent algorithm and so any other consumer (trackers, the safety checker)
can sample the same field the same way.

A :class:`PotentialFieldSampler` wraps the field produced by
:class:`sparx_agency.core.mapping.costmap.potential_field_layer.PotentialFieldLayer`
(``U_rep`` and optionally ``D_obs``) plus its grid metadata, and answers queries
at arbitrary world ``(x, y)`` points:

* :meth:`potential` — bilinear ``U_rep`` (``None`` outside the field).
* :meth:`descent` — bilinear descent direction ``-∇U_rep`` as ``[x, y]``.
* :meth:`clearance` — bilinear distance-to-obstacle in metres.
* :meth:`is_observed` — is the point inside the field *and* on an observed cell.

Frame convention (standard planning / BEV; matches ``OccupancyGrid2D`` and
``GridSpec``)::

    col = (x - origin_x) / resolution_m      # field column ↔ world x
    row = (y - origin_y) / resolution_m      # field row    ↔ world y
    field is indexed [row, col]              # i.e. field[gy, gx]

Python 3.8 compatible (the FALCON Noetic adapter imports ``core`` under 3.8):
no PEP 604 unions, no ``match``/``case``; numpy-only at import time.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class PotentialFieldSampler:
    """Bilinear world-coordinate sampler over a repulsive potential field."""

    def __init__(
        self,
        u_rep: np.ndarray,
        resolution_m: float,
        origin_x: float,
        origin_y: float,
        *,
        d_obs: Optional[np.ndarray] = None,
        known_mask: Optional[np.ndarray] = None,
    ) -> None:
        """Validate the field and precompute its world-frame descent gradient.

        Args:
            u_rep: ``(H, W)`` repulsive potential. Must be finite — flag
                unobserved regions with ``known_mask``, not NaN.
            resolution_m: Metres per cell (> 0).
            origin_x, origin_y: World coordinates of field cell ``(0, 0)``.
            d_obs: Optional ``(H, W)`` distance-to-obstacle (metres).
            known_mask: Optional ``(H, W)`` bool grid; ``False`` cells are
                treated as unobserved.

        Raises:
            ValueError: On a non-2D / too-small / non-finite field, a
                non-positive resolution, or shape-mismatched auxiliaries.
        """
        u = np.asarray(u_rep, dtype=np.float64)
        if u.ndim != 2 or u.shape[0] < 2 or u.shape[1] < 2:
            raise ValueError(f"u_rep must be a 2D grid at least 2x2, got {u.shape}")
        if not np.all(np.isfinite(u)):
            raise ValueError(
                "u_rep contains non-finite values; mark unobserved cells via "
                "known_mask and pass a finite potential field."
            )
        if resolution_m <= 0.0:
            raise ValueError("resolution_m must be > 0")

        self._u = u
        self._res = float(resolution_m)
        self._origin_x = float(origin_x)
        self._origin_y = float(origin_y)

        # np.gradient(u, res) -> (dU/drow, dU/dcol). Descent is -∇U; stored as
        # [x-component (-dU/dcol), y-component (-dU/drow)] since col↔x, row↔y.
        g_row, g_col = np.gradient(u, self._res)
        self._grad = np.stack([-g_col, -g_row], axis=-1)

        self._d = self._coerce(d_obs, "d_obs", np.float64)
        self._known = self._coerce(known_mask, "known_mask", bool)

        # Ascending gradient of the distance field (+∇D_obs). The distance
        # transform is eikonal, so |∇D_obs| ≈ 1 away from medial axes — moving a
        # waypoint by ``s`` along this raises its clearance by ≈ ``s``, which
        # makes the clearance push converge in (almost) one step. None when no
        # distance field was supplied.
        if self._d is None:
            self._grad_d = None
        else:
            gd_row, gd_col = np.gradient(self._d, self._res)
            self._grad_d = np.stack([gd_col, gd_row], axis=-1)

    def _coerce(self, arr: Optional[np.ndarray], name: str, dtype) -> Optional[np.ndarray]:
        """Coerce an optional auxiliary grid and check it matches the field."""
        if arr is None:
            return None
        out = np.asarray(arr, dtype=dtype)
        if out.shape != self._u.shape:
            raise ValueError(f"{name} shape {out.shape} must match u_rep {self._u.shape}")
        return out

    @property
    def has_distance(self) -> bool:
        """True if a distance-to-obstacle field was supplied."""
        return self._d is not None

    # ------------------------------------------------------------------
    # World-coordinate queries
    # ------------------------------------------------------------------
    def potential(self, x: float, y: float) -> Optional[float]:
        """Bilinear ``U_rep`` at world ``(x, y)``; ``None`` if outside the field."""
        rf, cf = self._world_to_grid(x, y)
        v = self._bilinear(self._u, rf, cf)
        return None if v is None else float(v)

    def descent(self, x: float, y: float) -> Optional[np.ndarray]:
        """Bilinear descent direction ``-∇U_rep`` as ``[x, y]``; ``None`` if OOB."""
        rf, cf = self._world_to_grid(x, y)
        return self._bilinear(self._grad, rf, cf)

    def clearance(self, x: float, y: float) -> Optional[float]:
        """Bilinear distance-to-obstacle (m); ``None`` if no field or OOB."""
        if self._d is None:
            return None
        rf, cf = self._world_to_grid(x, y)
        v = self._bilinear(self._d, rf, cf)
        return None if v is None else float(v)

    def clearance_ascent(self, x: float, y: float) -> Optional[np.ndarray]:
        """Ascent direction of the distance field ``+∇D_obs`` as ``[x, y]``.

        Points toward greater distance-to-obstacle; magnitude ≈ 1 away from
        medial axes. ``None`` if no distance field was supplied or OOB.
        """
        if self._grad_d is None:
            return None
        rf, cf = self._world_to_grid(x, y)
        return self._bilinear(self._grad_d, rf, cf)

    def is_observed(self, x: float, y: float) -> bool:
        """True if ``(x, y)`` is in-field and its whole sampling cell is observed.

        Observation is tested against the *same* floor-based 4-cell footprint
        that :meth:`potential`/:meth:`descent` interpolate over (bilinear of the
        boolean mask, requiring all contributing cells known), so the visibility
        gate can never admit a point whose gradient blends in an unobserved cell.
        """
        rf, cf = self._world_to_grid(x, y)
        if self._bilinear(self._u, rf, cf) is None:
            return False
        if self._known is None:
            return True
        k = self._bilinear(self._known, rf, cf)     # bool grid → float in [0, 1]
        return k is not None and float(k) >= 1.0 - 1e-9

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _world_to_grid(self, x: float, y: float) -> Tuple[float, float]:
        """World ``(x, y)`` → fractional grid ``(row, col)``."""
        return (y - self._origin_y) / self._res, (x - self._origin_x) / self._res

    @staticmethod
    def _bilinear(grid: np.ndarray, row_f: float, col_f: float):
        """Bilinear sample of a scalar ``(H,W)`` or vector ``(H,W,2)`` grid.

        Returns the interpolated value (float or ``(2,)`` array), or ``None`` if
        ``(row_f, col_f)`` lies outside the grid.
        """
        h, w = grid.shape[0], grid.shape[1]
        if not (0.0 <= row_f <= h - 1 and 0.0 <= col_f <= w - 1):
            return None
        r0 = min(int(np.floor(row_f)), h - 2)
        c0 = min(int(np.floor(col_f)), w - 2)
        dr = row_f - r0
        dc = col_f - c0
        top = grid[r0, c0] * (1.0 - dc) + grid[r0, c0 + 1] * dc
        bot = grid[r0 + 1, c0] * (1.0 - dc) + grid[r0 + 1, c0 + 1] * dc
        return top * (1.0 - dr) + bot * dr
