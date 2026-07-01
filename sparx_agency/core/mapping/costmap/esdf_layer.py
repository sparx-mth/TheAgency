# ``from __future__ import annotations`` keeps the ``tuple[...]`` style return
# annotations from being evaluated at import time, so this module is importable
# under Python 3.8 — the FALCON Noetic adapter imports it to build the ESDF that
# feeds the ESDF path corrector (see CLAUDE.md: core must stay 3.8).
"""Euclidean distance field (ESDF) from an occupancy probability grid.

Computes ``D(x)`` = distance to the nearest obstacle in metres, via an exact
Euclidean distance transform (cv2). This is the field an *ESDF path corrector*
ascends: its gradient ``+∇D`` points away from the nearest wall toward open
space, so following it nudges a planned route toward the centre of a corridor or
doorway and away from walls (the FALCON B-spline ``safe_distance`` idea applied to
a path).

It is the distance-transform sibling of
:class:`sparx_agency.core.mapping.costmap.potential_field_layer.PotentialFieldLayer`
(which builds a Gaussian *repulsive* potential). Kept separate (single
responsibility) so each correction strategy owns its own field generator. Unknown
cells (NaN) are treated as free by default, exactly like the potential layer, so
``D`` is drawn only from KNOWN walls.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception as e:  # pragma: no cover - exercised only where cv2 is absent
    cv2 = None
    _cv2_import_error = e


class EsdfLayer:
    """Distance-to-nearest-obstacle field (metres) from a probability grid.

    Args:
        occ_thresh: Probability at/above which a cell counts as an obstacle.
        smooth_sigma_m: Optional Gaussian blur (metres) applied to the distance
            field for a cleaner, less staircased gradient. 0 disables it.
        unknown_as_obstacle: If True, NaN (unknown) cells count as obstacles;
            default False treats them as free (matches the planner/potential layer).
    """

    def __init__(
        self,
        occ_thresh: float = 0.65,
        smooth_sigma_m: float = 0.0,
        unknown_as_obstacle: bool = False,
    ) -> None:
        if cv2 is None:
            raise RuntimeError(
                "EsdfLayer requires OpenCV (cv2). Import error: %s" % (_cv2_import_error,))
        self.occ_thresh = float(occ_thresh)
        self.smooth_sigma_m = float(smooth_sigma_m)
        self.unknown_as_obstacle = bool(unknown_as_obstacle)

    def compute_from_prob_grid(self, p_occ: np.ndarray, resolution_m: float) -> np.ndarray:
        """Return the ``(H, W)`` float32 distance-to-obstacle field in metres.

        Args:
            p_occ: ``(H, W)`` occupancy probability in ``[0, 1]``; unknown may be NaN.
            resolution_m: Metres per cell.
        """
        p = np.asarray(p_occ, dtype=np.float32)
        if p.ndim != 2:
            raise ValueError("Expected 2D grid, got shape=%s" % (p.shape,))
        res = float(resolution_m)

        is_unknown = ~np.isfinite(p)
        is_occ = p >= self.occ_thresh
        if self.unknown_as_obstacle:
            is_occ = is_occ | is_unknown
        else:
            is_occ = is_occ & (~is_unknown)

        # cv2.distanceTransform measures each non-zero pixel's distance to the
        # nearest zero pixel, so obstacles must be 0 and free space non-zero.
        free_mask = (~is_occ).astype(np.uint8) * 255
        d_pix = cv2.distanceTransform(free_mask, distanceType=cv2.DIST_L2, maskSize=5)
        d_m = d_pix.astype(np.float32) * res

        if self.smooth_sigma_m > 0.0:
            sigma_px = max(self.smooth_sigma_m / res, 1e-3)
            d_m = cv2.GaussianBlur(d_m, (0, 0), sigmaX=sigma_px, sigmaY=sigma_px)
        return d_m.astype(np.float32)
