"""Evaluator-only SemExp Gibson FMM distances; never a policy planner.

Independent implementation of the documented computation in SemExp
``envs/habitat/objectgoal_env.py:116-144,290-299,404-422`` and
``envs/utils/fmm_planner.py:69-74`` (revision 5d76902).
Do not replace FMM with Euclidean distance, navmesh distance or grid A*.
"""
from __future__ import annotations

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL


class GibsonDistanceField:
    """FMM distance to a 1 m dilation of the goal category on the start floor.

    The source fills masked/unreachable cells with max(valid distance)+1 cell.
    That finite sentinel is preserved for DTG, rather than dropping failures
    from its mean. Published starts there are retained as upstream does;
    callers can audit them through start_uses_sentinel().
    """

    def __init__(self, semantic, origin_cm, category_index):
        import skfmm
        from skimage.morphology import binary_dilation, disk

        traversible = binary_dilation(semantic[0], disk(2))
        radius = int(PROTOCOL.success_radius_m / PROTOCOL.map_resolution_m)
        goal = binary_dilation(semantic[category_index + 1], disk(radius))
        level_set = np.ma.masked_values(traversible.astype(np.int32), 0)
        # Deliberately NOT intersected with traversible: SemExp unmasks goals.
        # PONI's later validate_goal variant is a different evaluator.
        level_set[goal] = 0
        distances = skfmm.distance(level_set, dx=1)
        self.unreachable = np.ma.getmaskarray(distances)
        self.cells = np.asarray(np.ma.filled(distances, np.max(distances) + 1))
        self.origin_m = np.asarray(origin_cm, dtype=np.float64) / 100.0
        if not np.isfinite(self.cells).all() or np.any(self.cells < 0):
            raise ValueError("Gibson FMM did not produce a finite distance field")

    def map_cell(self, habitat_position):
        """Raw Habitat XYZ -> (row, col), matching upstream truncation, not round.

        Map origin is (Habitat Z, Habitat X), in centimetres in val_info.pbz2.
        This is NOT our public ENU frame. Negative indices must not wrap.
        """
        x, _, z = habitat_position
        col_f = (z - self.origin_m[0]) * 20.0
        row_f = (x - self.origin_m[1]) * 20.0
        h, w = self.cells.shape
        if not (0 <= row_f < h and 0 <= col_f < w):
            raise ValueError("Habitat position is outside its Gibson floor map")
        return int(row_f), int(col_f)

    def distance(self, habitat_position, start=False):
        """Reference DTG/DTS in metres, including the upstream finite sentinel."""
        cell = self.map_cell(habitat_position)
        return float(self.cells[cell]) * PROTOCOL.map_resolution_m

    def start_uses_sentinel(self, habitat_position):
        """Whether this in-bounds published start uses the reference fill value."""
        return bool(self.unreachable[self.map_cell(habitat_position)])

