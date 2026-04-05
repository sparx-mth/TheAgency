import numpy as np
from dataclasses import dataclass
from typing import Optional

from sparx_agency.core.common.types import Observation
from sparx_agency.core.mapping.costmap.log_odds_grid import LogOddsGridCostmap
from sparx_agency.core.mapping.costmap.potential_field_layer import PotentialFieldLayer

@dataclass
class DA3MappingConfig:
    cloud_is_optical: bool = True
    range_min: float = 0.5
    range_max: float = 12.0
    z_min: float = -2.0
    z_max: float = 2.0

    # Occupancy classification from points in BASE frame (simple)
    occ_z_band: tuple[float, float] = (-0.2, 1.5)  # consider obstacles roughly around drone height band

    # Window outputs (for tmp map + potential)
    window_size_m: float = 20.0


class DA3MappingLayers:
    """
    Core mapping module:
      - accum log-odds grid (global fixed)
      - tmp log-odds grid (resets each frame)
      - potential field computed from occupancy probability window
    """

    def __init__(
        self,
        accum_grid: LogOddsGridCostmap,
        tmp_grid: Optional[LogOddsGridCostmap],
        potential: PotentialFieldLayer,
        cfg: Optional[DA3MappingConfig] = None,
    ):
        self.accum = accum_grid
        self.tmp = tmp_grid
        self.potential = potential
        self.cfg = cfg or DA3MappingConfig()

    def step(self, obs: Observation) -> None:
        if obs.cloud is None:
            return

        pts = np.asarray(obs.cloud.xyz, dtype=np.float32).reshape((-1, 3))
        if pts.shape[0] == 0:
            return

        if self.cfg.cloud_is_optical:
            pts_base = optical_to_base_xyz(pts)
        else:
            pts_base = pts

        # filter in base frame
        x, y, z = pts_base[:, 0], pts_base[:, 1], pts_base[:, 2]
        r = np.sqrt(x * x + y * y)
        m = (
            np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            & (r >= self.cfg.range_min) & (r <= self.cfg.range_max)
            & (z >= self.cfg.z_min) & (z <= self.cfg.z_max)
        )
        pts_base = pts_base[m]
        if pts_base.shape[0] == 0:
            return

        # classify occupied points (simple band; later you can do raycasting/free-space)
        z0, z1 = self.cfg.occ_z_band
        is_occ = (pts_base[:, 2] >= z0) & (pts_base[:, 2] <= z1)

        # transform to global/map
        if obs.pose_map_base is not None:
            pts_map = obs.pose_map_base.transform_points(pts_base)
            cx, cy, cz = obs.pose_map_base.t.tolist()
        else:
            pts_map = pts_base
            cx, cy, cz = 0.0, 0.0, 0.0

        # update grids with hits (x,y)
        self.accum.update_hits(pts_map[:, 0], pts_map[:, 1], is_occ)

        if self.tmp is not None:
            self.tmp.reset()
            self.tmp.update_hits(pts_map[:, 0], pts_map[:, 1], is_occ)

    def get_tmp_window(self, center_xy: tuple[float, float]) -> tuple:
        if self.tmp is None:
            raise ValueError("tmp grid is not enabled")
        spec, p = self.tmp.extract_window(center_xy[0], center_xy[1], self.cfg.window_size_m)
        return spec, p

    def get_accum_window(self, center_xy: tuple[float, float]) -> tuple:
        spec, p = self.accum.extract_window(center_xy[0], center_xy[1], self.cfg.window_size_m)
        return spec, p

    def get_potential_window(self, center_xy: tuple[float, float]) -> tuple:
        spec, p = self.get_accum_window(center_xy)
        U, D = self.potential.compute_from_prob(p, spec.resolution_m)
        return spec, U, D