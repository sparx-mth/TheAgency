"""The ground truth a training sample is scored against: one surveyed building.

A :class:`Scene` bundles everything about a scene that both label generation and
the training loss need, computed once and shared:

* the surveyed 2D occupancy grid at cruise altitude (``robots/PEGASUS/maps``),
* a **signed** ESDF over it, in metres, positive in free space and negative
  inside geometry -- the single number that answers "how safe is this point",
* the traversable and goal-eligible regions (free, clear, landable, and all one
  connected component), which is what makes an unreachable or in-a-wall goal
  structurally impossible rather than something to filter out later,
* a weighted A* planner, whose clearance cost already prefers the middle of a
  corridor to its edge, and
* the repulsive field the medial-axis corrector centres routes on.

UNKNOWN is treated as obstacle everywhere here. An unsurveyed cell is not a cell
to route through, and it is not clearance either -- measuring "distance to the
nearest wall" across unmapped space would reward flying into it.

Pure numpy + the ROS-free core. No torch, no Isaac Sim: a scene loads and plans
on a laptop, which is what keeps the whole label pipeline unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.mapping.costmap.sdf import compute_sdf
from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.interfaces.planner import PlanRequest
from sparx_agency.core.planning.mission import largest_region
from sparx_agency.core.planning.planners.astar.params import WeightedAStarParams
from sparx_agency.core.planning.planners.astar.weighted_planner_2d import (
    WeightedAStarPlanner2D,
)
from sparx_agency.robots.PEGASUS.adapters.scene_map import LANDABLE_LAYER, load_scene_map


@dataclass(frozen=True)
class SceneConfig:
    """How a scene is interpreted for training.

    Attributes:
        scene: Scene key, e.g. ``"office"``. Must have been surveyed with
            ``tasks/planning/sim_flight_recording/survey_scene.py``.
        altitude_m: The altitude the map was surveyed at. A map is only valid at
            its own altitude, so this selects the file.
        goal_clearance_m: Obstacle clearance a *goal* must have. Larger than the
            planner standoff on purpose -- a goal is somewhere the aircraft has
            to hold position, not merely pass through.
        inflate_radius_m: Standoff the expert route is planned at (airframe
            radius plus expected position error).
        inflate_floor_m: Hard lower bound the planner may relax to in a genuinely
            narrow doorway. Never below the airframe's inscribed radius.
        heading_penalty_m: Extra cost, in metres, for a route that starts by
            turning away from where the aircraft is already looking. Unlike
            episode planning (which is free to begin with a turn), a *label* must
            be something the aircraft can start flying now, so this is non-zero.
        waypoint_spacing_m: Spacing of the A* output before densification. Small,
            because a label needs geometric fidelity, not a sparse mission plan.
        clearance_weight: Weight on A*'s soft clearance cost -- what pulls the
            route to the middle of a corridor.
        clearance_margin_m: How far beyond the standoff that soft cost reaches.
        max_expansions: A* node budget. The planner default (200k) is smaller
            than this building has cells, so a 25 m route across it could run
            out and report NO_PATH -- which silently deleted the long-range
            goals from the dataset rather than failing loudly.
        map_dir: Override the map directory (tests). None = the committed maps.
    """

    scene: str = "office"
    altitude_m: float = 1.5
    goal_clearance_m: float = 0.8
    inflate_radius_m: float = 0.6
    inflate_floor_m: float = 0.45
    heading_penalty_m: float = 1.0
    waypoint_spacing_m: float = 0.5
    clearance_weight: float = 3.0
    clearance_margin_m: float = 1.2
    max_expansions: int = 800_000
    map_dir: Optional[str] = None


def bilinear_sample(field: np.ndarray, resolution: float, origin_x: float,
                    origin_y: float, xs, ys) -> np.ndarray:
    """Bilinearly sample an ``(H, W)`` field at world coordinates.

    Uses the cell-*centre* convention shared with
    :meth:`OccupancyGrid2D.grid_to_world` (``x = (gx + 0.5) * res + origin_x``),
    which is also what the torch-side ``sample_sdf`` assumes -- so a value read
    here during label generation and the same value read on the GPU during
    training agree to floating point.

    Queries outside the grid are clamped to the border rather than raising: the
    border of a surveyed building is solid, so the clamped value is the correct
    conservative answer.

    Args:
        field: ``(H, W)`` array indexed ``[gy, gx]``.
        resolution: Metres per cell.
        origin_x: World x of cell ``(0, 0)``.
        origin_y: World y of cell ``(0, 0)``.
        xs: World x coordinates, scalar or array.
        ys: World y coordinates, same shape as ``xs``.

    Returns:
        Sampled values, same shape as ``xs``, float64.
    """
    height, width = field.shape
    col = (np.asarray(xs, dtype=np.float64) - origin_x) / resolution - 0.5
    row = (np.asarray(ys, dtype=np.float64) - origin_y) / resolution - 0.5
    col = np.clip(col, 0.0, width - 1.0)
    row = np.clip(row, 0.0, height - 1.0)

    c0 = np.floor(col).astype(np.int64)
    r0 = np.floor(row).astype(np.int64)
    c1 = np.minimum(c0 + 1, width - 1)
    r1 = np.minimum(r0 + 1, height - 1)
    fc = col - c0
    fr = row - r0

    top = field[r0, c0] * (1.0 - fc) + field[r0, c1] * fc
    bottom = field[r1, c0] * (1.0 - fc) + field[r1, c1] * fc
    return top * (1.0 - fr) + bottom * fr


class Scene:
    """One surveyed building, ready to generate labels against.

    Build with :meth:`load`; the constructor takes already-computed fields so a
    test can assemble a synthetic scene without a map file.
    """

    def __init__(self, config: SceneConfig, grid: OccupancyGrid2D,
                 sdf: np.ndarray, region: np.ndarray, goal_region: np.ndarray,
                 metadata: Dict) -> None:
        self.config = config
        self.name = config.scene
        self.grid = grid
        self.sdf = sdf.astype(np.float32)
        self.region = region
        self.goal_region = goal_region
        self.metadata = metadata
        self.planner = WeightedAStarPlanner2D(WeightedAStarParams(
            inflate_radius_m=config.inflate_radius_m,
            inflate_floor_m=config.inflate_floor_m,
            waypoint_spacing_m=config.waypoint_spacing_m,
            clearance_weight=config.clearance_weight,
            clearance_margin_m=config.clearance_margin_m,
            heading_penalty_m=config.heading_penalty_m,
            max_expansions=config.max_expansions,
            unknown_blocked=True,
        ))
        self._goal_cells: Optional[np.ndarray] = None
        self._goal_world: Optional[np.ndarray] = None
        self._field = None

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, config: SceneConfig) -> "Scene":
        """Read the surveyed map and derive every field training needs.

        Raises:
            FileNotFoundError: If the scene has not been surveyed at that
                altitude; the message quotes the command that produces it.
            ValueError: If nothing in the map has the requested clearance.
        """
        map_dir = Path(config.map_dir) if config.map_dir else None
        grid, metadata, layers = load_scene_map(config.scene, config.altitude_m, map_dir)

        blocked = grid.grid != grid.values.free          # UNKNOWN counts as obstacle
        sdf = compute_sdf(blocked.astype(np.uint8), grid.resolution)

        region = largest_region(grid, config.goal_clearance_m)
        landable = layers.get(LANDABLE_LAYER)
        goal_region = region if landable is None else (region & landable.astype(bool))
        if not goal_region.any():
            raise ValueError(
                f"scene {config.scene!r} has no cell that is both traversable at "
                f"{config.goal_clearance_m:.2f} m and landable -- no goal can be drawn"
            )
        return cls(config, grid, sdf, region, goal_region, metadata)

    # ------------------------------------------------------------ properties
    @property
    def resolution(self) -> float:
        return self.grid.resolution

    @property
    def origin(self) -> Tuple[float, float]:
        """World coordinate of cell ``(0, 0)`` -- the ESDF's origin too."""
        return self.grid.origin_x, self.grid.origin_y

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """``(x_min, y_min, x_max, y_max)`` of the surveyed area, metres."""
        return (self.grid.origin_x, self.grid.origin_y,
                self.grid.origin_x + self.grid.width * self.resolution,
                self.grid.origin_y + self.grid.height * self.resolution)

    @property
    def goal_world(self) -> np.ndarray:
        """``(K, 2)`` world centres of every goal-eligible cell (cached)."""
        if self._goal_world is None:
            cells = np.argwhere(self.goal_region)
            xs = (cells[:, 1] + 0.5) * self.resolution + self.grid.origin_x
            ys = (cells[:, 0] + 0.5) * self.resolution + self.grid.origin_y
            self._goal_cells = cells
            self._goal_world = np.stack([xs, ys], axis=1).astype(np.float64)
        return self._goal_world

    # --------------------------------------------------------------- queries
    def clearance(self, xs, ys) -> np.ndarray:
        """Signed distance to the nearest obstacle, metres (``<0`` inside one)."""
        return bilinear_sample(self.sdf, self.resolution,
                               self.grid.origin_x, self.grid.origin_y, xs, ys)

    def in_goal_region(self, x: float, y: float) -> bool:
        """True if a goal may be placed at this world point."""
        gx, gy = self.grid.world_to_grid(x, y)
        if not self.grid.in_bounds(gx, gy):
            return False
        return bool(self.goal_region[gy, gx])

    # -------------------------------------------------------------- planning
    def plan_route(self, start: Pose2D, goal: Pose2D) -> Optional[np.ndarray]:
        """Weighted-A* route from ``start`` to ``goal``, or None if there is none.

        Args:
            start: Where the aircraft is; its ``yaw`` feeds the heading penalty.
            goal: Where it is being sent.

        Returns:
            ``(N, 2)`` world ``[x, y]`` including the start point, or ``None``.
        """
        result = self.planner.plan(
            PlanRequest(start=start, goal=goal, frame_id=self.grid.frame_id), self.grid)
        if not result.ok or result.path is None:
            return None
        points = [(start.x, start.y)] + [(p.x, p.y) for p in result.path.points]
        route = np.asarray(points, dtype=np.float64)
        return route if route.shape[0] >= 2 else None

    # -------------------------------------------------------------- centring
    def corrector_field(self, sigma_m: float = 0.6):
        """The repulsive field + distance field the medial-axis corrector uses.

        Derived analytically from the signed ESDF this scene already holds,
        rather than from :class:`PotentialFieldLayer`. Two reasons: the
        repulsion is then *exactly* a Gaussian in the true distance
        (``exp(-d^2 / 2 sigma^2)``, 1.0 on and inside geometry, decaying to 0 in
        open space) instead of a truncated blur kernel, and the corrector ends up
        reading the very same field the training loss does -- so a label that
        looks safe here cannot look unsafe on the GPU. It also keeps this whole
        path free of OpenCV.

        Built lazily and cached: one array pass over the building.

        Args:
            sigma_m: Falloff of the repulsion, metres.

        Returns:
            ``(u_rep, d_obs, origin_x, origin_y)`` -- the two ``(H, W)`` fields
            and the origin **already shifted by half a cell**, because
            ``PotentialFieldSampler`` indexes without the half-cell offset that
            ``grid_to_world`` applies.
        """
        if self._field is None:
            d_obs = np.maximum(self.sdf, 0.0).astype(np.float32)
            u_rep = np.exp(-0.5 * (d_obs / float(sigma_m)) ** 2).astype(np.float32)
            half = 0.5 * self.resolution
            self._field = (u_rep, d_obs,
                           self.grid.origin_x + half, self.grid.origin_y + half)
        return self._field


def load_scenes(configs: List[SceneConfig]) -> Dict[str, Scene]:
    """Load several scenes by name, so a run can span more than one building."""
    return {cfg.scene: Scene.load(cfg) for cfg in configs}
