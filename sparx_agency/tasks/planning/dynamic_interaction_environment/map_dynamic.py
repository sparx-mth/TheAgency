"""
Obstacle map with static + dynamic obstacles.

Environment-only responsibilities:
- Geometry checks: occupied(x,y,margin)
- Dynamic objects: update(dt), spawn, remove
- Conversion to costmap for initial planning (optional)

Dynamic obstacle enhancement:
- A dynamic circle can optionally patrol along a segment A<->B.
  When reaching B it reverses toward A, and vice versa.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np

from sparx_agency.core.planning.environment import Costmap2D, CostmapParams
from sparx_agency.core.mapping.costmap.inflation import inflate_occupancy, InflationParams
from sparx_agency.core.mapping.costmap.distance_field import compute_clearance_field, DistanceFieldParams


@dataclass
class DynamicCircle:
    cx: float
    cy: float
    r: float
    vx: float
    vy: float
    id: int = 0

    # Optional "patrol along segment" mode.
    # If patrol_a and patrol_b are set and they are far enough -> object moves along that segment.
    patrol_a: Optional[Tuple[float, float]] = None
    patrol_b: Optional[Tuple[float, float]] = None
    patrol_dir: int = 1  # +1 moves toward B, -1 toward A

    # Speed along the segment (m/s). Used only when patrol is active.
    patrol_speed: float = 0.6


@dataclass
class ObstacleMapDynamic:
    width: float = 10.0
    height: float = 10.0
    origin_x: float = 0.0
    origin_y: float = 0.0
    resolution: float = 0.02

    rectangles: List[Tuple[float, float, float, float]] = field(default_factory=list)
    circles: List[Tuple[float, float, float]] = field(default_factory=list)

    dynamic_circles: List[DynamicCircle] = field(default_factory=list)
    _next_dyn_id: int = 1

    def add_rectangle(self, x: float, y: float, w: float, h: float) -> None:
        self.rectangles.append((x, y, w, h))

    def add_circle(self, cx: float, cy: float, r: float) -> None:
        self.circles.append((cx, cy, r))

    def add_dynamic_circle(
        self,
        cx: float,
        cy: float,
        r: float,
        vx: float,
        vy: float,
        patrol_a: Optional[Tuple[float, float]] = None,
        patrol_b: Optional[Tuple[float, float]] = None,
        patrol_speed: float = 0.6,
    ) -> Optional[int]:
        if len(self.dynamic_circles) >= 10_000:
            return None

        obj = DynamicCircle(
            cx=float(cx),
            cy=float(cy),
            r=float(r),
            vx=float(vx),
            vy=float(vy),
            id=self._next_dyn_id,
            patrol_a=patrol_a,
            patrol_b=patrol_b,
            patrol_speed=float(patrol_speed),
            patrol_dir=1,
        )
        self._next_dyn_id += 1
        self.dynamic_circles.append(obj)
        return obj.id

    def remove_dynamic_by_id(self, obj_id: int) -> bool:
        for i, o in enumerate(self.dynamic_circles):
            if o.id == obj_id:
                self.dynamic_circles.pop(i)
                return True
        return False

    def clear_dynamic(self) -> None:
        self.dynamic_circles.clear()

    def update_dynamic(self, dt: float, bounce_on_walls: bool = True) -> None:
        x_min = self.origin_x
        x_max = self.origin_x + self.width
        y_min = self.origin_y
        y_max = self.origin_y + self.height

        for o in self.dynamic_circles:
            if self._is_patrol_active(o):
                self._update_patrol(o, dt)
            else:
                o.cx += o.vx * dt
                o.cy += o.vy * dt

            if not bounce_on_walls:
                continue

            # Bounce with radius margins (works for both modes).
            if o.cx - o.r < x_min:
                o.cx = x_min + o.r
                o.vx *= -1.0
                o.patrol_dir *= -1
            if o.cx + o.r > x_max:
                o.cx = x_max - o.r
                o.vx *= -1.0
                o.patrol_dir *= -1
            if o.cy - o.r < y_min:
                o.cy = y_min + o.r
                o.vy *= -1.0
                o.patrol_dir *= -1
            if o.cy + o.r > y_max:
                o.cy = y_max - o.r
                o.vy *= -1.0
                o.patrol_dir *= -1

    def _is_patrol_active(self, o: DynamicCircle) -> bool:
        if o.patrol_a is None or o.patrol_b is None:
            return False
        ax, ay = o.patrol_a
        bx, by = o.patrol_b
        return (ax - bx) ** 2 + (ay - by) ** 2 >= 1e-6

    def _update_patrol(self, o: DynamicCircle, dt: float) -> None:
        assert o.patrol_a is not None and o.patrol_b is not None
        ax, ay = o.patrol_a
        bx, by = o.patrol_b

        # Decide target based on current direction
        tx, ty = (bx, by) if o.patrol_dir >= 0 else (ax, ay)

        dx = tx - o.cx
        dy = ty - o.cy
        dist = math.hypot(dx, dy)

        if dist < 1e-6:
            # Snap and flip direction
            o.cx, o.cy = tx, ty
            o.patrol_dir *= -1
            return

        step = float(o.patrol_speed) * float(dt)
        if step >= dist:
            # Reached target this step -> snap and reverse
            o.cx, o.cy = tx, ty
            o.patrol_dir *= -1
            # Velocity is informational (for debug/visual); keep consistent with direction.
            # Compute new target velocity on next update.
            o.vx = 0.0
            o.vy = 0.0
            return

        ux = dx / dist
        uy = dy / dist
        o.cx += ux * step
        o.cy += uy * step
        o.vx = ux * o.patrol_speed
        o.vy = uy * o.patrol_speed

    def is_occupied(self, x: float, y: float, margin: float = 0.0) -> bool:
        # Static rectangles
        for rx, ry, rw, rh in self.rectangles:
            closest_x = max(rx, min(x, rx + rw))
            closest_y = max(ry, min(y, ry + rh))
            if math.hypot(x - closest_x, y - closest_y) <= margin:
                return True

        # Static circles
        for cx, cy, r in self.circles:
            if math.hypot(x - cx, y - cy) <= r + margin:
                return True

        # Dynamic circles
        for o in self.dynamic_circles:
            if math.hypot(x - o.cx, y - o.cy) <= o.r + margin:
                return True

        return False

    def to_costmap(self, inflate_radius: float = 0.1, include_dynamic: bool = True) -> Costmap2D:
        nx = int(self.width / self.resolution)
        ny = int(self.height / self.resolution)
        occupancy = np.zeros((ny, nx), dtype=np.uint8)

        for iy in range(ny):
            for ix in range(nx):
                x = self.origin_x + ix * self.resolution
                y = self.origin_y + iy * self.resolution
                if self._occupied_for_costmap(x, y, include_dynamic=include_dynamic):
                    occupancy[iy, ix] = 255

        if inflate_radius > 0.0:
            occupancy = inflate_occupancy(
                occupancy,
                resolution=self.resolution,
                params=InflationParams(radius_m=inflate_radius),
            )

        clearance = compute_clearance_field(
            occupancy,
            resolution=self.resolution,
            params=DistanceFieldParams(),
        )
        params = CostmapParams(
            resolution=self.resolution,
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            frame_id="map",
        )
        return Costmap2D(occupancy, params, clearance=clearance)

    def _occupied_for_costmap(self, x: float, y: float, include_dynamic: bool) -> bool:
        # Current implementation includes dynamic only if is_occupied checks them,
        # but we keep include_dynamic=False for planning by calling to_costmap(... include_dynamic=False).
        if not include_dynamic:
            # Recheck static-only quickly:
            for rx, ry, rw, rh in self.rectangles:
                if rx <= x <= rx + rw and ry <= y <= ry + rh:
                    return True
            for cx, cy, r in self.circles:
                if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                    return True
            return False

        return self.is_occupied(x, y, margin=0.0)
