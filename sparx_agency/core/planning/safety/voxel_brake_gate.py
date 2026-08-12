"""Refuse to fly a commanded velocity into occupied voxels.

The last line of defence between a trajectory follower and geometry: given the
planner's own occupancy grid (streamed as occupied/free voxel-centre point
clouds) and the velocity a follower is about to command, decide how much of
that velocity is safe to fly. The follower brakes to the returned scale and
treats a persistent full stop as a physical blockage.

Exists because upstream FALCON's runtime safety check demonstrably does not
protect this deployment: in a 40-minute hospital exploration its pre-publish
collision check fired 45 times without ever signalling the follower, its
executing-trajectory check fired zero times, and the aircraft logged 336
contacts with obstacles that WERE present in the voxel map at the time. The
map knew; nothing braking-side asked it. This class is the asking.

Design constraints:

* Pure numpy (1.17 API) and Python 3.8 -- imported inside the Noetic FALCON
  container.
* The map arrives as ADD/REMOVE streams: FALCON publishes both full-map
  sweeps and high-resolution local-update boxes on the same topics, so the
  only message-shape-independent client is an accumulator -- add every voxel
  reported occupied, remove every voxel reported free. Voxels wiped to
  UNKNOWN by a planner respawn appear in neither stream and therefore linger
  as occupied: deliberately conservative, cleared the moment they are
  re-observed free.
"""
from __future__ import annotations

import math

import numpy as np


class VoxelBrakeGateConfig(object):
    """Tunables for :class:`VoxelBrakeGate`.

    Attributes:
        voxel_m: Map resolution; keys are quantised to this grid.
        drone_radius_m: Airframe radius the corridor must clear laterally.
        z_band: ``(lo, hi)`` band of voxel heights that can strike the
            airframe at flight altitude; voxels outside are ignored.
        brake_decel: Deceleration the airframe can actually deliver, m/s^2.
        react_s: Command-to-plant latency budgeted before braking starts.
        margin_m: Extra clearance beyond the stopping point.
        hard_stop_m: Anything with a nose gap under this is a full stop no
            matter how slow the command is.

    The defaults are deliberately calibrated to the PLANNER's clearance
    model, not to comfort: FALCON with obstacles_inflation 0.25 legally
    routes the aircraft's centre ~0.35 m from occupied voxel centres, so a
    gate that vetoes anything nearer ~0.75 m centre distance livelocks every
    doorway the planner is entitled to fly (measured in run 009: brake ->
    retreat -> identical replan cycles until the mission cap). The gate's
    job is the last half-voxel the planner gets wrong, not a second, more
    timid planner.
    """

    def __init__(self,
                 voxel_m=0.1,
                 drone_radius_m=0.25,
                 z_band=(0.4, 2.0),
                 z_layer_m=0.2,
                 body_halfheight_m=0.35,
                 brake_decel=0.8,
                 react_s=0.30,
                 margin_m=0.08,
                 hard_stop_m=0.10):
        # type: (float, float, tuple, float, float, float, float, float, float) -> None
        self.voxel_m = float(voxel_m)
        self.drone_radius_m = float(drone_radius_m)
        self.z_band = (float(z_band[0]), float(z_band[1]))
        self.z_layer_m = float(z_layer_m)
        self.body_halfheight_m = float(body_halfheight_m)
        self.brake_decel = float(brake_decel)
        self.react_s = float(react_s)
        self.margin_m = float(margin_m)
        self.hard_stop_m = float(hard_stop_m)


class VoxelBrakeGate(object):
    """Accumulate the planner's voxel stream; veto velocities that hit it."""

    def __init__(self, config=None):
        # type: (object) -> None
        self._cfg = config or VoxelBrakeGateConfig()
        self._occupied = set()   # packed int keys

    # ── map ingest ───────────────────────────────────────────────────────

    def _keys(self, points_xyz):
        # type: (np.ndarray) -> set
        """Quantise Nx3 world points in the strike band to packed voxel keys.

        Keys carry a quantised z LAYER, not a collapsed band: a 2D gate reads
        legally flying OVER a desk as blocked (the desk voxels share the x, y
        column) and livelocks against the planner, which is entitled to route
        above furniture. Measured in run 011: brake-retreat cycles every ~14 s
        against over-desk routes the planner kept correctly re-issuing.
        """
        if points_xyz.size == 0:
            return set()
        lo, hi = self._cfg.z_band
        band = points_xyz[(points_xyz[:, 2] >= lo) & (points_xyz[:, 2] <= hi)]
        if band.size == 0:
            return set()
        idx = np.floor(band[:, :2] / self._cfg.voxel_m).astype(np.int64)
        iz = np.clip(np.floor(band[:, 2] / self._cfg.z_layer_m), 0, 31).astype(np.int64)
        packed = (((idx[:, 0] + (1 << 20)) * (1 << 22)
                   + (idx[:, 1] + (1 << 20))) * 32 + iz)
        return set(packed.tolist())

    def update_occupied(self, points_xyz):
        # type: (np.ndarray) -> None
        """Add voxels reported OCCUPIED (full sweep or local update alike)."""
        self._occupied.update(self._keys(points_xyz))

    def replace_occupied(self, points_xyz):
        # type: (np.ndarray) -> None
        """Adopt a FULL occupied sweep as the entire truth.

        The right ingest when the publisher emits complete sweeps (FALCON's
        0.1 m map does -- its local-box variant only runs at coarser
        resolutions): accumulation can only ever ADD ghosts on top of this,
        and a run-013 post-mortem showed exactly that failure -- freed
        voxels lost in transit left a phantom ring around the spawn that
        vetoed every exit for 45 minutes. Replacement makes the gate track
        the planner's map faithfully with staleness bounded by one sweep.
        """
        self._occupied = self._keys(points_xyz)

    def update_free(self, points_xyz):
        # type: (np.ndarray) -> None
        """Remove voxels reported FREE."""
        self._occupied.difference_update(self._keys(points_xyz))

    def occupied_count(self):
        # type: () -> int
        """Number of accumulated occupied voxels (diagnostics)."""
        return len(self._occupied)

    # ── queries ──────────────────────────────────────────────────────────

    def _blocked_at(self, x, y, layers):
        # type: (float, float, tuple) -> bool
        v = self._cfg.voxel_m
        col = ((int(math.floor(x / v)) + (1 << 20)) * (1 << 22)
               + (int(math.floor(y / v)) + (1 << 20))) * 32
        for iz in layers:
            if (col + iz) in self._occupied:
                return True
        return False

    def _layers_for_z(self, z):
        # type: (float) -> tuple
        """The z layers the airframe can strike at altitude ``z``."""
        h = self._cfg.body_halfheight_m
        lm = self._cfg.z_layer_m
        lo = max(0, int(math.floor((z - h) / lm)))
        hi = min(31, int(math.floor((z + h) / lm)))
        return tuple(range(lo, hi + 1))

    def nearest_occupied(self, pos_xyz, max_r):
        # type: (tuple, float) -> object
        """Distance to the nearest occupied voxel centre within ``max_r``.

        Feeds the proximity speed governor: closing speed must be bounded by
        room at EVERY bearing, because the directional brakes each have a
        blind arc (nose-only depth, commanded-direction corridor) and a
        cruise-speed strike proved a race between them is winnable. Returns
        None when nothing occupied is within ``max_r``.
        """
        layers = self._layers_for_z(float(pos_xyz[2]) if len(pos_xyz) > 2 else 1.2)
        v = self._cfg.voxel_m
        n = int(math.ceil(max_r / v))
        cx, cy = float(pos_xyz[0]), float(pos_xyz[1])
        best = None
        for ix in range(-n, n + 1):
            for iy in range(-n, n + 1):
                d = math.hypot(ix * v, iy * v)
                if d > max_r or (best is not None and d >= best):
                    continue
                if self._blocked_at(cx + ix * v, cy + iy * v, layers):
                    best = d
        return best

    def bubble_blocked(self, pos_xyz, clearance_m):
        # type: (tuple, float) -> bool
        """Whether any occupied voxel sits within ``clearance_m`` at any bearing.

        The corridor test only guards the direction of MOTION: an aircraft
        sliding parallel to a wall shows a clear corridor while lateral drift
        closes the last centimetres unchecked (measured: a 2000-contact grind
        along a pile face with every forward check passing). This is the
        personal-space check that catches closure from ANY side.
        """
        layers = self._layers_for_z(float(pos_xyz[2]) if len(pos_xyz) > 2 else 1.2)
        v = self._cfg.voxel_m
        n = int(math.ceil(clearance_m / v))
        cx, cy = float(pos_xyz[0]), float(pos_xyz[1])
        for ix in range(-n, n + 1):
            for iy in range(-n, n + 1):
                x = cx + ix * v
                y = cy + iy * v
                if math.hypot(x - cx, y - cy) <= clearance_m + v * 0.5:
                    if self._blocked_at(x, y, layers):
                        return True
        return False

    def blocked_distance(self, pos_xyz, dir_xy, max_dist):
        # type: (tuple, tuple, float) -> object
        """Distance to the first blocked sample along a swept corridor.

        Samples every half voxel along ``dir_xy`` (unit not required); at each
        step checks the centre plus lateral offsets covering the airframe
        radius, in the z layers the airframe occupies at ``pos_xyz[2]``.
        Returns the distance, or None when the corridor is clear.
        """
        norm = math.hypot(dir_xy[0], dir_xy[1])
        if norm < 1e-6:
            return None
        layers = self._layers_for_z(float(pos_xyz[2]) if len(pos_xyz) > 2 else 1.2)
        ux, uy = dir_xy[0] / norm, dir_xy[1] / norm
        px, py = -uy, ux                      # lateral unit
        step = self._cfg.voxel_m * 0.5
        r = self._cfg.drone_radius_m
        # Lateral rays no further apart than one voxel: rays run PARALLEL to
        # travel, so a one-voxel-thin column lying between two rays would be
        # missed at every along-track step, not just one.
        n = max(2, int(math.ceil(2.0 * r / (self._cfg.voxel_m * 0.99))))
        laterals = tuple(-r + i * (2.0 * r / n) for i in range(n + 1))
        d = step
        while d <= max_dist:
            cx, cy = pos_xyz[0] + ux * d, pos_xyz[1] + uy * d
            for off in laterals:
                if self._blocked_at(cx + px * off, cy + py * off, layers):
                    return d
            d += step
        return None

    def command_scale(self, pos_xyz, vel_xy):
        # type: (tuple, tuple) -> tuple
        """How much of ``vel_xy`` is safe to fly right now.

        Returns ``(scale, blocked_dist)``: scale 1.0 when the stopping
        corridor is clear (blocked_dist None or beyond it), 0.0 for a hard
        stop, else the largest factor whose stopping distance still fits.
        """
        cfg = self._cfg
        speed = math.hypot(vel_xy[0], vel_xy[1])
        if speed < 1e-3:
            return 1.0, None
        stop = speed * cfg.react_s + speed * speed / (2.0 * cfg.brake_decel)
        # blocked_distance measures centre-to-voxel; the NOSE arrives a full
        # airframe radius earlier, so the radius joins the horizon and is
        # subtracted from every gap below.
        horizon = stop + cfg.margin_m + cfg.drone_radius_m + cfg.voxel_m
        blocked = self.blocked_distance(pos_xyz, vel_xy, horizon)
        if blocked is None:
            return 1.0, None
        gap = blocked - cfg.drone_radius_m
        if gap <= cfg.hard_stop_m:
            return 0.0, blocked
        # largest speed whose react+brake distance fits inside the gap
        avail = max(0.0, gap - cfg.margin_m)
        # solve v*t + v^2/2a = avail for v
        a, t = cfg.brake_decel, cfg.react_s
        v_safe = a * (-t + math.sqrt(t * t + 2.0 * avail / a))
        return max(0.0, min(1.0, v_safe / speed)), blocked
