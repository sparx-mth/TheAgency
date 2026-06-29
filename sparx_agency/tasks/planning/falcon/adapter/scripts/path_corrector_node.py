#!/usr/bin/env python3
"""path_corrector_node.py -- ROS1 adapter: correct a planned path against the BEV.

This node owns ONE responsibility: take a planned ``nav_msgs/Path`` from *any*
planner, reshape it to be safer/centred against the live 2D BEV, and republish
the corrected path on the topic the follower flies. It is deliberately blind to
which planner produced the path and to which correction *strategy* is used:

  - WHICH PLANNER: it subscribes to ``~input_path_topic``. Point that at A*
    (``/path/waypoints_astar``), NavDP (``/path/waypoints_navdp``), RRT*, ... --
    swapping the planner needs no change here.
  - WHICH STRATEGY: it builds a core :class:`PathCorrector` by ``~corrector``
    name (default ``potential_field``). Swapping the repulsive field for an ESDF
    follower is a new core strategy + one factory branch -- again no change here.

All of the correction maths -- building the repulsive field ``U_rep`` from the
BEV, recentring each waypoint toward the corridor centre, damping the push into
unknown space and clipping any corrected waypoint back to stay clear of inflated
obstacles -- lives in ROS-free, unit-tested core
(``sparx_agency.core.planning.safety.path_correction``). This node owns only ROS
concerns: decoding the BEV + input path, invoking the corrector, publishing the
corrected path (and, for the BEV viewer, the raw input echo + the repulsive-force
arrows), and logging.

Freeze semantics: a correction is computed once per INPUT PATH message and the
result is then latched. A BEV update alone does NOT re-correct, so the published
trajectory stays frozen between planner ticks (the planner's strict cadence is
what drives a new path; this node just follows it).

Passthrough: with ``~enabled:=false`` (the old ``safety_correct:=false``) the
input path is republished verbatim on ``~path_topic`` -- so the topology is the
same whether or not correction is on, and the follower always reads one topic.

  in   ~input_path_topic (Path)  /path/waypoints_astar  (the planner's raw path)
  in   ~bev_topic (OccupancyGrid) /falcon/bev_2d
  out  ~path_topic (Path, latched)      /path/waypoints_safe (corrected, SAFE -> simplifier)
  out  ~raw_path_topic (Path, latched)  /path/waypoints_raw  (input echo, viz)
  out  ~forces_topic (MarkerArray)      /path/forces         (F_rep arrows, viz)

See the file footer for the full rosparam list.
"""
import math

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray

from sparx_agency.core.common.types import Path2D, Pose2D
from sparx_agency.core.planning.environment import (
    OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues)
from sparx_agency.core.planning.planners.common.utils_2d import decimate_min_spacing_2d
from sparx_agency.core.planning.safety.path_correction import (
    EsdfCorrectorConfig, PotentialFieldCorrectorConfig, PotentialFieldPathCorrector,
    make_path_corrector)

# nav_msgs/OccupancyGrid int8 convention published by bev_publisher_node.
BEV_VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)


class PathCorrectorNode:
    def __init__(self):
        rospy.init_node("path_corrector")
        G = rospy.get_param

        self.input_path_topic = G("~input_path_topic", "/path/waypoints_astar")
        self.bev_topic = G("~bev_topic", "/falcon/bev_2d")
        # SAFE (corrected) path -> the trajectory_simplifier cleans this into the
        # flown /path/waypoints. Defaults to /path/waypoints_safe so a standalone
        # rosrun does not double-publish /path/waypoints against the simplifier.
        self.path_topic = G("~path_topic", "/path/waypoints_safe")
        self.raw_path_topic = G("~raw_path_topic", "/path/waypoints_raw")
        self.frame_id = G("~frame_id", "world")

        # ~enabled is the old ~safety_correct: when false this node is a pure
        # passthrough (input -> /path/waypoints verbatim), so the follower always
        # reads one topic regardless of whether correction runs.
        self.enabled = bool(G("~enabled", True))
        self.corrector_name = G("~corrector", "potential_field")
        self._apf_debug = bool(G("~apf_debug", True))
        # Thin a dense input path to >= this spacing (m) BEFORE correcting, so a
        # lateral corrector does not push near-coincident neighbours opposite ways
        # (the NavDP zig-zag). 0 = off. A*'s metre-scale waypoints are unaffected.
        self.resample_min_spacing_m = float(G("~resample_min_spacing_m", 0.0))

        # Build the chosen correction strategy from rosparams. Defaults match the
        # historical astar_planner node, so wiring the same ~apf_* params through is
        # behaviour-preserving. inflate_radius_m feeds the corrected-path collision
        # clip and should match the planner's inflation (the launch wires the same
        # arg to both). Raises loudly on an unknown ~corrector name (no fallback).
        self.corrector = None
        if self.enabled:
            cfg = self._build_config(G, self.corrector_name)
            self.corrector = make_path_corrector(self.corrector_name, cfg)

        # Repulsive-force arrows (F_rep = -grad U_rep) for RViz / the BEV viewer.
        # Only meaningful for the potential-field strategy, which exposes the field.
        self._is_pf = isinstance(self.corrector, PotentialFieldPathCorrector)
        self._publish_force_markers = bool(G("~publish_forces", True)) and self._is_pf
        self.forces_topic = G("~forces_topic", "/path/forces")
        self.force_arrow_scale = float(G("~force_arrow_scale", 1.0))
        self.force_arrow_z = float(G("~force_arrow_z", 0.0))
        self._publish_force_field = bool(G("~publish_force_field", True))
        self.force_field_stride = max(1, int(G("~force_field_stride", 4)))

        self.grid = None          # OccupancyGrid2D (latest BEV)
        self.n_corrected = 0

        self.pub_path = rospy.Publisher(self.path_topic, Path, queue_size=1, latch=True)
        self.pub_raw = rospy.Publisher(self.raw_path_topic, Path, queue_size=1, latch=True)
        self._pub_forces = (rospy.Publisher(self.forces_topic, MarkerArray,
                                            queue_size=1, latch=True)
                            if self._publish_force_markers else None)

        rospy.Subscriber(self.bev_topic, OccupancyGrid, self._bev_cb, queue_size=1)
        rospy.Subscriber(self.input_path_topic, Path, self._path_cb, queue_size=1)
        self._banner()

    # ─── Strategy config ─────────────────────────────────────────
    @staticmethod
    def _build_config(G, name):
        """Build the strategy-specific config from rosparams.

        Strategy-specific tuning uses a per-strategy prefix (~apf_* / ~esdf_*); the
        map-aware SAFETY knobs (~apf_collision_recheck / ~apf_treat_unknown_as_free /
        ~apf_unknown_damping / ~apf_unknown_radius_m / ~apf_pin_last and the shared
        ~inflate_radius_m) apply to whichever corrector is active. An unknown name
        falls through with a potential_field config; make_path_corrector then raises.
        """
        if name == "esdf":
            return EsdfCorrectorConfig(
                occ_thresh=float(G("~esdf_occ_thresh", 0.65)),
                smooth_sigma_m=float(G("~esdf_smooth_sigma_m", 0.1)),
                # 0 = centre on the ESDF ridge (corridor/doorway middle); >0 = stop
                # once this far from walls (a safety floor, like FALCON safe_distance).
                target_clearance_m=float(G("~esdf_target_clearance_m", 0.0)),
                max_step_m=float(G("~esdf_max_step_m", 0.2)),
                max_total_shift_m=float(G("~esdf_max_total_shift_m", 1.0)),
                iterations=int(G("~esdf_iterations", 12)),
                lateral_only=bool(G("~esdf_lateral_only", True)),
                pin_last=bool(G("~apf_pin_last", True)),
                collision_recheck=bool(G("~apf_collision_recheck", True)),
                inflate_radius_m=float(G("~inflate_radius_m", 0.4)),
                treat_unknown_as_free=bool(G("~apf_treat_unknown_as_free", True)),
                unknown_damping=bool(G("~apf_unknown_damping", True)),
                unknown_radius_m=float(G("~apf_unknown_radius_m", 0.75)),
            )
        return PotentialFieldCorrectorConfig(
            occ_thresh=float(G("~apf_occ_thresh", 0.65)),
            sigma_m=float(G("~apf_sigma_m", 0.6)),
            centering=str(G("~apf_centering", "line_search")),
            center_step_m=float(G("~apf_center_step_m", 0.05)),
            corner_swing=float(G("~apf_corner_swing", 1.0)),
            iterations=int(G("~apf_iterations", 5)),
            gain=float(G("~apf_gain", 1.0)),
            max_step_m=float(G("~apf_max_step_m", 0.4)),
            max_total_shift_m=float(G("~apf_max_total_shift_m", 2.0)),
            smoothing_passes=int(G("~apf_smoothing_passes", 2)),
            min_clearance_m=float(G("~apf_min_clearance_m", 0.0)),
            lateral_only=bool(G("~apf_lateral_only", True)),
            pin_last=bool(G("~apf_pin_last", True)),
            collision_recheck=bool(G("~apf_collision_recheck", True)),
            inflate_radius_m=float(G("~inflate_radius_m", 0.4)),
            treat_unknown_as_free=bool(G("~apf_treat_unknown_as_free", True)),
            unknown_damping=bool(G("~apf_unknown_damping", True)),
            unknown_radius_m=float(G("~apf_unknown_radius_m", 0.75)),
        )

    # ─── Callbacks ───────────────────────────────────────────────
    def _bev_cb(self, msg):
        """Store the latest BEV. A map update does NOT re-correct -- correction is
        driven strictly by a new input path, so the published trajectory stays
        frozen between planner ticks (mirrors the planner's own freeze)."""
        first = self.grid is None
        self.grid = self._decode(msg)
        if first:
            i = msg.info
            rospy.loginfo("path_corrector: first BEV W=%d H=%d res=%.2f origin=(%.1f,%.1f)",
                          i.width, i.height, i.resolution,
                          i.origin.position.x, i.origin.position.y)

    def _path_cb(self, msg):
        """Correct one planned path and publish it (+ raw echo + force arrows)."""
        pts = self._decode_path(msg)

        if not self.enabled or len(pts) < 2:
            self.pub_raw.publish(self._path_msg(pts))     # input echo (viewer red)
            self._publish(pts, corrected=False)           # passthrough
            return
        if self.grid is None:
            self.pub_raw.publish(self._path_msg(pts))
            rospy.logwarn_throttle(
                5.0, "path_corrector: no BEV yet -- passing the input path through "
                "on %s uncorrected", self.path_topic)
            self._publish(pts, corrected=False)
            return

        # Thin a dense input (NavDP) to >= resample_min_spacing_m BEFORE correcting,
        # so the lateral corrector cannot push near-coincident neighbours opposite
        # ways (zig-zag). A*'s metre-scale waypoints come back unchanged. The echo
        # and force arrows use this conditioned path -- the actual correction input.
        cond = (decimate_min_spacing_2d(pts, self.resample_min_spacing_m)
                if self.resample_min_spacing_m > 0.0 else pts)
        self.pub_raw.publish(self._path_msg(cond))        # input echo (viewer red)

        frame = msg.header.frame_id or self.frame_id
        path = Path2D(points=tuple(cond), frame_id=frame)
        try:
            result = self.corrector.correct(path, self.grid)
            out = result.path.points
            rospy.loginfo_throttle(
                5.0, "path_corrector: %s recentred %d/%d waypoint(s) (input %d -> %d)",
                self.corrector_name, result.num_moved, result.num_points,
                len(pts), len(cond))
            if self._apf_debug:
                self._log_debug(cond, out)
        except Exception as exc:                          # noqa: BLE001 -- stay flying
            rospy.logwarn_throttle(
                5.0, "path_corrector: correction failed (%s) -- publishing the "
                "(uncorrected) input path on %s", exc, self.path_topic)
            self._publish(cond, corrected=False)
            return

        self._publish(out, corrected=True)
        if self._pub_forces is not None and self.corrector.field is not None:
            self._publish_forces(cond)                    # sampled at the input waypoints

    # ─── Decode ──────────────────────────────────────────────────
    def _decode(self, msg):
        """nav_msgs/OccupancyGrid -> core OccupancyGrid2D (BEV value convention)."""
        i = msg.info
        try:
            data = np.frombuffer(bytes(bytearray(msg.data)), dtype=np.int8)
        except Exception:
            data = np.asarray(msg.data, dtype=np.int8)
        data = data.reshape(i.height, i.width).astype(np.int16)
        params = OccupancyGrid2DParams(
            resolution=i.resolution,
            origin_x=i.origin.position.x,
            origin_y=i.origin.position.y,
            frame_id=self.frame_id,
        )
        return OccupancyGrid2D(data, params, values=BEV_VALUES)

    @staticmethod
    def _decode_path(msg):
        """nav_msgs/Path -> tuple[Pose2D] (x, y only; the follower derives heading)."""
        return tuple(Pose2D(float(p.pose.position.x), float(p.pose.position.y))
                     for p in msg.poses)

    # ─── Publish ─────────────────────────────────────────────────
    def _path_msg(self, points):
        """Build a nav_msgs/Path (identity orientation) from Pose2D points."""
        m = Path()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = self.frame_id
        for pt in points:
            ps = PoseStamped()
            ps.header = m.header
            ps.pose.position.x = pt.x
            ps.pose.position.y = pt.y
            ps.pose.orientation.w = 1.0
            m.poses.append(ps)
        return m

    def _publish(self, points, corrected):
        """Publish the (corrected or passed-through) path on ~path_topic."""
        self.pub_path.publish(self._path_msg(points))
        if corrected:
            self.n_corrected += 1
        if len(points) >= 2:
            length = sum(a.distance_to(b) for a, b in zip(points[:-1], points[1:]))
            rospy.loginfo("path_corrector: PATH PUBLISHED %d wp %.2fm %s "
                          "first=(%.2f,%.2f) last=(%.2f,%.2f)", len(points), length,
                          "(corrected)" if corrected else "(passthrough)",
                          points[0].x, points[0].y, points[-1].x, points[-1].y)

    def _log_debug(self, input_points, out_points):
        """Per-cycle proof the push reaches the waypoints (~apf_debug).

        Logs the most-shifted waypoint's input [x,y], applied push [dx,dy] and new
        [x,y], plus a summary. Disable with ~apf_debug:=false once verified.
        """
        n = min(len(input_points), len(out_points))
        if n == 0:
            return
        dxy = [(out_points[i].x - input_points[i].x,
                out_points[i].y - input_points[i].y) for i in range(n)]
        mags = [math.hypot(dx, dy) for dx, dy in dxy]
        n_moved = sum(1 for mg in mags if mg > 1e-4)
        i = max(range(n), key=lambda k: mags[k])
        rospy.loginfo("path_corrector DEBUG: %d/%d wp moved, mean|push|=%.3fm, "
                      "max|push|=%.3fm @wp%d", n_moved, n, sum(mags) / n, mags[i], i)
        rospy.loginfo("path_corrector DEBUG wp%d: in=[%.3f, %.3f] push=[%.3f, %.3f] "
                      "new=[%.3f, %.3f]", i, input_points[i].x, input_points[i].y,
                      dxy[i][0], dxy[i][1], out_points[i].x, out_points[i].y)

    # ─── Force-arrow visualization (potential-field strategy only) ──
    def _publish_forces(self, points):
        """Publish F_rep = -grad U_rep at each waypoint as bright-yellow ARROW
        markers, plus (optionally) a coarse F_rep field over the free cells.

        ``field.descent(x, y)`` is the gradient of the Gaussian-summed U_rep -- the
        repulsive force from ALL nearby walls -- so each arrow points away from the
        walls toward open space, direct proof the field pushes the path off the
        walls. Sampled at the raw waypoints where the path hugs the walls.
        """
        field = self.corrector.field
        arr = MarkerArray()
        wipe = Marker()                            # clear last plan's arrows first
        wipe.header.frame_id = self.frame_id
        wipe.ns = "apf_forces"
        wipe.action = Marker.DELETEALL
        arr.markers.append(wipe)
        stamp = rospy.Time.now()
        for idx, p in enumerate(points):
            f = field.descent(p.x, p.y)            # -grad U_rep = F_rep, as [x, y]
            if f is None:
                continue
            fx, fy = float(f[0]), float(f[1])
            if math.hypot(fx, fy) < 1e-6:
                continue                           # ~no force here -> skip zero arrow
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = stamp
            m.ns = "apf_forces"
            m.id = idx + 1
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.points = [
                Point(p.x, p.y, self.force_arrow_z),
                Point(p.x + fx * self.force_arrow_scale,
                      p.y + fy * self.force_arrow_scale, self.force_arrow_z),
            ]
            m.scale.x = 0.05                       # shaft diameter
            m.scale.y = 0.12                       # arrowhead diameter
            m.scale.z = 0.0
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 0.0, 0.95  # bright yellow
            m.pose.orientation.w = 1.0             # identity (ARROW uses points[])
            arr.markers.append(m)
        if self._publish_force_field:
            self._append_force_field(arr, stamp)
        self._pub_forces.publish(arr)

    def _append_force_field(self, arr, stamp):
        """Append a coarse grid of F_rep arrows over the free cells (ns 'apf_field').

        Sampling -grad U_rep across the free space -- not only at the waypoints --
        makes it visually clear how hard EACH wall section pushes: dense and strong
        next to walls, fading to nothing in open space. Drawn dimmer/thinner than
        the per-waypoint arrows. ``force_field_stride`` controls density.
        """
        field = self.corrector.field
        if field is None or self.grid is None:
            return
        g = self.grid
        cells = g.grid
        wipe = Marker()                            # clear last plan's field arrows
        wipe.header.frame_id = self.frame_id
        wipe.ns = "apf_field"
        wipe.action = Marker.DELETEALL
        arr.markers.append(wipe)
        stride = self.force_field_stride
        mid = 0
        for gy in range(0, g.height, stride):
            for gx in range(0, g.width, stride):
                if int(cells[gy, gx]) != g.values.free:
                    continue                       # only sample observed FREE cells
                x, y = g.grid_to_world(gx, gy)
                f = field.descent(x, y)
                if f is None:
                    continue
                fx, fy = float(f[0]), float(f[1])
                if math.hypot(fx, fy) < 1e-2:
                    continue                       # negligible push -> skip (open space)
                m = Marker()
                m.header.frame_id = self.frame_id
                m.header.stamp = stamp
                m.ns = "apf_field"
                m.id = mid
                mid += 1
                m.type = Marker.ARROW
                m.action = Marker.ADD
                m.points = [
                    Point(x, y, self.force_arrow_z),
                    Point(x + fx * self.force_arrow_scale,
                          y + fy * self.force_arrow_scale, self.force_arrow_z),
                ]
                m.scale.x = 0.025                  # thin shaft
                m.scale.y = 0.06                   # small head
                m.scale.z = 0.0
                m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.75, 0.0, 0.55  # dim gold
                m.pose.orientation.w = 1.0
                arr.markers.append(m)

    # ─── Banner ──────────────────────────────────────────────────
    def _banner(self):
        L = rospy.loginfo
        L("=" * 64)
        L("path_corrector (core PathCorrector: %s)", self.corrector_name)
        L("  input path in = %s", self.input_path_topic)
        L("  bev        in = %s", self.bev_topic)
        L("  path      out = %s   (corrected, flown)", self.path_topic)
        L("  raw echo  out = %s   (input echo, viz)", self.raw_path_topic)
        if self.resample_min_spacing_m > 0.0:
            L("  resample: thin dense input to >= %.2fm before correcting",
              self.resample_min_spacing_m)
        if not self.enabled:
            L("  correction = OFF (passthrough: input -> %s verbatim)", self.path_topic)
        elif self._is_pf:
            cp = self.corrector.params
            L("  potential_field: centering=%s search=%.2fm corner_swing=%.2f "
              "step=%.2fm pin_goal=%s", cp.centering, cp.max_total_shift_m,
              cp.corner_swing, cp.center_step_m, cp.pin_last)
            L("  F_rep arrows: publish=%s scale=%.2f field=%s -> %s",
              self._publish_force_markers, self.force_arrow_scale,
              self._publish_force_field, self.forces_topic)
        else:
            ec = self.corrector.cfg
            L("  esdf: target_clearance=%.2fm search=%.2fm step=%.2fm iters=%d "
              "lateral=%s pin_goal=%s", ec.target_clearance_m, ec.max_total_shift_m,
              ec.max_step_m, ec.iterations, ec.lateral_only, ec.pin_last)
            L("  (force arrows are potential_field-only; not published for esdf)")
        L("=" * 64)


def main():
    try:
        PathCorrectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


# ============================================================================
# ROSPARAMS (all private ~; defaults in parentheses). The correction maths live
# in core.planning.safety.path_correction; this node owns ROS I/O only.
#
#   IO: ~input_path_topic (/path/waypoints_astar; the planner's raw path -- point
#         it at /path/waypoints_navdp for NavDP, or any planner topic)
#       ~bev_topic (/falcon/bev_2d) ~path_topic (/path/waypoints; corrected, flown)
#       ~raw_path_topic (/path/waypoints_raw; input echo for the BEV viewer's red)
#       ~frame_id (world)
#   strategy: ~enabled (true; false = passthrough, input -> path_topic verbatim)
#       ~corrector (potential_field | esdf; the core PathCorrector to build --
#         raises on an unknown name)
#       ~resample_min_spacing_m (0.0; thin a dense input path to >= this spacing
#         BEFORE correcting, to stop a lateral corrector zig-zagging on
#         near-coincident points -- e.g. NavDP. 0 = off; ~0.3-0.5 suits NavDP;
#         A*'s metre-scale waypoints are unaffected)
#   shared map-safety knobs (apply to whichever corrector is active):
#       ~inflate_radius_m (0.4; obstacle inflation for the corrected-path collision
#         clip -- match the planner's inflate_radius_m)
#       ~apf_collision_recheck (true; per-waypoint clip against inflation)
#       ~apf_treat_unknown_as_free (true; recentre over UNKNOWN cells too)
#       ~apf_unknown_damping (true) ~apf_unknown_radius_m (0.75)
#       ~apf_pin_last (true; keep the goal fixed)
#       ~apf_debug (true; TEMP per-waypoint in/push/new logging each cycle)
#   potential-field knobs (used when ~corrector=potential_field):
#       ~apf_occ_thresh (0.65) ~apf_sigma_m (0.6; Gaussian repulsion spread)
#       ~apf_centering (line_search | descent)
#       ~apf_center_step_m (0.05) ~apf_corner_swing (1.0; swing wide at corners)
#       ~apf_max_total_shift_m (2.0; line_search lateral search half-range)
#       ~apf_iterations (5) ~apf_gain (1.0) ~apf_max_step_m (0.4)
#         ~apf_smoothing_passes (2) ~apf_lateral_only (true) -- DESCENT ONLY
#       ~apf_min_clearance_m (0.0; >0 also pushes to a min distance-to-wall)
#   esdf knobs (used when ~corrector=esdf; ascend +grad D up the distance field):
#       ~esdf_occ_thresh (0.65) ~esdf_smooth_sigma_m (0.1; blur for a clean grad)
#       ~esdf_target_clearance_m (0.0; 0 = centre on the ESDF ridge / corridor
#         middle, >0 = stop once this far from walls -- a safety floor)
#       ~esdf_max_step_m (0.2) ~esdf_max_total_shift_m (1.0) ~esdf_iterations (12)
#       ~esdf_lateral_only (true; push perpendicular to the path)
#   force arrows (potential_field only): ~publish_forces (true)
#       ~forces_topic (/path/forces; visualization_msgs/MarkerArray, yellow)
#       ~force_arrow_scale (1.0) ~force_arrow_z (0.0)
#       ~publish_force_field (true; coarse gold F_rep field over free cells)
#       ~force_field_stride (4; field-arrow spacing in cells)
# ============================================================================
