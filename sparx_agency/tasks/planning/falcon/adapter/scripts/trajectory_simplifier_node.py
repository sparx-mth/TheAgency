#!/usr/bin/env python3
"""trajectory_simplifier_node.py -- ROS1 adapter: clean up a corrected path.

This node owns ONE responsibility and is deliberately SEPARATE from the potential
field: it takes the safety-corrected ``nav_msgs/Path`` (the corrector pushed it
off the walls) and makes it CLEANER and easier to fly -- it merges near-duplicate
waypoints, flattens field-induced zig-zags, drops redundant collinear points and
enforces a sensible spacing. It derives no repulsion and owns no safety logic.

All of the cleanup maths live in ROS-free, unit-tested core
(``sparx_agency.core.planning.path_simplification``). This node owns only ROS
concerns: decoding the BEV + input path, building the obstacle clearance test the
core passes inject (``clear_fn``) so a cleanup step never routes through a wall the
corrector avoided, invoking the simplifier and publishing the flown path.

Pipeline position (separate from safety):
    planner -> path_corrector (potential field / safety) -> ~input_path_topic
            -> trajectory_simplifier (THIS node, cleanup) -> ~path_topic -> follower

Freeze semantics: one cleanup per INPUT PATH message, latched -- a BEV update alone
does NOT re-simplify, so the published trajectory stays frozen between planner
ticks (mirrors the planner + corrector freeze).

Passthrough: with ``~enabled:=false`` (or before the first BEV) the input path is
republished verbatim on ``~path_topic`` -- the follower always reads one topic.

  in   ~input_path_topic (Path)   /path/waypoints_safe  (corrected, pre-cleanup)
  in   ~bev_topic (OccupancyGrid)  /falcon/bev_2d        (for the clearance check)
  out  ~path_topic (Path, latched) /path/waypoints       (cleaned, flown)

See the file footer for the full rosparam list.
"""
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path

from sparx_agency.core.common.types import Pose2D
from sparx_agency.core.planning.environment import (
    OccupancyGrid2D, OccupancyGrid2DParams, OccupancyValues)
from sparx_agency.core.planning.safety.path_correction import InflatedGridCollisionChecker
from sparx_agency.core.planning.path_simplification import (
    TrajectorySimplifier2D, TrajectorySimplifierConfig)

# nav_msgs/OccupancyGrid int8 convention published by bev_publisher_node.
BEV_VALUES = OccupancyValues(free=0, occupied=100, unknown=-1)

_TRUE_STRINGS = ("true", "1", "yes", "on")
_FALSE_STRINGS = ("false", "0", "no", "off")


def _get_bool_param(name, default):
    """Read a boolean rosparam, raising on a value that is not clearly boolean.

    roslaunch leaves an unrecognised string (e.g. a typo'd ``"fales"``) as a raw
    string, and ``bool("fales")`` is ``True`` -- a plain ``bool(get_param(...))``
    cast would silently flip a default-off flag on. Validate explicitly and raise
    (per the repo "no silent fallbacks" rule).
    """
    value = rospy.get_param(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_STRINGS:
            return True
        if token in _FALSE_STRINGS:
            return False
    raise ValueError(
        "rosparam %s expected a boolean (true/false), got %r -- check the launch "
        "file / overrides for a typo" % (name, value))


class TrajectorySimplifierNode:
    def __init__(self):
        rospy.init_node("trajectory_simplifier")
        G = rospy.get_param

        self.input_path_topic = G("~input_path_topic", "/path/waypoints_safe")
        self.bev_topic = G("~bev_topic", "/falcon/bev_2d")
        self.path_topic = G("~path_topic", "/path/waypoints")
        self.frame_id = G("~frame_id", "world")

        # ~enabled false (or no BEV yet) -> verbatim passthrough, so the follower
        # always reads one topic whether or not cleanup runs.
        self.enabled = _get_bool_param("~enabled", True)
        self.debug = _get_bool_param("~debug", True)

        # Obstacle inflation for the clearance test the cleanup passes consult. Match
        # the corrector/planner inflation so "clear" means the same thing everywhere.
        self.inflate_radius_m = float(G("~inflate_radius_m", 0.4))

        self.cfg = TrajectorySimplifierConfig(
            merge_enabled=_get_bool_param("~merge_enabled", True),
            merge_radius_m=float(G("~merge_radius_m", 0.30)),
            zigzag_enabled=_get_bool_param("~zigzag_enabled", True),
            zigzag_angle_deg=float(G("~zigzag_angle_deg", 60.0)),
            zigzag_strength=float(G("~zigzag_strength", 0.5)),
            zigzag_passes=int(G("~zigzag_passes", 1)),
            collinear_enabled=_get_bool_param("~collinear_enabled", True),
            collinear_angle_deg=float(G("~collinear_angle_deg", 10.0)),
            max_segment_m=float(G("~max_segment_m", 3.0)),
            min_spacing_enabled=_get_bool_param("~min_spacing_enabled", True),
            min_spacing_m=float(G("~min_spacing_m", 1.0)),
            turn_keep_deg=float(G("~turn_keep_deg", 25.0)),
        )
        self.simplifier = TrajectorySimplifier2D(self.cfg)

        # Latest BEV as ONE object. _path_cb snapshots this reference once and
        # builds the collision checker from it, so a BEV arriving mid-simplify
        # cannot mix a new grid with an old inflation (the assignment is atomic).
        self.grid = None            # OccupancyGrid2D

        self.pub_path = rospy.Publisher(self.path_topic, Path, queue_size=1, latch=True)
        rospy.Subscriber(self.bev_topic, OccupancyGrid, self._bev_cb, queue_size=1)
        rospy.Subscriber(self.input_path_topic, Path, self._path_cb, queue_size=1)
        self._banner()

    # ─── Callbacks ───────────────────────────────────────────────
    def _bev_cb(self, msg):
        """Cache the latest BEV as one object. Does NOT re-simplify: cleanup is
        driven strictly by a new input path, so the flown trajectory stays frozen
        between planner ticks. The inflated collision checker is built per input
        path in _path_cb (planner cadence), not here (the BEV is ~10 Hz)."""
        first = self.grid is None
        self.grid = self._decode(msg)
        if first:
            i = msg.info
            rospy.loginfo("trajectory_simplifier: first BEV W=%d H=%d res=%.2f",
                          i.width, i.height, i.resolution)

    def _path_cb(self, msg):
        """Simplify one corrected path and publish the flown result."""
        pts = self._decode_path(msg)
        grid = self.grid                       # snapshot once (atomic reference read)
        if not self.enabled or len(pts) < 3 or grid is None:
            # Passthrough: cleanup off, trivial path, or no map to validate against.
            self._publish(pts, cleaned=False)
            return

        # Build the inflated collision checker from the SNAPSHOTTED grid; its
        # segment_clear is the clear_fn the cleanup passes consult before moving or
        # removing a waypoint. This is the SAME checker the corrector uses, so
        # "clear" is identical (inflation radius, out-of-bounds-permissive,
        # unknown-as-free) and cleanup never undoes the corrector's safety.
        checker = InflatedGridCollisionChecker(grid, self.inflate_radius_m)
        result = self.simplifier.simplify(pts, clear_fn=checker.segment_clear)
        out = result.points
        if self.debug:
            rospy.loginfo_throttle(
                5.0, "trajectory_simplifier: %d -> %d wp (%d removed, %d smoothed)",
                result.num_in, result.num_out, result.num_in - result.num_out,
                result.num_moved)
        self._publish(out, cleaned=True)

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

    def _publish(self, points, cleaned):
        """Publish the (cleaned or passed-through) path on ~path_topic."""
        self.pub_path.publish(self._path_msg(points))
        if len(points) >= 2:
            length = sum(a.distance_to(b) for a, b in zip(points[:-1], points[1:]))
            rospy.loginfo("trajectory_simplifier: PATH PUBLISHED %d wp %.2fm %s "
                          "first=(%.2f,%.2f) last=(%.2f,%.2f)", len(points), length,
                          "(cleaned)" if cleaned else "(passthrough)",
                          points[0].x, points[0].y, points[-1].x, points[-1].y)

    # ─── Banner ──────────────────────────────────────────────────
    def _banner(self):
        c, L = self.cfg, rospy.loginfo
        L("=" * 64)
        L("trajectory_simplifier (core TrajectorySimplifier2D)")
        L("  input path in = %s   (corrected, pre-cleanup)", self.input_path_topic)
        L("  bev        in = %s   (clearance check, inflate=%.2fm)",
          self.bev_topic, self.inflate_radius_m)
        L("  path      out = %s   (cleaned, flown)", self.path_topic)
        if not self.enabled:
            L("  cleanup = OFF (passthrough: input -> %s verbatim)", self.path_topic)
        else:
            L("  merge=%s r=%.2fm | zigzag=%s >%.0fdeg x%.2f p%d | collinear=%s "
              "<%.0fdeg cap=%.2fm | min_spacing=%s %.2fm keep_turn>%.0fdeg",
              c.merge_enabled, c.merge_radius_m, c.zigzag_enabled, c.zigzag_angle_deg,
              c.zigzag_strength, c.zigzag_passes, c.collinear_enabled,
              c.collinear_angle_deg, c.max_segment_m, c.min_spacing_enabled,
              c.min_spacing_m, c.turn_keep_deg)
        L("=" * 64)


def main():
    try:
        TrajectorySimplifierNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


# ============================================================================
# ROSPARAMS (all private ~; defaults in parentheses). The cleanup maths live in
# core.planning.path_simplification; this node owns ROS I/O + the clearance test.
#
#   IO: ~input_path_topic (/path/waypoints_safe; the corrector's safe path)
#       ~bev_topic (/falcon/bev_2d) ~path_topic (/path/waypoints; cleaned, flown)
#       ~frame_id (world)
#   gate: ~enabled (true; false = verbatim passthrough)
#         ~debug (true; per-cycle in->out waypoint logging)
#         ~inflate_radius_m (0.4; obstacle inflation for the clearance test -- match
#           the corrector/planner inflate_radius_m so "clear" is consistent)
#   merge near-duplicates:  ~merge_enabled (true) ~merge_radius_m (0.30)
#   zig-zag smoothing:      ~zigzag_enabled (true) ~zigzag_angle_deg (60)
#       ~zigzag_strength (0.5; 0..1 toward the neighbour midpoint) ~zigzag_passes (1)
#   collinear simplify:     ~collinear_enabled (true) ~collinear_angle_deg (10)
#       ~max_segment_m (3.0; never drop a point if the bypass leg would exceed this)
#   min spacing (turns excepted): ~min_spacing_enabled (true) ~min_spacing_m (1.0)
#       ~turn_keep_deg (25; turns sharper than this may sit closer than min_spacing)
#
# Every geometry-changing pass is validated by an inflated line-of-sight clear_fn,
# so cleanup never routes the path through a wall the corrector avoided -- the
# "smoothing is not safety" separation holds while safety is still respected.
# ============================================================================
