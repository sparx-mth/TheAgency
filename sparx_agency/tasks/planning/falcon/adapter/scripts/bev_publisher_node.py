#!/usr/bin/env python3
"""
bev_publisher_node.py -- ROS1 adapter: FALCON voxel clouds -> nav_msgs/OccupancyGrid.

Thin glue around sparx_agency.core.mapping.bev.BevProjector. The node owns ONLY
ROS concerns: rosparams, bounds resolution (launch > /map_config > fallback),
cloud parsing, latched publishing, per-map wall overrides, and logging. All
projection logic lives in core and is unit-testable without ROS.

Runs inside FALCON's Noetic container (the falcon_adapter package). Drop-in for
the legacy bev_publisher.py: identical topics and message types.

  in   ~occ_topic  (PointCloud2)  /voxel_mapping/occupancy_grid_occupied
  in   ~free_topic (PointCloud2)  /voxel_mapping/occupancy_grid_free
  out  ~out_topic  (OccupancyGrid, latched)  /falcon/bev_2d

See the bottom of this file for the full rosparam list and a launch snippet.
"""
import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2

from sparx_agency.core.mapping.bev import BevConfig, BevProjector, OCCUPIED
from cloud_utils import cloud_to_xyz   # sibling module in this scripts/ dir


class BevPublisherNode:
    def __init__(self):
        rospy.init_node("bev_publisher")
        G = rospy.get_param

        self.out_topic = G("~out_topic", "/falcon/bev_2d")
        self.occ_topic = G("~occ_topic", "/voxel_mapping/occupancy_grid_occupied")
        self.free_topic = G("~free_topic", "/voxel_mapping/occupancy_grid_free")
        self.publish_hz = float(G("~publish_hz", 10.0))
        self.always_recompute = bool(G("~always_recompute", False))
        self.skip_unchanged = bool(G("~skip_unchanged_publish", True))
        margin = float(G("~bbox_margin_m", 1.0))

        x_min, sx0 = self._bound("bbox_xmin", "map_min_x", -12.0, -margin)
        y_min, sy0 = self._bound("bbox_ymin", "map_min_y", -12.0, -margin)
        x_max, sx1 = self._bound("bbox_xmax", "map_max_x", 12.0, +margin)
        y_max, sy1 = self._bound("bbox_ymax", "map_max_y", 12.0, +margin)

        self.cfg = BevConfig(
            resolution_m=float(G("~resolution", 0.15)),
            x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
            frame_id=G("~frame_id", "world"),
            occ_dilate_cells=int(G("~occ_dilate_cells", 0)),
            z_floor=float(G("~z_floor", 0.30)),
            z_ceil=float(G("~z_ceil", 2.20)),
            z_peak=float(G("~z_peak", 1.00)),
            weight_profile=str(G("~weight_profile", "triangular")),
            weight_sigma=float(G("~weight_sigma", 0.50)),
            voxel_size_m=(float(G("~voxel_size_m"))
                          if rospy.has_param("~voxel_size_m") else None),
            occ_weight_thresh=float(G("~occ_weight_thresh", 1.2)),
            min_occ_voxels=int(G("~min_occ_voxels", 2)),
            min_free_voxels=int(G("~min_free_voxels", 1)),
            confirm_3d=bool(G("~confirm_3d", True)),
            neighbors_3d=int(G("~neighbors_3d", 6)),
            min_occ_neighbors_3d=int(G("~min_occ_neighbors_3d", 1)),
            protect_openings=bool(G("~protect_openings", True)),
            door_band_m=float(G("~door_band_m", 0.60)),
            door_free_voxels=int(G("~door_free_voxels", 2)),
            door_occ_tol=int(G("~door_occ_tol", 0)),
            wall_fill_mode=str(G("~wall_fill_mode", "directional")),
            wall_fill_neighbors=int(G("~wall_fill_neighbors", 5)),
            wall_fill_iters=int(G("~wall_fill_iters", 1)),
            temporal_filter=bool(G("~temporal_filter", False)),
            t_inc=float(G("~t_inc", 1.0)),
            t_dec=float(G("~t_dec", 1.0)),
            t_max=float(G("~t_max", 5.0)),
            t_on=float(G("~t_on", 2.0)),
            t_off=float(G("~t_off", 0.5)),
        )
        self.projector = BevProjector(self.cfg)
        self.spec = self.projector.lattice.spec()
        self.walls = self._load_walls()
        bw = rospy.get_param("/map_config/behind_wall_x", None)
        self.behind_wall_x = None if bw is None else float(bw)
        # Static env obstacles -> a force-occupied mask stamped each frame
        # before dilation (so manual/back walls inflate, as in the legacy node).
        self.force_occ = self._build_force_occ()

        self._occ = np.empty((0, 3), np.float32)
        self._free = np.empty((0, 3), np.float32)
        self._grid = None
        self._dirty = True
        self._hb = dict(occ=0, free=0, pub=0)

        self.pub = rospy.Publisher(self.out_topic, OccupancyGrid,
                                   queue_size=1, latch=True)
        rospy.Subscriber(self.occ_topic, PointCloud2, self._occ_cb, queue_size=2)
        rospy.Subscriber(self.free_topic, PointCloud2, self._free_cb, queue_size=2)
        rospy.Timer(rospy.Duration(1.0 / self.publish_hz), self._tick)
        rospy.Timer(rospy.Duration(5.0), self._heartbeat)
        self._banner(sx0, sx1, sy0, sy1)

    # -- rosparams ------------------------------------------------------------
    @staticmethod
    def _bound(local, mapcfg, fallback, margin_signed):
        """~<local> (exact) > /map_config/map_size/<mapcfg> (+margin) > fallback."""
        if rospy.has_param("~" + local):
            return float(rospy.get_param("~" + local)), "launch"
        g = "/map_config/map_size/" + mapcfg
        if rospy.has_param(g):
            return float(rospy.get_param(g)) + margin_signed, "mapcfg"
        return float(fallback) + margin_signed, "fallback"

    def _load_walls(self):
        walls = []
        for w in (rospy.get_param("/map_config/walls", []) or []):
            try:
                yc, t = float(w["y"]), float(w.get("thickness", 0.10))
                walls.append((float(w["x_min"]), float(w["x_max"]),
                              yc - 0.5 * t, yc + 0.5 * t))
            except (KeyError, TypeError, ValueError) as e:
                rospy.logwarn("bev: bad wall %r: %s", w, e)
        return walls

    # -- subscribers ----------------------------------------------------------
    def _occ_cb(self, msg):
        self._occ = cloud_to_xyz(msg); self._hb["occ"] += 1; self._dirty = True

    def _free_cb(self, msg):
        self._free = cloud_to_xyz(msg); self._hb["free"] += 1; self._dirty = True

    # -- per-map overrides (env-specific; kept out of core) -------------------
    def _build_force_occ(self):
        """Static (H,W) bool mask: virtual back-wall + manual walls, or None."""
        s = self.spec
        if self.behind_wall_x is None and not self.walls:
            return None
        mask = np.zeros((s.height, s.width), bool)
        if self.behind_wall_x is not None:
            cx = min(s.width, int((self.behind_wall_x - s.origin_x) / s.resolution_m))
            if cx > 0:
                mask[:, :cx] = True
        for x0, x1, y0, y1 in self.walls:
            cx0 = max(0, int(np.floor((x0 - s.origin_x) / s.resolution_m)))
            cx1 = min(s.width, int(np.ceil((x1 - s.origin_x) / s.resolution_m)))
            cy0 = max(0, int(np.floor((y0 - s.origin_y) / s.resolution_m)))
            cy1 = min(s.height, int(np.ceil((y1 - s.origin_y) / s.resolution_m)))
            if cx1 > cx0 and cy1 > cy0:
                mask[cy0:cy1, cx0:cx1] = True
        return mask

    # -- publish --------------------------------------------------------------
    def _tick(self, _evt):
        if self._dirty or self._grid is None or self.always_recompute:
            self._dirty = False
            _, self._grid = self.projector.project(
                self._occ, self._free, force_occ=self.force_occ)
            self._publish()
        elif not self.skip_unchanged:
            self._publish()

    def _publish(self):
        s = self.spec
        m = OccupancyGrid()
        m.header.stamp = rospy.Time.now()
        m.header.frame_id = s.frame_id
        m.info.map_load_time = m.header.stamp
        m.info.resolution = s.resolution_m
        m.info.width, m.info.height = s.width, s.height
        m.info.origin.position.x = s.origin_x
        m.info.origin.position.y = s.origin_y
        m.info.origin.orientation.w = 1.0
        m.data = self._grid.flatten().tolist()
        self.pub.publish(m)
        self._hb["pub"] += 1

    def _heartbeat(self, _evt):
        st = self.projector.last_stats
        rospy.loginfo("bev hb  in occ=%d free=%d  pub=%d  |  voxels raw=%d conf=%d  "
                      "|  grid occ=%d free=%d unk=%d open=%d fill=%d",
                      self._hb["occ"], self._hb["free"], self._hb["pub"],
                      st.get("raw", 0), st.get("confirmed", 0), st.get("occ", 0),
                      st.get("free", 0), st.get("unknown", 0),
                      st.get("openings", 0), st.get("fill", 0))
        # Loud warning when most occupied points fall outside the BEV bounds
        # (a strong hint the bbox / map_size is wrong for this environment).
        if self._occ.shape[0]:
            _, _, inb = self.projector.lattice.world_to_cell(self._occ)
            n_tot, n_in = int(self._occ.shape[0]), int(inb.sum())
            if n_tot > 100 and n_in < 0.9 * n_tot:
                c = self.cfg
                rospy.logwarn_throttle(
                    20.0, "bev: %d/%d occ points OUTSIDE bbox x=[%.1f,%.1f] "
                    "y=[%.1f,%.1f] -- bounds may be wrong",
                    n_tot - n_in, n_tot, c.x_min, c.x_max, c.y_min, c.y_max)
        self._hb = dict(occ=0, free=0, pub=0)

    def _banner(self, sx0, sx1, sy0, sy1):
        s, c, L = self.spec, self.cfg, rospy.loginfo
        L("=" * 64)
        L(" BEV publisher (core.mapping.bev)  %dx%d @ %.3fm  frame=%s",
          s.width, s.height, s.resolution_m, s.frame_id)
        L(" bounds x=[%.2f,%.2f](%s/%s)  y=[%.2f,%.2f](%s/%s)",
          c.x_min, c.x_max, sx0, sx1, c.y_min, c.y_max, sy0, sy1)
        L(" z=[%.2f,%.2f] peak=%.2f profile=%s  |  occ: w>=%.2f & n>=%d",
          c.z_floor, c.z_ceil, c.z_peak, c.weight_profile,
          c.occ_weight_thresh, c.min_occ_voxels)
        L(" confirm_3d=%s(conn=%d,min=%d) protect=%s wall=%s dilate=%d",
          c.confirm_3d, c.neighbors_3d, c.min_occ_neighbors_3d,
          c.protect_openings, c.wall_fill_mode, c.occ_dilate_cells)
        if self.behind_wall_x is not None:
            L(" behind_wall_x=%.2f", self.behind_wall_x)
        if self.walls:
            L(" manual walls: %d", len(self.walls))
        L(" in  occ =%s", self.occ_topic)
        L(" in  free=%s", self.free_topic)
        L(" out     =%s  @%.1fHz latched", self.out_topic, self.publish_hz)
        L("=" * 64)


def main():
    try:
        BevPublisherNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


# ============================================================================
# ROSPARAMS (all private ~; defaults in parentheses). Geometry/tuning that used
# to be hard-coded in the node now lives in core BevConfig; the node only maps
# rosparams -> BevConfig and resolves bounds/overrides from /map_config.
#
#   IO / runtime
#     ~out_topic (/falcon/bev_2d)  ~occ_topic (/voxel_mapping/occupancy_grid_occupied)
#     ~free_topic (/voxel_mapping/occupancy_grid_free)  ~frame_id (world)
#     ~publish_hz (10.0)  ~always_recompute (false)  ~skip_unchanged_publish (true)
#   bounds: ~bbox_{xmin,xmax,ymin,ymax} (launch) > /map_config/map_size/* (+~bbox_margin_m, 1.0) > +-12
#   column/height: ~z_floor (.30) ~z_ceil (2.20) ~z_peak (1.00)
#     ~weight_profile (triangular) ~weight_sigma (.50) ~voxel_size_m (=resolution)
#   occupancy: ~resolution (.15) ~occ_weight_thresh (1.2) ~min_occ_voxels (2) ~min_free_voxels (1)
#   3D confirm: ~confirm_3d (true) ~neighbors_3d (6) ~min_occ_neighbors_3d (1)
#   doors: ~protect_openings (true) ~door_band_m (.60) ~door_free_voxels (2) ~door_occ_tol (0)
#   walls: ~wall_fill_mode (directional) ~wall_fill_neighbors (5) ~wall_fill_iters (1)
#   temporal (optional): ~temporal_filter (false) ~t_inc (1) ~t_dec (1) ~t_max (5) ~t_on (2) ~t_off (0.5)
#   dilation: ~occ_dilate_cells (0)
#   per-map overrides (node-side, from /map_config): behind_wall_x, walls[]
#
# Launch (replaces the legacy bev_publisher block):
#   <node pkg="falcon_adapter" type="bev_publisher_node.py" name="bev_publisher"
#         output="screen">
#     <param name="resolution"        value="0.15"/>
#     <param name="z_peak"            value="1.00"/>   <!-- flight altitude -->
#     <param name="wall_fill_mode"    value="directional"/>
#   </node>
# ============================================================================