#!/usr/bin/env python3
"""
sim_adapter_node.py -- ROS1 shim: make Gazebo's sjtu_drone look like the XTEND.

With this node running, the Gazebo sim publishes on the EXACT topic names the
real XTEND uses (/xtend/rgb, /xtend/depth_m, /xtend/april_tag_pose) and
RESAMPLES the camera images so the output matches the real XTEND's intrinsics
exactly -- so ``real_drone.launch`` is reused for sim with ZERO camera or topic
changes downstream. Two differences are absorbed entirely inside this node:

1. Camera intrinsics. The XTEND depth is 504x294 with fx=253.07, fy=287.54,
   cx=236.14, cy=81.73 -- an OFF-CENTRE principal point and, crucially,
   fx != fy. Gazebo's ``libgazebo_ros_camera.so`` has square pixels (fx == fy
   always) and a centred principal point, so neither a crop nor any stock
   setting can reproduce the XTEND. This node renders WIDER and BIGGER than the
   target and resamples to the exact target intrinsics with a separable
   nearest-neighbour map (``core.common.intrinsic_remap``): rescaling fx and fy
   independently reproduces the anisotropy, and the shift relocates the optical
   axis. The render FOV must be wide enough to contain the target -- if it is
   not, ``build_remap`` raises at startup (fail loud, never sample the edge).

2. Clocks. FALCON runs on the ROS1 WALL clock; Gazebo stamps depth with the SIM
   clock and publishes ``gt_pose`` as a bare Pose (no stamp at all). On the real
   XTEND the localization is computed FROM the depth and its stamp is OVERWRITTEN
   to that depth's stamp, so depth and pose are co-temporal by construction and
   ``mapping_sync`` pairs them with epsilon tolerance. This node reproduces that
   contract for sim: every depth frame is RE-STAMPED with ``rospy.Time.now()``
   (wall, fresh), and every pose is stamped with that SAME value, so depth and
   pose carry an identical wall-clock stamp -- they pair exactly, and they are
   fresh enough that FALCON's age checks accept them. (The real drone uses
   ``pose_adapter`` for its already-wall-clock localization; this node is
   sim-only and never touches that path.)

It also stands in for the XTEND's ROS2-owned DemoMode state machine: it OWNS
/xtend/demo_mode (latched) and honours transition requests on
/xtend/demo_mode_request. And it relays the cmd_vel rename: FALCON publishes
/cmd_vel (ROS1); Gazebo's sjtu_drone listens on /simple_drone/cmd_vel.

Responsibilities are split:
  - the intrinsic-resample is the ROS-free algorithm in
    ``core.common.intrinsic_remap`` (unit testable without ROS);
  - this node owns the ROS topics, the wall-clock restamp, the latched mode
    state, and the DemoMode vocabulary it emulates (the authoritative
    state-machine node lives elsewhere; in sim, this node stands in for it).

  in   ~in_rgb_topic   (Image, render_w x render_h)  /simple_drone/front/image_raw
  in   ~in_depth_topic (Image, render_w x render_h)  /simple_drone/front_depth/depth/image_raw
  in   ~in_pose_topic  (Pose)                         /simple_drone/gt_pose
  in   ~in_cmd_topic   (Twist, from FALCON)           /cmd_vel
  in   /xtend/demo_mode_request (String)
  out  ~out_rgb_topic   (Image, target_w x target_h)  /xtend/rgb
  out  ~out_depth_topic (Image, target_w x target_h)  /xtend/depth_m
  out  ~out_pose_topic  (PoseStamped)                 /xtend/april_tag_pose
  out  ~out_cmd_topic   (Twist, bridged to Gazebo)    /simple_drone/cmd_vel
  out  /xtend/demo_mode (String, latched)

See the file footer for the full rosparam list.
"""
import math

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseStamped, Twist
from std_msgs.msg import String

from sparx_agency.core.common.intrinsic_remap import build_remap, remap_raw_image


class DemoMode:
    """XTEND demo-mode payloads bridged over /xtend/demo_mode(_request).

    Defined locally because this node *is* the sim stand-in for the demo-mode
    state machine; the authoritative ROS2 state-machine node lives elsewhere.
    """

    FLY_STRAIGHT = "fly_straight"
    TURNING = "turning"
    VISUAL_SERVOING = "visual_servoing"
    ALL = {FLY_STRAIGHT, TURNING, VISUAL_SERVOING}


class SimAdapterNode:
    def __init__(self):
        rospy.init_node("sim_adapter")
        G = rospy.get_param

        # ── topic plumbing ──
        self.in_rgb_t = G("~in_rgb_topic", "/simple_drone/front/image_raw")
        self.in_depth_t = G("~in_depth_topic",
                            "/simple_drone/front_depth/depth/image_raw")
        self.in_pose_t = G("~in_pose_topic", "/simple_drone/gt_pose")
        self.in_cmd_t = G("~in_cmd_topic", "/cmd_vel")
        self.out_rgb_t = G("~out_rgb_topic", "/xtend/rgb")
        self.out_depth_t = G("~out_depth_topic", "/xtend/depth_m")
        self.out_pose_t = G("~out_pose_topic", "/xtend/april_tag_pose")
        self.out_cmd_t = G("~out_cmd_topic", "/simple_drone/cmd_vel")
        self.pose_frame = G("~pose_frame_id", "world")

        # ── target intrinsics (real XTEND depth calibration) ──
        # These MUST match the cam_* args real_drone.launch hands FALCON, so
        # the resampled stream is geometrically what FALCON is told it is.
        self.target_w = int(G("~target_width", 504))
        self.target_h = int(G("~target_height", 294))
        self.fx = float(G("~fx", 253.066668300591147))
        self.fy = float(G("~fy", 287.535109816403349))
        self.cx = float(G("~cx", 236.140442706411449))
        self.cy = float(G("~cy", 81.734160313465040))

        # ── render intrinsics (Gazebo SDF; square pixels, centred PP) ──
        # render_hfov MUST match the SDF <horizontal_fov> of front_camera and
        # front_depth_camera; render_w/h MUST match their <image> size. The
        # render must be WIDER and TALLER than the target's FOV -- build_remap
        # fails loud below if it is not. 640x480 @ 100 deg covers the XTEND
        # FOV with margin while keeping the resample near 1:1 (least blur).
        self.render_w = int(G("~render_width", 640))
        self.render_h = int(G("~render_height", 480))
        self.render_hfov = float(G("~render_hfov", 1.7453292519943295))  # 100 deg
        f_render = self.render_w / (2.0 * math.tan(self.render_hfov / 2.0))
        self.src_fx = self.src_fy = f_render
        self.src_cx = self.render_w / 2.0
        self.src_cy = self.render_h / 2.0

        # Build the nearest-neighbour resample map once. Fail loud at startup
        # if the render FOV cannot cover the target (the common misconfig).
        try:
            self.row_idx, self.col_idx = build_remap(
                self.src_fx, self.src_fy, self.src_cx, self.src_cy,
                self.render_w, self.render_h,
                self.fx, self.fy, self.cx, self.cy,
                self.target_w, self.target_h)
        except ValueError as e:
            rospy.logfatal("sim_adapter: %s", e)
            raise

        # Newest depth wall-stamp, copied onto every pose so depth and pose are
        # co-temporal for mapping_sync (None until the first depth arrives).
        self._last_depth_stamp = None

        # ── initial DemoMode state ──
        self.demo_mode = str(G("~initial_demo_mode", DemoMode.FLY_STRAIGHT))
        if self.demo_mode not in DemoMode.ALL:
            rospy.logwarn("sim_adapter: ~initial_demo_mode=%r invalid; forcing %s",
                          self.demo_mode, DemoMode.FLY_STRAIGHT)
            self.demo_mode = DemoMode.FLY_STRAIGHT

        # ── pub/sub ──
        self.pub_rgb = rospy.Publisher(self.out_rgb_t, Image, queue_size=1)
        self.pub_depth = rospy.Publisher(self.out_depth_t, Image, queue_size=1)
        self.pub_pose = rospy.Publisher(self.out_pose_t, PoseStamped, queue_size=10)
        self.pub_cmd = rospy.Publisher(self.out_cmd_t, Twist, queue_size=10)
        # latch=True ~ transient_local on the real ROS2 publisher, so a
        # late-joining subscriber immediately gets the current mode.
        self.pub_demo = rospy.Publisher("/xtend/demo_mode", String,
                                        queue_size=10, latch=True)

        rospy.Subscriber(self.in_rgb_t, Image, self._rgb_cb, queue_size=1)
        rospy.Subscriber(self.in_depth_t, Image, self._depth_cb, queue_size=1)
        rospy.Subscriber(self.in_pose_t, Pose, self._pose_cb, queue_size=10)
        rospy.Subscriber(self.in_cmd_t, Twist, self._cmd_cb, queue_size=10)
        rospy.Subscriber("/xtend/demo_mode_request", String,
                         self._demo_request_cb, queue_size=10)

        self.pub_demo.publish(String(data=self.demo_mode))
        self._banner()

    # ── encoding-agnostic intrinsic resample via the core algorithm ──
    def _remap(self, msg):
        if msg.width != self.render_w or msg.height != self.render_h:
            rospy.logwarn_throttle(
                5.0, "sim_adapter: incoming %dx%d, expected %dx%d -- SDF render "
                "size doesn't match ~render_width/height. Dropping frame.",
                msg.width, msg.height, self.render_w, self.render_h)
            return None
        data, step = remap_raw_image(msg.data, msg.height, msg.width, msg.step,
                                     self.row_idx, self.col_idx)
        out = Image()
        out.header = msg.header           # frame_id preserved; stamp reset by caller
        out.height, out.width = self.target_h, self.target_w
        out.encoding = msg.encoding
        out.is_bigendian = msg.is_bigendian
        out.step = step
        out.data = data
        return out

    # ── callbacks ──
    def _rgb_cb(self, msg):
        out = self._remap(msg)
        if out is not None:
            out.header.stamp = rospy.Time.now()   # fresh wall clock
            self.pub_rgb.publish(out)

    def _depth_cb(self, msg):
        out = self._remap(msg)
        if out is not None:
            # Restamp onto the wall clock (Gazebo's sim-clock stamp is
            # meaningless to FALCON) and remember it so the next poses pair.
            stamp = rospy.Time.now()
            out.header.stamp = stamp
            self._last_depth_stamp = stamp
            self.pub_depth.publish(out)

    def _pose_cb(self, msg):
        s = PoseStamped()
        # Stamp with the latest depth's wall stamp so pose and depth are
        # co-temporal for mapping_sync (mirrors the real XTEND, which stamps
        # localization with the depth it was computed from). Before the first
        # depth, fall back to now() -- those poses simply have no depth to pair.
        s.header.stamp = (self._last_depth_stamp
                          if self._last_depth_stamp is not None
                          else rospy.Time.now())
        s.header.frame_id = self.pose_frame
        s.pose = msg
        self.pub_pose.publish(s)

    def _cmd_cb(self, msg):
        # Pure passthrough under the name Gazebo's sjtu_drone listens on; the
        # bridge handles the ROS1->ROS2 hop.
        self.pub_cmd.publish(msg)

    def _demo_request_cb(self, msg):
        req = msg.data.strip()
        if req not in DemoMode.ALL:
            rospy.logwarn("sim_adapter: ignoring demo_mode_request %r -- "
                          "not one of %s", req, sorted(DemoMode.ALL))
            return
        if req == self.demo_mode:
            return
        rospy.loginfo("sim_adapter: demo_mode  %s -> %s", self.demo_mode, req)
        self.demo_mode = req
        self.pub_demo.publish(String(data=self.demo_mode))

    def _banner(self):
        L = rospy.loginfo
        L("=" * 64)
        L("sim_adapter -- Gazebo sjtu_drone -> XTEND topic/camera emulation")
        L("  rgb   : %s -> %s", self.in_rgb_t, self.out_rgb_t)
        L("  depth : %s -> %s", self.in_depth_t, self.out_depth_t)
        L("  pose  : %s -> %s   (frame_id=%s, stamp=depth wall-clock)",
          self.in_pose_t, self.out_pose_t, self.pose_frame)
        L("  cmd   : %s -> %s", self.in_cmd_t, self.out_cmd_t)
        L("  render %dx%d hfov=%.4frad -> fx=fy=%.3f cx=%.1f cy=%.1f",
          self.render_w, self.render_h, self.render_hfov,
          self.src_fx, self.src_cx, self.src_cy)
        L("  target %dx%d -> fx=%.3f fy=%.3f cx=%.3f cy=%.3f  (resample x%.3f y%.3f)",
          self.target_w, self.target_h, self.fx, self.fy, self.cx, self.cy,
          self.fx / self.src_fx, self.fy / self.src_fy)
        L("  demo mode: %s   (initial; latched on /xtend/demo_mode)", self.demo_mode)
        L("=" * 64)


def main():
    try:
        SimAdapterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()


# ============================================================================
# ROSPARAMS (all private ~; defaults in parentheses). The intrinsic resample
# lives in core.common.intrinsic_remap; this node owns the ROS topics, the
# wall-clock restamp, the latched DemoMode state and the mode vocabulary it
# emulates.
#
#   topics: ~in_rgb_topic (/simple_drone/front/image_raw)
#       ~in_depth_topic (/simple_drone/front_depth/depth/image_raw)
#       ~in_pose_topic (/simple_drone/gt_pose) ~in_cmd_topic (/cmd_vel)
#       ~out_rgb_topic (/xtend/rgb) ~out_depth_topic (/xtend/depth_m)
#       ~out_pose_topic (/xtend/april_tag_pose) ~out_cmd_topic (/simple_drone/cmd_vel)
#       ~pose_frame_id (world)
#   target intrinsics (must match real_drone.launch cam_*):
#       ~target_width (504) ~target_height (294)
#       ~fx (253.0667) ~fy (287.5351) ~cx (236.1404) ~cy (81.7342)
#   render intrinsics (must match the SDF cameras; render FOV must cover the
#       target or build_remap raises at startup):
#       ~render_width (640) ~render_height (480) ~render_hfov (1.74533 = 100 deg)
#   mode: ~initial_demo_mode (fly_straight)
# ============================================================================
