#!/usr/bin/env python3
"""
sim_adapter_node.py -- ROS1 shim: make Gazebo's sjtu_drone look like the XTEND.

With this node running, the Gazebo sim publishes on the EXACT topic names the
real XTEND uses (/xtend/rgb, /xtend/depth_m, /flow_depth/pose_est) and crops the
camera images so the cropped output has the real XTEND's off-centre principal
point -- so real_drone.launch can be reused for sim with zero remapping
downstream.

It also stands in for the XTEND's ROS2-owned DemoMode state machine: it OWNS
/xtend/demo_mode (latched) and honours transition requests on
/xtend/demo_mode_request. And it relays the cmd_vel rename: FALCON publishes
/cmd_vel (ROS1); Gazebo's sjtu_drone listens on /simple_drone/cmd_vel.

Responsibilities are split:
  - the principal-point-relocating crop is the ROS-free algorithm in
    ``core.common.principal_point_crop`` (unit tested without ROS);
  - this node owns the ROS topics, the latched mode state, and the DemoMode
    vocabulary it emulates (the authoritative state-machine node lives
    elsewhere; in sim, this node stands in for it).

Why crop at all: stock Gazebo emits a centred principal point (cx=W/2, cy=H/2);
the real XTEND's is well off-centre. The SDF cameras render BIGGER than the
target at the XTEND's focal length, and this node crops asymmetrically down to
the target size -- cropping preserves focal length per pixel, so the cropped
principal point lands where the real camera's is.

  in   ~in_rgb_topic   (Image, render_w x render_h)  /simple_drone/front/image_raw
  in   ~in_depth_topic (Image, render_w x render_h)  /simple_drone/front_depth/depth/image_raw
  in   ~in_pose_topic  (Pose)                         /simple_drone/gt_pose
  in   ~in_cmd_topic   (Twist, from FALCON)           /cmd_vel
  in   /xtend/demo_mode_request (String)
  out  ~out_rgb_topic   (Image, target_w x target_h)  /xtend/rgb
  out  ~out_depth_topic (Image, target_w x target_h)  /xtend/depth_m
  out  ~out_pose_topic  (PoseStamped)                 /flow_depth/pose_est
  out  ~out_cmd_topic   (Twist, bridged to Gazebo)    /simple_drone/cmd_vel
  out  /xtend/demo_mode (String, latched)

See the file footer for the full rosparam list.
"""
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseStamped, Twist
from std_msgs.msg import String

from sparx_agency.core.common.principal_point_crop import (
    compute_crop_offsets, crop_raw_image)


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

        # ── camera geometry (real XTEND target vs Gazebo render) ──
        self.target_w = int(G("~target_width", 504))
        self.target_h = int(G("~target_height", 392))
        self.cx = float(G("~cx", 222.273))
        self.cy = float(G("~cy", 108.548))
        self.render_w = int(G("~render_width", 600))
        self.render_h = int(G("~render_height", 600))

        # Crop offsets relocate the optical axis to (cx, cy). Fail loudly at
        # startup if the window does not fit the render (the common misconfig).
        try:
            self.crop_x, self.crop_y = compute_crop_offsets(
                self.render_w, self.render_h, self.target_w, self.target_h,
                self.cx, self.cy)
        except ValueError as e:
            rospy.logfatal("sim_adapter: %s", e)
            raise

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

    # ── encoding-agnostic crop via the core algorithm ──
    def _crop(self, msg):
        if msg.width != self.render_w or msg.height != self.render_h:
            rospy.logwarn_throttle(
                5.0, "sim_adapter: incoming %dx%d, expected %dx%d -- SDF render "
                "size doesn't match ~render_width/height. Dropping frame.",
                msg.width, msg.height, self.render_w, self.render_h)
            return None
        data, step = crop_raw_image(msg.data, msg.height, msg.width, msg.step,
                                    self.crop_x, self.crop_y,
                                    self.target_w, self.target_h)
        out = Image()
        out.header = msg.header           # preserve stamp + frame_id
        out.height, out.width = self.target_h, self.target_w
        out.encoding = msg.encoding
        out.is_bigendian = msg.is_bigendian
        out.step = step
        out.data = data
        return out

    # ── callbacks ──
    def _rgb_cb(self, msg):
        out = self._crop(msg)
        if out is not None:
            self.pub_rgb.publish(out)

    def _depth_cb(self, msg):
        out = self._crop(msg)
        if out is not None:
            self.pub_depth.publish(out)

    def _pose_cb(self, msg):
        s = PoseStamped()
        s.header.stamp = rospy.Time.now()
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
        L("  pose  : %s -> %s   (frame_id=%s)",
          self.in_pose_t, self.out_pose_t, self.pose_frame)
        L("  cmd   : %s -> %s", self.in_cmd_t, self.out_cmd_t)
        L("  render %dx%d -> target %dx%d   cx=%.3f cy=%.3f   crop=(%d,%d)",
          self.render_w, self.render_h, self.target_w, self.target_h,
          self.cx, self.cy, self.crop_x, self.crop_y)
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
# ROSPARAMS (all private ~; defaults in parentheses). The principal-point crop
# lives in core.common.principal_point_crop; this node owns the ROS topics, the
# latched DemoMode state and the mode vocabulary it emulates.
#
#   topics: ~in_rgb_topic (/simple_drone/front/image_raw)
#       ~in_depth_topic (/simple_drone/front_depth/depth/image_raw)
#       ~in_pose_topic (/simple_drone/gt_pose) ~in_cmd_topic (/cmd_vel)
#       ~out_rgb_topic (/xtend/rgb) ~out_depth_topic (/xtend/depth_m)
#       ~out_pose_topic (/flow_depth/pose_est) ~out_cmd_topic (/simple_drone/cmd_vel)
#       ~pose_frame_id (world)
#   camera: ~target_width (504) ~target_height (392) ~cx (222.273) ~cy (108.548)
#       ~render_width (600) ~render_height (600)   [must match the SDF cameras]
#   mode: ~initial_demo_mode (fly_straight)
# ============================================================================
