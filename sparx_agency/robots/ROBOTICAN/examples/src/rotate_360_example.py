#!/usr/bin/python3

import os
import time
import math
import json
import threading
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from fcu_driver_interfaces.msg import CommandLong, UAVState, ManualControl
from rooster_handler_interfaces.msg import KeepAlive
from rooster_manager_interfaces.msg import RoosterState


MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_CONDITION_YAW = 115

MANUAL_CONTROL_RATE_HZ = 20.0
KEEP_ALIVE_RATE_HZ = 2.0
PRE_ARM_SETTLE_TIME = 0.5
ARM_CONFIRM_TIMEOUT = 5.0
SWITCH_ALTITUDE = 1.0
POSITION_RETRY_INTERVAL = 3.0
MANUAL_CONTROL_NEUTRAL_Z = 540.0

# Rotation: completion is tracked from UAVState.azimuth (real heading feedback, radians,
# 0=North, positive CW), not a time estimate - a first flight test showed MAV_CMD_CONDITION_YAW
# achieves only a fraction of the requested deg/s in Position mode here (~90 deg actually
# turned in the ~14s a clean 360 deg at 30 deg/s should have taken - probably the continuously
# published manual_control stream's neutral r=0.0 damping the commanded rate, since
# manual_control is a live, continuously-effective input in this codebase, not inert - see
# force_arm_and_roll_example.py). Rather than guess a corrected rate, re-send the command
# with whatever's left to turn until the real accumulated azimuth confirms the turn is done.
YAW_TURN_DEGREES = 360.0
YAW_RATE_DEG_S = 30.0        # requested rate; actual rate may be lower - see above
YAW_DIRECTION = 1.0          # 1 = clockwise, -1 = counter-clockwise
YAW_RELATIVE = 1.0           # 1 = relative to current heading, 0 = absolute
YAW_TOLERANCE_DEG = 10.0     # consider the turn done within this margin
YAW_RESEND_INTERVAL_S = 4.0  # re-issue the command for the remaining degrees at this cadence
YAW_MAX_WAIT_S = 40.0        # safety fallback if azimuth feedback never confirms completion
YAW_SETTLE_TIME_S = 3.0      # hold still after Position mode confirms, before yawing - lets
                             # the optical-flow-based position/velocity hold settle first

# Safety: a flight test hit repeated "ddsi_udp_conn_write ... failed" DDS/multicast errors
# mid-flight, visible in both this node's and land_and_disarm_example.py's logs at the same
# moment - a real link dropout, not a bug in either script. State feedback (armed/flight_mode/
# azimuth) can silently stop updating during one of these without the step machine noticing,
# which would mean we keep re-issuing commands based on stale data. Track how long it's been
# since each topic last delivered a message, and stop automated commands rather than trust
# data that's gone silent - holding is safer than continuing blind.
STATE_STALE_TIMEOUT_S = 2.0

# Pitch forward slightly during the capture window so the camera (rigidly body-mounted, no
# separate gimbal) looks more toward the floor/room contents instead of straight ahead at
# wall height. Sign/magnitude are a first guess, NOT flight-validated - MAVLink's usual
# convention is manual_control.x = pitch, negative = nose down, but this codebase's actual
# mapping hasn't been confirmed. Start small and watch closely the first time; if it drifts
# forward/backward instead of just tilting, the sign or magnitude needs correcting.
CAPTURE_PITCH_DOWN = -50.0

# Scan capture: runs from Position-mode confirmation through the end of the turn, at the
# same device, resolution and MJPG format as auto_capture_interval.py in NanoLLM_VILA_and_OWL
# (confirmed working via OpenCV inside this container: 640x360 MJPG open+read succeeds -
# camera_frames_example.py's H264 caps-negotiation issue in this same workspace is specific
# to its hardware H264 decode path, not plain MJPG capture).
#
# /home/iai/captures (the NanoLLM_VILA_and_OWL repo's own convention, used on the host) is
# NOT mounted into this container - only /home/iai/frames is (-> host's
# ~/iai_ws/frames, same as camera_frames_example.py already uses), confirmed via
# `docker inspect iai --format '{{range .Mounts}}...'`. Write there, in a "rotate_scan_*"
# subfolder so repeated runs don't overwrite each other; the host side copies it into the
# shared captures root afterward (see run_rotate_scan_test.sh in that repo).
CAPTURE_DEVICE = "/dev/HD_CAMERA"
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 360
CAPTURE_MJPG = True
# 0.5s produced 79 frames over a 40s run - more than comm_manager_vllm.py (1-9s/frame VLM
# calls) could process in a reasonable time. 1.5s keeps decent coverage across a ~14-40s
# turn (9-27 frames) while actually finishing before someone gives up waiting.
CAPTURE_INTERVAL_S = 1.5
FRAMES_ROOT = "/home/iai/frames"

STEP_WAIT_ARM_CONFIRM = 0
STEP_TAKEOFF = 1
STEP_MONITOR_ALTITUDE = 2
STEP_VERIFY_POSITION = 3
STEP_SETTLE_BEFORE_YAW = 4
STEP_YAW_360 = 5
STEP_WAIT_YAW_DONE = 6
STEP_DONE = 7


def get_namespace_id():
    rooster_id = os.environ.get('ROOSTER_ID')
    if rooster_id:
        return rooster_id
    return "R" + os.environ.get('ROS_DOMAIN_ID', '1')


class TakeoffAndRotate(Node):
    def __init__(self):
        super().__init__('takeoff_rotate_node')
        self.id = get_namespace_id()
        self.armed = False
        self.flight_mode = None
        self.current_altitude = None
        self.altitude_at_arm = None
        self.takeoff_sent_time = None
        self.arm_wait_start_time = None
        self.position_wait_start_time = None
        self.settle_start_time = None
        self.last_state_msg_time = None
        self.last_uav_state_msg_time = None
        self.current_azimuth = None
        self.yaw_last_azimuth = None
        self.yaw_accumulated_deg = 0.0
        self.yaw_wait_start_time = None
        self.yaw_last_resend_time = None
        self.step = None

        self.cam_thread = None
        self.cam_running = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.capture_timer = None
        self.capture_session_dir = None
        self.capture_count = 0

        # Services
        self.arm_rooster = self.create_client(SetBool, f'/{self.id}/arm_rooster')

        # MAVLink command publisher
        self.command_long_pub = self.create_publisher(
            CommandLong, f'/{self.id}/fcu/command_long', 10)

        # State subscriptions
        self.state_sub = self.create_subscription(
            RoosterState, f'/{self.id}/state', self.state_cb, 10)
        self.uav_state_sub = self.create_subscription(
            UAVState, f'/{self.id}/fcu/state', self.uav_state_cb, 10)

        self.manual_control = ManualControl()
        self.manual_control.x = 0.0
        self.manual_control.y = 0.0
        self.manual_control.z = 0.0
        self.manual_control.r = 0.0
        self.manual_control.buttons = 0
        self.manual_control_pub = self.create_publisher(
            ManualControl, f'/{self.id}/manual_control', 10)

        self.manual_control_timer = self.create_timer(
            1.0 / MANUAL_CONTROL_RATE_HZ, self.publish_manual_control)

        self.keep_alive_is_active = False
        self.keep_alive_mode = KeepAlive.FLIGHT_MODE_NONE
        self.keep_alive_pub = self.create_publisher(KeepAlive, f'/{self.id}/keep_alive', 10)
        self.keep_alive_timer = self.create_timer(
            1.0 / KEEP_ALIVE_RATE_HZ, self.publish_keep_alive)

        self.get_logger().info(
            f"Streaming centered manual_control for {PRE_ARM_SETTLE_TIME}s before arming...")
        self.arm_delay_timer = self.create_timer(PRE_ARM_SETTLE_TIME, self.arm_delay_cb)

        self.step_timer = self.create_timer(0.2, self.step_timer_cb)

    # ---------- callbacks ----------
    def publish_manual_control(self):
        self.manual_control_pub.publish(self.manual_control)

    def publish_keep_alive(self):
        msg = KeepAlive()
        msg.is_active = self.keep_alive_is_active
        msg.requested_flight_mode = self.keep_alive_mode
        msg.command_reboot = False
        self.keep_alive_pub.publish(msg)

    def request_flight_mode(self, mode):
        # While in Takeoff, rooster_manager only acts on a request that differs from the
        # previous one, and that previous request persists across flights. Sending NONE
        # first (maps to Invalid, ignored) guarantees the real request counts as a change.
        self.keep_alive_is_active = True
        self.keep_alive_mode = KeepAlive.FLIGHT_MODE_NONE
        self.publish_keep_alive()
        self.keep_alive_mode = mode
        self.publish_keep_alive()

    def arm_delay_cb(self):
        self.arm_delay_timer.cancel()
        self.get_logger().info("Requesting force arm (via arm_rooster)")
        req = SetBool.Request()
        req.data = True
        future = self.arm_rooster.call_async(req)

        def arm_response_cb(fut):
            result = fut.result()
            if result is None or not result.success:
                self.get_logger().error(f"arm_rooster call failed: {result}")
        future.add_done_callback(arm_response_cb)

        self.altitude_at_arm = self.current_altitude or 0.0
        self.arm_wait_start_time = time.monotonic()
        self.step = STEP_WAIT_ARM_CONFIRM

    def state_cb(self, msg):
        self.last_state_msg_time = time.monotonic()
        if self.armed != msg.armed or self.flight_mode != msg.flight_mode:
            self.armed = msg.armed
            self.flight_mode = msg.flight_mode
            self.get_logger().info(f"RoosterState armed={msg.armed} flight_mode={msg.flight_mode}")

    def uav_state_cb(self, msg):
        self.last_uav_state_msg_time = time.monotonic()
        self.current_altitude = msg.altitude_relative
        if self.yaw_last_azimuth is not None:
            # azimuth wraps at +-pi; a plain subtraction breaks crossing that boundary,
            # so normalize the per-sample delta into (-pi, pi] before accumulating.
            delta = msg.azimuth - self.yaw_last_azimuth
            delta = (delta + math.pi) % (2 * math.pi) - math.pi
            self.yaw_accumulated_deg += math.degrees(delta)
        self.yaw_last_azimuth = msg.azimuth
        self.current_azimuth = msg.azimuth

    # ---------- helpers ----------
    def send_command_long(self, command, p1=0.0, p2=0.0, p3=0.0, p4=0.0, p5=0.0, p6=0.0, p7=0.0):
        cmd = CommandLong()
        cmd.target_system = 1
        cmd.target_component = 1
        cmd.command = command
        cmd.confirmation = 0
        cmd.param1, cmd.param2, cmd.param3 = float(p1), float(p2), float(p3)
        cmd.param4, cmd.param5, cmd.param6 = float(p4), float(p5), float(p6)
        cmd.param7 = float(p7)
        self.command_long_pub.publish(cmd)

    def height_above_arm(self):
        if self.current_altitude is None or self.altitude_at_arm is None:
            return None
        return self.current_altitude - self.altitude_at_arm

    def comms_stale(self):
        if self.last_state_msg_time is None or self.last_uav_state_msg_time is None:
            return False  # nothing received yet at all - a startup condition, not a dropout
        now = time.monotonic()
        return (now - self.last_state_msg_time > STATE_STALE_TIMEOUT_S or
                now - self.last_uav_state_msg_time > STATE_STALE_TIMEOUT_S)

    # ---------- camera scan capture ----------
    def start_camera_capture(self):
        if self.cam_thread is not None:
            return
        self.manual_control.x = CAPTURE_PITCH_DOWN
        self.cam_running = True
        self.cam_thread = threading.Thread(target=self._camera_read_loop, daemon=True)
        self.cam_thread.start()

        ts = time.strftime("%Y_%m_%d___%H_%M_%S")
        self.capture_session_dir = Path(FRAMES_ROOT) / f"rotate_scan_{ts}"
        self.capture_session_dir.mkdir(parents=True, exist_ok=True)
        self.capture_count = 0
        self.get_logger().info(f"Capturing rotation scan to {self.capture_session_dir}")
        self.capture_timer = self.create_timer(CAPTURE_INTERVAL_S, self.capture_frame_cb)

    def _camera_read_loop(self):
        device = os.path.realpath(CAPTURE_DEVICE) if not CAPTURE_DEVICE.isdigit() else int(CAPTURE_DEVICE)
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2) if isinstance(device, str) else cv2.VideoCapture(device)
        if CAPTURE_MJPG:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

        if not cap.isOpened():
            self.get_logger().error(f"Could not open camera {device} for rotation capture")
            self.cam_running = False
            return

        while self.cam_running:
            ret, frame = cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.01)
        cap.release()

    def capture_frame_cb(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return
            frame = self.latest_frame.copy()

        ts = time.strftime("%Y_%m_%d___%H_%M_%S")
        idx = self.capture_count
        img_name = f"frame_{ts}_{idx:03d}.jpg"
        img_path = self.capture_session_dir / img_name
        if not cv2.imwrite(str(img_path), frame):
            return

        meta = {
            "image": img_name,
            "timestamp": ts,
            "index_in_session": idx,
            "resolution": frame.shape[:2],
            "session": self.capture_session_dir.name,
        }
        with open(self.capture_session_dir / f"frame_{ts}_{idx:03d}.json", "w") as f:
            json.dump(meta, f, indent=2)

        self.capture_count += 1

    def stop_camera_capture(self):
        self.manual_control.x = 0.0
        if self.capture_timer is not None:
            self.capture_timer.cancel()
            self.capture_timer = None
        self.cam_running = False
        if self.cam_thread is not None:
            self.cam_thread.join(timeout=2.0)
            self.cam_thread = None
        self.get_logger().info(
            f"Rotation scan done: {self.capture_count} frame(s) saved to {self.capture_session_dir}")

    def step_timer_cb(self):
        if self.step is None or self.step == STEP_DONE:
            return

        if self.comms_stale():
            self.get_logger().error(
                f"No state update in over {STATE_STALE_TIMEOUT_S:.1f}s (comms dropout?) -- "
                "stopping automated commands and holding. Confirm the link is back, then "
                "run land_and_disarm_example.py.")
            self.stop_camera_capture()
            self.step = STEP_DONE
            return

        if self.step == STEP_WAIT_ARM_CONFIRM:
            if self.armed:
                self.get_logger().info("Arm confirmed via RoosterState")
                self.step = STEP_TAKEOFF
                return
            if time.monotonic() - self.arm_wait_start_time > ARM_CONFIRM_TIMEOUT:
                self.get_logger().error(
                    f"Arm not confirmed after {ARM_CONFIRM_TIMEOUT}s -- stopping "
                    "(manual_control stream keeps running)")
                self.step = STEP_DONE

        elif self.step == STEP_TAKEOFF:
            self.get_logger().info(
                "Send MAV_CMD_NAV_TAKEOFF (PX4 uses MIS_TAKEOFF_ALT for target altitude)")
            nan = float('nan')
            self.send_command_long(MAV_CMD_NAV_TAKEOFF,
                                   0.0, 0.0, 0.0,
                                   nan, nan, nan, nan)
            self.takeoff_sent_time = time.monotonic()
            self.step = STEP_MONITOR_ALTITUDE

        elif self.step == STEP_MONITOR_ALTITUDE:
            elapsed = time.monotonic() - self.takeoff_sent_time
            h = self.height_above_arm()
            if int(elapsed * 2) != int((elapsed - 0.2) * 2):
                self.get_logger().info(
                    f"Monitoring takeoff... flight_mode={self.flight_mode}, height={h}")
            if h is not None and h >= SWITCH_ALTITUDE:
                self.get_logger().info(
                    f"Reached {SWITCH_ALTITUDE}m (height={h}), requesting Position mode")
                self.manual_control.z = MANUAL_CONTROL_NEUTRAL_Z
                self.request_flight_mode(KeepAlive.FLIGHT_MODE_POSITION)
                self.position_wait_start_time = time.monotonic()
                self.step = STEP_VERIFY_POSITION

        elif self.step == STEP_VERIFY_POSITION:
            if self.flight_mode == KeepAlive.FLIGHT_MODE_POSITION:
                self.get_logger().info(f"Position mode confirmed (flight_mode={self.flight_mode})")
                self.start_camera_capture()
                self.settle_start_time = time.monotonic()
                self.step = STEP_SETTLE_BEFORE_YAW
                return
            if time.monotonic() - self.position_wait_start_time > POSITION_RETRY_INTERVAL:
                self.get_logger().warn(
                    f"Position not confirmed yet (flight_mode={self.flight_mode}) -- re-requesting")
                self.request_flight_mode(KeepAlive.FLIGHT_MODE_POSITION)
                self.position_wait_start_time = time.monotonic()

        elif self.step == STEP_SETTLE_BEFORE_YAW:
            if time.monotonic() - self.settle_start_time >= YAW_SETTLE_TIME_S:
                self.get_logger().info(
                    f"Settled {YAW_SETTLE_TIME_S:.1f}s in Position mode -- starting the turn")
                self.step = STEP_YAW_360

        elif self.step == STEP_YAW_360:
            self.get_logger().info(
                f"Requesting {YAW_TURN_DEGREES:.0f} deg relative turn at {YAW_RATE_DEG_S} deg/s "
                f"(MAV_CMD_CONDITION_YAW)")
            self.send_command_long(MAV_CMD_CONDITION_YAW,
                                   YAW_TURN_DEGREES, YAW_RATE_DEG_S,
                                   YAW_DIRECTION, YAW_RELATIVE)
            self.yaw_accumulated_deg = 0.0
            self.yaw_last_azimuth = self.current_azimuth
            now = time.monotonic()
            self.yaw_wait_start_time = now
            self.yaw_last_resend_time = now
            self.step = STEP_WAIT_YAW_DONE

        elif self.step == STEP_WAIT_YAW_DONE:
            now = time.monotonic()
            done = abs(self.yaw_accumulated_deg) >= YAW_TURN_DEGREES - YAW_TOLERANCE_DEG
            timed_out = now - self.yaw_wait_start_time > YAW_MAX_WAIT_S

            if done or timed_out:
                reason = "azimuth feedback confirms" if done else \
                    f"safety timeout after {YAW_MAX_WAIT_S:.0f}s, giving up"
                self.get_logger().info(
                    f"Turn complete ({reason}: {self.yaw_accumulated_deg:.1f} deg accumulated) -- "
                    "hovering in Position mode. Run land_and_disarm_example.py to land.")
                self.stop_camera_capture()
                self.step = STEP_DONE
                return

            if now - self.yaw_last_resend_time > YAW_RESEND_INTERVAL_S:
                remaining = YAW_TURN_DEGREES - abs(self.yaw_accumulated_deg)
                self.get_logger().info(
                    f"Re-sending yaw for the remaining {remaining:.1f} deg "
                    f"({self.yaw_accumulated_deg:.1f} deg done so far)")
                self.send_command_long(MAV_CMD_CONDITION_YAW,
                                       remaining, YAW_RATE_DEG_S,
                                       YAW_DIRECTION, YAW_RELATIVE)
                self.yaw_last_resend_time = now

    def destroy_node(self):
        self.stop_camera_capture()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TakeoffAndRotate()
    print("Starting arm, takeoff and 360 rotate sequence...")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
