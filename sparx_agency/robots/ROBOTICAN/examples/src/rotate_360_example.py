#!/usr/bin/python3

import os
import time
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

# Rotation: no heading telemetry is available here, so completion is time-based -
# the duration is derived from the commanded rate, plus a margin. If it visibly
# over/under-rotates on a real flight, tune YAW_RATE_DEG_S, not the logic.
YAW_TURN_DEGREES = 360.0
YAW_RATE_DEG_S = 30.0      # slow/conservative default - tune to taste
YAW_DIRECTION = 1.0        # 1 = clockwise, -1 = counter-clockwise
YAW_RELATIVE = 1.0         # 1 = relative to current heading, 0 = absolute
YAW_COMPLETE_MARGIN_S = 2.0

# Scan capture: runs from Position-mode confirmation through the end of the turn, at the
# same device, resolution and MJPG format as auto_capture_interval.py in NanoLLM_VILA_and_OWL
# (confirmed working via OpenCV inside this container: 640x360 MJPG open+read succeeds -
# camera_frames_example.py's H264 caps-negotiation issue in this same workspace is specific
# to its hardware H264 decode path, not plain MJPG capture). Writes into this board's own
# captures root under its own "rotate_scan_*" subfolder, so comm_manager_vllm.py can process
# these frames too, without colliding with that pipeline's "latest"-symlinked folder rotation.
CAPTURE_DEVICE = "/dev/HD_CAMERA"
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 360
CAPTURE_MJPG = True
CAPTURE_INTERVAL_S = 0.5
CAPTURES_ROOT = "/home/iai/captures"

STEP_WAIT_ARM_CONFIRM = 0
STEP_TAKEOFF = 1
STEP_MONITOR_ALTITUDE = 2
STEP_VERIFY_POSITION = 3
STEP_YAW_360 = 4
STEP_WAIT_YAW_DONE = 5
STEP_DONE = 6


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
        self.yaw_sent_time = None
        self.yaw_duration_s = None
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
        if self.armed != msg.armed or self.flight_mode != msg.flight_mode:
            self.armed = msg.armed
            self.flight_mode = msg.flight_mode
            self.get_logger().info(f"RoosterState armed={msg.armed} flight_mode={msg.flight_mode}")

    def uav_state_cb(self, msg):
        self.current_altitude = msg.altitude_relative

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

    # ---------- camera scan capture ----------
    def start_camera_capture(self):
        if self.cam_thread is not None:
            return
        self.cam_running = True
        self.cam_thread = threading.Thread(target=self._camera_read_loop, daemon=True)
        self.cam_thread.start()

        root = Path(CAPTURES_ROOT) / self.id
        ts = time.strftime("%Y_%m_%d___%H_%M_%S")
        self.capture_session_dir = root / f"rotate_scan_{ts}"
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
        img_path = self.capture_session_dir / f"frame_{self.capture_count:03d}.jpg"
        if cv2.imwrite(str(img_path), frame):
            self.capture_count += 1

    def stop_camera_capture(self):
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
        if self.step is None:
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
                self.step = STEP_YAW_360
                return
            if time.monotonic() - self.position_wait_start_time > POSITION_RETRY_INTERVAL:
                self.get_logger().warn(
                    f"Position not confirmed yet (flight_mode={self.flight_mode}) -- re-requesting")
                self.request_flight_mode(KeepAlive.FLIGHT_MODE_POSITION)
                self.position_wait_start_time = time.monotonic()

        elif self.step == STEP_YAW_360:
            self.get_logger().info(
                f"Requesting {YAW_TURN_DEGREES:.0f} deg relative turn at {YAW_RATE_DEG_S} deg/s "
                f"(MAV_CMD_CONDITION_YAW)")
            self.send_command_long(MAV_CMD_CONDITION_YAW,
                                   YAW_TURN_DEGREES, YAW_RATE_DEG_S,
                                   YAW_DIRECTION, YAW_RELATIVE)
            self.yaw_sent_time = time.monotonic()
            self.yaw_duration_s = YAW_TURN_DEGREES / YAW_RATE_DEG_S + YAW_COMPLETE_MARGIN_S
            self.step = STEP_WAIT_YAW_DONE

        elif self.step == STEP_WAIT_YAW_DONE:
            elapsed = time.monotonic() - self.yaw_sent_time
            if elapsed >= self.yaw_duration_s:
                self.get_logger().info(
                    "Turn complete (time-based, no heading feedback available) -- "
                    "hovering in Position mode. Run land_and_disarm_example.py to land.")
                self.stop_camera_capture()
                self.step = STEP_DONE

        elif self.step == STEP_DONE:
            pass

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
