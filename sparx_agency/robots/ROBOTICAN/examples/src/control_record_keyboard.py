import rclpy
from rclpy.node import Node
import sys
import select
import termios
import tty
import csv
import time
import threading
from datetime import datetime

# Import custom message definitions as per ICD Section 8
from fcu_driver_interfaces.msg import ManualControl
from rooster_handler_interfaces.msg import KeepAlive
from rooster_manager_interfaces.msg import RoosterState

# Import ROS service type for arming
from std_srvs.srv import SetBool

# Constants from ICD Section 4.1.2
FLIGHT_MODE_GROUND_ROLL = 1
FLIGHT_MODE_MANUAL = 2


class KeyboardHandler:
    """Handles raw keyboard input in a non-blocking way."""

    def __init__(self):
        self.settings = termios.tcgetattr(sys.stdin)

    def get_key(self):
        """Reads a single character from stdin without requiring enter."""
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key


class FlightRecorder:
    """Handles logging of commands and telemetry to CSV."""

    def __init__(self, filename):
        self.filename = filename
        self.headers = [
            'timestamp',
            'cmd_x', 'cmd_y', 'cmd_z', 'cmd_r',
            'state_roll', 'state_pitch', 'state_azimuth',
            'flight_mode', 'battery', 'is_armed'
        ]
        self._init_csv()

    def _init_csv(self):
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(self.headers)

    def log(self, cmd_msg, state_msg):
        """Writes a synchronized row of command and state data."""
        if state_msg is None:
            # Cannot log without state data
            return

        with open(self.filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            row = [
                time.time(),
                cmd_msg.x, cmd_msg.y, cmd_msg.z, cmd_msg.r,
                state_msg.roll,
                state_msg.pitch,
                state_msg.azimuth,
                state_msg.flight_mode,
                getattr(state_msg, 'battery_voltage', 0.0),  # Assuming field might be named battery_voltage
                state_msg.armed  # Log if drone is armed
            ]
            writer.writerow(row)


class RoosterController(Node):
    """Main ROS 2 Node controlling the Robotican Rooster."""

    def __init__(self):
        super().__init__('rooster_controller')

        # --- Parameters ---
        self.declare_parameter('rooster_id', 'R1')
        self.rooster_id = self.get_parameter('rooster_id').get_parameter_value().string_value

        # --- State Management ---
        self.axes = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'r': 0.0}  # Z=-1000 is 0 thrust/power
        self.increment = 10.0
        self.latest_telemetry = None
        self.is_armed = False

        # --- Components ---
        self.input_handler = KeyboardHandler()
        self.recorder = FlightRecorder(f'rooster_log_{datetime.now().strftime("%H%M%S")}.csv')

        # --- ROS Communication ---
        self._init_communication()

        self.get_logger().info(f"Controller Initialized for {self.rooster_id} in GROUND ROLL Mode.")
        self.get_logger().info("Press 'F' to try and force arm the drone.")
        self.print_instructions()

    def _init_communication(self):
        # 1. Manual Control Publisher (40Hz)
        self.pub_manual = self.create_publisher(
            ManualControl,
            f'/{self.rooster_id}/manual_control',
            10
        )
        self.timer_control = self.create_timer(1.0 / 40.0, self.control_loop)

        # 2. Keep Alive Publisher (1Hz)
        self.pub_keepalive = self.create_publisher(
            KeepAlive,
            f'/{self.rooster_id}/keep_alive',
            10
        )
        self.timer_keepalive = self.create_timer(1.0, self.keep_alive_loop)

        # 3. State Subscriber
        self.sub_state = self.create_subscription(
            RoosterState,
            f'/{self.rooster_id}/state',
            self.telemetry_callback,
            10
        )

        # 4. Force Arm Service Client
        self.arm_client = self.create_client(
            SetBool,
            f'/{self.rooster_id}/fcu/command/force_arm'
        )

    def telemetry_callback(self, msg):
        self.latest_telemetry = msg
        self.is_armed = msg.armed

    def keep_alive_loop(self):
        """Sends KeepAlive to maintain connection and set mode."""
        msg = KeepAlive()
        msg.is_active = True
        # Setting mode to GROUND_ROLL (1) as requested
        msg.requested_flight_mode = FLIGHT_MODE_GROUND_ROLL
        msg.command_reboot = False
        self.pub_keepalive.publish(msg)

    def force_arm_drone(self, arm_state: bool):
        """Asynchronously calls the force_arm service."""
        self.get_logger().info(f"Attempting to {'ARM' if arm_state else 'DISARM'}...")

        if not self.arm_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Force Arm service not available. Check FCU connection.")
            return

        req = SetBool.Request()
        req.data = arm_state
        future = self.arm_client.call_async(req)

        def arm_callback(future):
            try:
                result = future.result()
            except Exception as e:
                self.get_logger().error(f"Arm service call failed: {e}")
                return

            if result.success:
                self.get_logger().info(f"Successfully {'ARMED' if arm_state else 'DISARMED'}.")
                # Note: self.is_armed should be updated via the telemetry_callback (RoosterState)
            else:
                self.get_logger().error(f"Arming failed: {result.message}")

        future.add_done_callback(arm_callback)

    def update_axes(self, key):
        """Updates virtual joystick axes based on key press."""
        # Check for arm/disarm command first
        if key == 'f':
            # Toggle arm state (Disarm if already armed, otherwise Arm)
            self.force_arm_drone(not self.is_armed)
            return

        if not self.is_armed:
            self.get_logger().warn("\rDrone is NOT ARMED. Press 'F' to attempt arming.")
            # Only allow control if armed, otherwise controls are ignored
            return

        # Movement Controls (only active if armed)
        if key == 'e':
            self.axes['x'] = min(self.axes['x'] + self.increment, 1000.0)
        elif key == 'x':
            self.axes['x'] = max(self.axes['x'] - self.increment, -1000.0)

        elif key == 'd':
            self.axes['r'] = min(self.axes['r'] + self.increment, 1000.0)
        elif key == 'a':
            self.axes['r'] = max(self.axes['r'] - self.increment, -1000.0)

        elif key == 'w':
            self.axes['z'] = min(self.axes['z'] + self.increment, 1000.0)
        elif key == 's':
            self.axes['z'] = max(self.axes['z'] - self.increment, -1000.0)

        elif key == 'l':
            self.axes['y'] = min(self.axes['y'] + self.increment, 1000.0)
        elif key == 'j':
            self.axes['y'] = max(self.axes['y'] - self.increment, -1000.0)

        elif key == ' ':
            self.axes['x'] = 0.0
            self.axes['y'] = 0.0
            self.axes['z'] = 0.0
            self.axes['r'] = 0.0
            self.get_logger().info("\rAxes reset to neutral.")

    def control_loop(self):
        """The main 40Hz loop: Publish Command -> Record Data."""
        # Check for input
        key = self.input_handler.get_key()
        if key == 'q':
            raise KeyboardInterrupt

        # Update axes, which handles arming/disarming and movement
        self.update_axes(key)
        self.print_status()

        # Construct Message (always send commands to prevent failsafe, even if zero)
        msg = ManualControl()
        msg.x = float(self.axes['x'])
        msg.y = float(self.axes['y'])
        msg.z = float(self.axes['z'])
        msg.r = float(self.axes['r'])
        msg.buttons = 0

        # Publish
        self.pub_manual.publish(msg)

        # Record
        self.recorder.log(msg, self.latest_telemetry)

    def print_status(self):
        """Prints current axes and arming status to stdout."""
        arm_status = "ARMED" if self.is_armed else "DISARMED"
        status = (f"\r[{arm_status}] X: {self.axes['x']:>5.0f} | "
                  f"Y: {self.axes['y']:>5.0f} | "
                  f"Z: {self.axes['z']:>5.0f} | "
                  f"R: {self.axes['r']:>5.0f}    ")
        sys.stdout.write(status)
        sys.stdout.flush()

    def print_instructions(self):
        print("\n--- ROOSTER GROUND ROLL CONTROLLER V2 ---")
        print("F     : TOGGLE ARM/DISARM (Uses force_arm service)")
        print("E/X   : Forward/Backward (Pitch)")
        print("A/D   : Yaw Left/Right (Yaw)")
        print("W/S   : Throttle Up/Down (Z)")
        print("L/J   : Roll Left/Right (Y)")
        print("SPACE : Reset Neutral (X, Y, R)")
        print("Q     : Quit")
        print("------------------------------------------\n")


def main(args=None):
    rclpy.init(args=args)

    try:
        controller = RoosterController()
        # Spin allows the timers (Control 40Hz, KeepAlive 1Hz) to fire
        # The control_loop method handles keyboard input in a non-blocking way
        rclpy.spin(controller)
    except KeyboardInterrupt:
        print("\nShutting down Rooster Controller...")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()