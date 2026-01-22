#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Range


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class AltitudeMux(Node):
    """
    - In nav:   /simple_drone/cmd_vel_nav  (Twist)  [from NoMaD]
    - In sonar: /simple_drone/sonar/out    (Range)
    - Out cmd:  /simple_drone/cmd_vel      (Twist)  [to drone]
    """

    def __init__(self):
        super().__init__("altitude_mux")

        # ---------- Params ----------
        self.declare_parameter("z_ref", 0.5)
        self.declare_parameter("kp", 1.2)
        self.declare_parameter("kd", 0.8)

        self.declare_parameter("u_min", -0.8)
        self.declare_parameter("u_max", 0.8)
        self.declare_parameter("max_du_per_sec", 1.5)

        self.declare_parameter("control_rate_hz", 30.0)
        self.declare_parameter("nav_timeout_sec", 0.6)

        self.declare_parameter("vz_lpf_alpha", 0.35)
        self.declare_parameter("auto_setpoint_on_start", False)

        # Range gating (keep permissive to avoid dropping valid sonar)
        self.declare_parameter("range_valid_min", 0.02)
        self.declare_parameter("range_valid_max", 20.0)

        # If True: when sonar missing, still publish nav passthrough
        self.declare_parameter("passthrough_without_sonar", True)

        self.z_ref = float(self.get_parameter("z_ref").value)
        self.kp = float(self.get_parameter("kp").value)
        self.kd = float(self.get_parameter("kd").value)

        self.u_min = float(self.get_parameter("u_min").value)
        self.u_max = float(self.get_parameter("u_max").value)
        self.max_du_per_sec = float(self.get_parameter("max_du_per_sec").value)

        self.rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.nav_timeout = float(self.get_parameter("nav_timeout_sec").value)

        self.vz_alpha = float(self.get_parameter("vz_lpf_alpha").value)
        self.auto_setpoint = bool(self.get_parameter("auto_setpoint_on_start").value)

        self.rmin = float(self.get_parameter("range_valid_min").value)
        self.rmax = float(self.get_parameter("range_valid_max").value)
        self.passthrough_without_sonar = bool(self.get_parameter("passthrough_without_sonar").value)

        # ---------- QoS (BEST_EFFORT everywhere to avoid mismatch) ----------
        qos_be = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ---------- State ----------
        self.have_nav = False
        self.last_nav_cmd = Twist()
        self.last_nav_t = 0.0

        self.have_range = False
        self.last_z = None
        self.last_t = None
        self.vz_filt = 0.0

        self.u_prev = 0.0
        self._last_log_t = 0.0

        # ---------- ROS I/O ----------
        self.create_subscription(Twist, "/simple_drone/cmd_vel_nav", self.on_nav, qos_be)
        self.create_subscription(Range, "/simple_drone/sonar/out", self.on_range, qos_be)
        self.pub = self.create_publisher(Twist, "/simple_drone/cmd_vel", qos_be)

        dt = 1.0 / max(self.rate_hz, 1.0)
        self.timer = self.create_timer(dt, self.step)

        self.get_logger().info("AltitudeMux started.")
        self.get_logger().info("In:  /simple_drone/cmd_vel_nav  +  /simple_drone/sonar/out")
        self.get_logger().info("Out: /simple_drone/cmd_vel  (BEST_EFFORT)")
        self.get_logger().info(
            f"z_ref={self.z_ref:.2f}, kp={self.kp}, kd={self.kd}, u=[{self.u_min},{self.u_max}], "
            f"rate={self.rate_hz}Hz, nav_timeout={self.nav_timeout}s"
        )

    def on_nav(self, msg: Twist):
        self.last_nav_cmd = msg
        self.last_nav_t = time.time()
        self.have_nav = True

    def on_range(self, msg: Range):
        z = float(msg.range)

        # permissive validity checks
        if z < self.rmin or z > self.rmax:
            return
        if msg.min_range > 0.0 and z < float(msg.min_range):
            return
        if msg.max_range > 0.0 and z > float(msg.max_range) + 1e-3:
            return

        t = time.time()

        if self.auto_setpoint and not self.have_range:
            self.z_ref = z
            self.get_logger().info(f"Auto setpoint: z_ref={self.z_ref:.2f}m")

        if self.last_z is not None and self.last_t is not None:
            dt = max(t - self.last_t, 1e-4)
            vz = (z - self.last_z) / dt
            self.vz_filt = self.vz_alpha * vz + (1.0 - self.vz_alpha) * self.vz_filt

        self.last_z = z
        self.last_t = t
        self.have_range = True

    def nav_fresh(self) -> bool:
        if not self.have_nav:
            return False
        return (time.time() - self.last_nav_t) <= self.nav_timeout

    def step(self):
        # Build base command from nav (or zero if missing/stale)
        nav_ok = self.nav_fresh()
        base = self.last_nav_cmd if nav_ok else Twist()

        out = Twist()
        out.linear.x = float(base.linear.x)
        out.linear.y = float(base.linear.y)
        out.angular.z = float(base.angular.z)

        # If no sonar yet: publish passthrough (so you'll SEE messages on /cmd_vel)
        if not self.have_range or self.last_z is None:
            if self.passthrough_without_sonar:
                out.linear.z = 0.0
                self.pub.publish(out)
            self._log_periodic(nav_ok, sonar_ok=False, z=None, vz=None, cmd_z=out.linear.z)
            return

        # Altitude PD
        z = float(self.last_z)
        vz = float(self.vz_filt)

        e = self.z_ref - z
        de = -vz

        u = self.kp * e + self.kd * de
        u = clamp(u, self.u_min, self.u_max)

        # rate limit
        dt = 1.0 / max(self.rate_hz, 1.0)
        max_du = self.max_du_per_sec * dt
        u = clamp(u, self.u_prev - max_du, self.u_prev + max_du)
        self.u_prev = u

        out.linear.z = float(u)
        self.pub.publish(out)

        self._log_periodic(nav_ok, sonar_ok=True, z=z, vz=vz, cmd_z=out.linear.z)

    def _log_periodic(self, nav_ok: bool, sonar_ok: bool, z, vz, cmd_z: float):
        now = time.time()
        if now - self._last_log_t < 1.0:
            return
        if sonar_ok:
            self.get_logger().info(
                f"nav_ok={nav_ok} sonar_ok={sonar_ok} z={z:.2f} z_ref={self.z_ref:.2f} vz={vz:.2f} cmd_z={cmd_z:.3f}"
            )
        else:
            self.get_logger().info(
                f"nav_ok={nav_ok} sonar_ok={sonar_ok} (waiting sonar) cmd_z={cmd_z:.3f}"
            )
        self._last_log_t = now


def main():
    rclpy.init()
    node = AltitudeMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
