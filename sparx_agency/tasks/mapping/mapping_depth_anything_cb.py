import numpy as np
import math
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from fcu_driver_interfaces.msg import UAVState


class _Intrinsics:
    def __init__(self, fx: float, fy: float, cx: float, cy: float, width: int, height: int):
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.width = int(width)
        self.height = int(height)


class _Rgb:
    def __init__(self, image: np.ndarray):
        self.image = image


class _Observation:
    def __init__(self, rgb, intrinsics, pose_map_base=None, depth=None, cloud=None):
        self.rgb = rgb
        self.intrinsics = intrinsics
        self.pose_map_base = pose_map_base
        self.depth = depth
        self.cloud = cloud


def _ros_image_to_rgb_np(msg: Image) -> np.ndarray:
    h, w = int(msg.height), int(msg.width)
    enc = (msg.encoding or "").lower()
    data = msg.data

    if enc in ("rgb8", "bgr8"):
        arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
        return arr if enc == "rgb8" else arr[:, :, ::-1]

    if enc in ("rgba8", "bgra8"):
        arr = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
        return arr if enc == "rgba8" else arr[:, :, ::-1]

    if enc == "mono8":
        gray = np.frombuffer(data, dtype=np.uint8).reshape((h, w))
        return np.stack([gray, gray, gray], axis=-1)

    raise ValueError(f"Unsupported Image encoding: {msg.encoding}")


def intrinsics_from_fov(width: int, height: int, hfov_deg: float, vfov_deg: float) -> _Intrinsics:
    hfov = math.radians(float(hfov_deg))
    vfov = math.radians(float(vfov_deg))

    fx = (width / 2.0) / math.tan(hfov / 2.0)
    fy = (height / 2.0) / math.tan(vfov / 2.0)

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return _Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


class MappingTask(Node):
    def __init__(
        self,
        pipeline,
        drone_id: str = "R1",
        process_period_sec: float = 1.5,
        width: int = 640,
        height: int = 360,
        hfov_deg: float = 130.0,
        vfov_deg: float = 90.0,
    ):
        super().__init__("mapping_task")
        self.pipeline = pipeline
        self.drone_id = drone_id

        self.intr = intrinsics_from_fov(width, height, hfov_deg, vfov_deg)

        self.last_img: Image | None = None
        self.last_state: UAVState | None = None
        self.last_processed_stamp = None  # (sec, nanosec)

        self.create_subscription(Image, f"/{drone_id}/camera/image_raw", self.image_cb, 10)
        self.create_subscription(UAVState, f"/{drone_id}/fcu/state", self.state_cb, 10)

        self.create_timer(float(process_period_sec), self._tick)

        self.get_logger().info(
            f"Bag mode (no CameraInfo): W={width} H={height} HFOV={hfov_deg} VFOV={vfov_deg} "
            f"-> fx={self.intr.fx:.1f} fy={self.intr.fy:.1f} cx={self.intr.cx:.1f} cy={self.intr.cy:.1f}. "
            f"Processing every {process_period_sec:.2f}s"
        )

    def image_cb(self, msg: Image):
        self.last_img = msg

    def state_cb(self, msg: UAVState):
        self.last_state = msg

    def _tick(self):
        if self.last_img is None:
            return

        st = self.last_img.header.stamp
        stamp = (int(st.sec), int(st.nanosec))
        if self.last_processed_stamp == stamp:
            return

        try:
            rgb = _ros_image_to_rgb_np(self.last_img)

            obs = _Observation(
                rgb=_Rgb(rgb),
                intrinsics=self.intr,
                pose_map_base=None,  # bag mode: local update
                depth=None,
                cloud=None,
            )

            self.pipeline.step(obs)
            self.last_processed_stamp = stamp

        except Exception as e:
            self.get_logger().error(f"pipeline.step failed: {e}")
