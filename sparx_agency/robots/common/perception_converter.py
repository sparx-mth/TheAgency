import numpy as np
from sensor_msgs.msg import Image

from sparx_agency.core.common.types.perception import RGBFrame, Observation, Intrinsics
from sparx_agency.robots.common import stamp_to_sec


def rgbframe_from_bgr(bgr: np.ndarray, stamp_sec: float, frame_id: str = "") -> RGBFrame:
    # OpenCV/GStreamer usually gives BGR
    rgb = bgr[..., ::-1].copy()
    return RGBFrame(image=rgb, stamp_sec=stamp_sec, frame_id=frame_id)

def observation_from_rgb(
    rgb_frame: RGBFrame,
    intr: Intrinsics | None = None,
    pose_map_base=None,
) -> Observation:
    return Observation(intrinsics=intr, pose_map_base=pose_map_base, rgb=rgb_frame)


def bgr_from_ros_image(msg: Image) -> np.ndarray:
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    enc = (msg.encoding or "").lower()

    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if step == w * 3:
        img = buf.reshape((h, w, 3))
    else:
        img = buf.reshape((h, step))[:, : w * 3].reshape((h, w, 3))

    if enc == "bgr8":
        return img.copy()
    if enc == "rgb8":
        return img[:, :, ::-1].copy()  # rgb->bgr
    raise ValueError(f"Unsupported encoding: {msg.encoding}")


def rgbframe_from_ros_image(msg: Image, frame_id: str) -> RGBFrame:
    bgr = bgr_from_ros_image(msg)
    rgb = bgr[:, :, ::-1]
    return RGBFrame(image=rgb.astype(np.uint8), stamp_sec=stamp_to_sec(msg.header.stamp), frame_id=frame_id)
#
