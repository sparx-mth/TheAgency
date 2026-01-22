import numpy as np
from sparx_agency.core.common.types.perception import RGBFrame, Observation, Intrinsics

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
