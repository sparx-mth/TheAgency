from sparx_agency.robots.common.adapters.perception_converter import (
    rgbframe_from_bgr,
    observation_from_rgb,
)

class XtendVideoAdapter:
    def __init__(self, rtsp_probe, intrinsics=None, frame_id="xtend_camera"):
        self.rtsp = rtsp_probe
        self.intrinsics = intrinsics
        self.frame_id = frame_id

    def get_latest_rgb(self):
        out = self.rtsp.get_latest()
        if out is None:
            return None
        bgr, stamp = out
        return rgbframe_from_bgr(bgr, stamp_sec=stamp, frame_id=self.frame_id)

    def get_latest_observation(self, pose_map_base=None):
        rgb = self.get_latest_rgb()
        if rgb is None:
            return None
        return observation_from_rgb(rgb, intr=self.intrinsics, pose_map_base=pose_map_base)
