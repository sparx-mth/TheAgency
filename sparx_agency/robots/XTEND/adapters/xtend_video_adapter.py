from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import threading
import time

import numpy as np

from sparx_agency.core.common.types import Intrinsics, Observation
from sparx_agency.robots.common.perception_converter import rgbframe_from_bgr,observation_from_rgb

@dataclass(frozen=True)
class XtendVideoConfig:
    rtsp_uri: str = "rtsp://192.0.0.15:8556/osd_snapshot"
    latency_ms: int = 0
    frame_id: str = "xtend_camera"

class XtendVideoAdapter:
    def __init__(self, probe, intrinsics: Optional[Intrinsics] = None, frame_id: str = "xtend_camera"):
        self.rtsp = probe
        self.intrinsics = intrinsics
        self.frame_id = frame_id

    def start(self) -> None:
        self.rtsp.start()

    def stop(self) -> None:
        self.rtsp.stop()

    def get_latest_rgb(self):
        out = self.rtsp.get_latest()
        if out is None:
            return None
        bgr, stamp = out
        return rgbframe_from_bgr(bgr, stamp_sec=stamp, frame_id=self.frame_id)

    def get_latest_observation(self, pose_map_base=None) -> Optional[Observation]:
        rgb = self.get_latest_rgb()
        if rgb is None:
            return None
        return observation_from_rgb(rgb, intr=self.intrinsics, pose_map_base=pose_map_base)
