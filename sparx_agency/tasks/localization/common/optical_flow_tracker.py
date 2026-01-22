from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2


@dataclass
class FlowResult:
    good_old: np.ndarray  # [N,2] previous pixel locations (x,y)
    good_new: np.ndarray  # [N,2] current pixel locations (x,y)
    dt: float              # time difference between frames [seconds]
    n_used: int            # number of tracked points


class OpticalFlowTracker:
    """
    Optical Flow tracking module (Lucas–Kanade).

    Externally:
      - Stateless API: call process(frame, stamp) and get flow results.

    Internally:
      - Keeps previous frame, tracked points, and timestamp.

    Responsibilities:
      - Convert RGB → grayscale
      - Detect feature points (corners)
      - Track them using LK optical flow
      - Compute dt from ROS timestamps
    """

    def __init__(
        self,
        max_corners: int = 300,
        min_corners: int = 30,
        lk_win: int = 21,
        lk_levels: int = 3,
        quality_level: float = 0.01,
        min_distance: int = 7,
        block_size: int = 7,
    ):
        self.max_corners = int(max_corners)
        self.min_corners = int(min_corners)

        self.quality_level = float(quality_level)
        self.min_distance = int(min_distance)
        self.block_size = int(block_size)

        self.lk_params = dict(
            winSize=(int(lk_win), int(lk_win)),
            maxLevel=int(lk_levels),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None  # [N,1,2]
        self.prev_stamp = None  # expects ROS-like stamp with sec,nanosec

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_pts = None
        self.prev_stamp = None

    @staticmethod
    def _stamp_to_ns(stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _detect_features(self, gray: np.ndarray) -> Optional[np.ndarray]:
        # Shi-Tomsi Algorithm for Feature Detection
        pts = cv2.goodFeaturesToTrack( 
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
        )
        return pts

    def process(self, frame_bgr: np.ndarray, stamp) -> Optional[FlowResult]:
        """
        Input:
          frame_bgr: HxWx3 uint8
          stamp: msg.header.stamp (sec,nanosec)

        Output:
          FlowResult or None if not enough info yet.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # init
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = self._detect_features(gray)
            self.prev_stamp = stamp
            return None

        # dt
        dt_ns = self._stamp_to_ns(stamp) - self._stamp_to_ns(self.prev_stamp)
        dt = float(dt_ns) * 1e-9
        if dt <= 0.0:
            self.prev_gray = gray
            self.prev_stamp = stamp
            return None

        # refresh features if too few
        if self.prev_pts is None or len(self.prev_pts) < self.min_corners:
            self.prev_pts = self._detect_features(self.prev_gray)
            if self.prev_pts is None:
                self.prev_gray = gray
                self.prev_stamp = stamp
                return None

        # LK -Lucas Kanade algorithm 
        next_pts, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )
        if next_pts is None or st is None:
            self.prev_gray = gray
            self.prev_pts = None
            self.prev_stamp = stamp
            return None

        
        good_new = next_pts[st == 1] #good_new= the new locations (x,y) of the points in the current frame
        good_old = self.prev_pts[st == 1] #good_old= the orignal locations of the same points in the previous frame

        if len(good_new) == 0:
            self.prev_gray = gray
            self.prev_pts = None
            self.prev_stamp = stamp
            return None

        # update internal state for next call
        self.prev_gray = gray
        self.prev_pts = good_new.reshape(-1, 1, 2).astype(np.float32)
        self.prev_stamp = stamp

        return FlowResult(
            good_old=good_old.reshape(-1, 2).astype(np.float32),
            good_new=good_new.reshape(-1, 2).astype(np.float32),
            dt=dt, 
            n_used=int(len(good_new)), #how many points we succeed to use
        )
