# sparx_agency/core/localization/tag_azimuth_estimator.py
from __future__ import annotations

import math
from collections import deque
from typing import Dict, Optional, Tuple

from .types.tag_azimuth import TagBearingObservation


class TagAzimuthEstimator:
    """
    Core (ROS-agnostic) estimator:
    Computes camera azimuth (0..360 deg) in world frame using known wall-azimuth per tag.
    Keeps a time history for lookup at (approximately) requested timestamps.

    Inputs:
      - tag_config_deg: mapping tag_id -> wall_azimuth_deg (clockwise from North=0)
      - observations: list of TagBearingObservation in camera frame

    Output:
      - yaw_deg: camera absolute azimuth (0..360)
      - best_tag_id: which tag was used (best centered)
    """

    def __init__(
        self,
        tag_config_deg: Dict[int, float],
        max_history: int = 20,
        max_time_diff_sec: float = 1.0,
    ):
        if not tag_config_deg:
            raise ValueError("tag_config_deg must not be empty")

        self.tag_config_deg = dict(tag_config_deg)
        self.known_tag_ids = list(self.tag_config_deg.keys())

        self._history: deque[Tuple[float, float]] = deque(maxlen=max_history)
        self.max_time_diff_sec = float(max_time_diff_sec)

    @staticmethod
    def _normalize_0_360(deg: float) -> float:
        deg = deg % 360.0
        return deg if deg >= 0.0 else deg + 360.0

    @staticmethod
    def relative_yaw_deg(tx: float, tz: float) -> float:
        """
        Relative yaw (deg) of the tag from camera center.
        """
        return math.degrees(math.atan2(-tx, tz))
    
    @staticmethod
    def obs_from_tvec(tag_id: int, tvec) -> TagBearingObservation:
        """
        Convenience helper for adapters that use solvePnP:
        OpenCV returns tvec in camera frame: [tx, ty, tz].
        We only need tx and tz for azimuth.
        """
        tx = float(tvec[0])
        tz = float(tvec[2])
        return TagBearingObservation(tag_id=tag_id, tx=tx, tz=tz)


    def estimate_from_observations(
        self, observations: list[TagBearingObservation]
    ) -> Optional[Tuple[float, int]]:
        """
        Pick the best tag (closest to center => minimal abs(relative_yaw)),
        and compute absolute camera azimuth.

        Returns:
          (camera_yaw_deg, best_tag_id) or None if no usable observations.
        """
        best = None  # (abs_rel_yaw, camera_yaw, tag_id)

        for obs in observations:
            if obs.tag_id not in self.tag_config_deg:
                continue

            # Guard against invalid geometry
            if obs.tz == 0.0 and obs.tx == 0.0:
                continue

            rel_deg = self.relative_yaw_deg(obs.tx, obs.tz)
            abs_rel = abs(rel_deg)

            wall_azimuth = self.tag_config_deg[obs.tag_id]
            camera_yaw = self._normalize_0_360(wall_azimuth + rel_deg)

            if best is None or abs_rel < best[0]:
                best = (abs_rel, camera_yaw, obs.tag_id)

        if best is None:
            return None

        _, yaw_deg, tag_id = best
        return yaw_deg, tag_id

    def update(
        self, observations: list[TagBearingObservation], stamp_sec: float
    ) -> Optional[Tuple[float, int]]:
        """
        Estimate from observations and store in history.

        stamp_sec: any monotonic-ish timestamp in seconds (e.g., ROS time converted to float)
        """
        result = self.estimate_from_observations(observations)
        if result is None:
            return None

        yaw_deg, tag_id = result
        self._history.append((float(stamp_sec), float(yaw_deg)))
        return yaw_deg, tag_id

    def get_at_time(self, request_stamp_sec: float) -> Optional[Tuple[float, float]]:
        """
        Returns (yaw_deg, dt_sec) where dt_sec = sample_time - request_time,
        using the closest stored sample, if within max_time_diff_sec.
        """
        if not self._history:
            return None

        req = float(request_stamp_sec)

        closest = min(self._history, key=lambda s: abs(s[0] - req))
        sample_t, sample_yaw = closest

        dt = sample_t - req
        if abs(dt) > self.max_time_diff_sec:
            return None

        return sample_yaw, dt
