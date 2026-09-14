"""Confirmed YOLO doorway landmarks from observed RGB-D, never surveyed doors."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from sparx_agency.core.common.math.bbox import iou
from sparx_agency.core.mapping.objects.geometry import robust_bbox_depth
from sparx_agency.core.mapping.objects.landmarks import ObjectLandmarkMap
from sparx_agency.core.planning.objnav.camera_geometry import world_T_camera_optical


DOOR_LABELS = frozenset(("door", "doorway", "open doorway", "door frame"))


@dataclass(frozen=True)
class DoorSettings:
    # Candidate threshold only. Targets retain their separate 0.35 gate.
    confidence: float = 0.05
    min_observations: int = 3
    dedupe_radius_m: float = 0.6
    min_view_translation_m: float = 0.2
    min_view_turn_deg: float = 20.0
    min_width_m: float = 0.35
    max_width_m: float = 1.5
    min_top_height_m: float = 1.4
    max_bottom_height_m: float = 0.5


def doorway_position(observation, detection, settings):
    """Estimate the doorway plane from its jamb/lintel bands, not its open centre.

    The centre of an open doorway usually measures the next room's far wall.
    Use the nearest robust frame-band depth to project the box centre onto the
    frame plane. This remains an RGB-box approximation, not a fitted door plane;
    width checks and repeated, separated views guard against single-box cuts.
    """
    x1, y1, x2, y2 = detection.xyxy
    k, camera = observation.camera.intrinsics, observation.camera
    width, height = x2 - x1, y2 - y1
    if not (0 <= x1 < x2 <= k.width and 0 <= y1 < y2 <= k.height):
        return None
    if width < 12 or height < 24 or height < 0.9 * width:
        return None
    bands = ((x1, y1 + 0.1 * height, x1 + 0.18 * width, y2 - 0.1 * height),
             (x2 - 0.18 * width, y1 + 0.1 * height, x2, y2 - 0.1 * height),
             (x1, y1, x2, y1 + 0.18 * height))
    depths = [robust_bbox_depth(observation.depth_m, box, camera.min_depth_m,
                               camera.max_depth_m, shrink=1.0, percentile=20.0,
                               min_valid_px=12) for box in bands]
    depths = [depth for depth in depths if depth is not None]
    if len(depths) < 2:
        return None
    depth = min(depths)
    apparent_width = width * depth / k.fx
    if not settings.min_width_m <= apparent_width <= settings.max_width_m:
        return None
    transform = world_T_camera_optical(observation.pose, camera)
    height_points = np.array([[(x1 + x2) / 2 - k.cx] * 2,
                              [y1 - k.cy, y2 - k.cy], [depth, depth], [1.0, 1.0]])
    height_points[0] *= depth / k.fx
    height_points[1] *= depth / k.fy
    top, bottom = (transform @ height_points)[2] - observation.pose.z
    if ((y1 > 2 and top < settings.min_top_height_m)
            or (y2 < k.height - 2 and abs(bottom) > settings.max_bottom_height_m)):
        return None
    point = np.array([((x1 + x2) / 2 - k.cx) * depth / k.fx,
                      ((y1 + y2) / 2 - k.cy) * depth / k.fy, depth, 1.0])
    xyz = (world_T_camera_optical(observation.pose, camera) @ point)[:3]
    return (float(xyz[0]), float(xyz[1])), apparent_width


class ObservedDoors:
    """Deduplicate door aliases, confirm across frames and separated viewpoints."""

    def __init__(self, settings=None):
        self.settings = settings or DoorSettings()
        self.landmarks = ObjectLandmarkMap(self.settings.dedupe_radius_m,
                                           self.settings.min_observations)
        self.first_view = {}
        self.separated = set()
        self.widths = {}
        self.detections = 0
        self.depth_accepted = 0
        self.revision = 0

    def update(self, observation, detections):
        before = {door.id for door in self.confirmed()}
        accepted = []
        for detection in sorted(detections, key=lambda d: -d.conf):
            if detection.cls not in DOOR_LABELS or detection.conf < self.settings.confidence:
                continue
            self.detections += 1
            if any(other.cls not in DOOR_LABELS and other.conf >= 0.5
                   and iou(detection.xyxy, other.xyxy) >= 0.6 for other in detections):
                continue  # e.g. a high-confidence cabinet is not a door boundary
            estimate = doorway_position(observation, detection, self.settings)
            if estimate is None:
                continue
            xy, width = estimate
            if any(math.dist(xy, old) < self.settings.dedupe_radius_m for old in accepted):
                continue  # overlapping aliases in ONE frame are not confirmations
            accepted.append(xy)
            self.depth_accepted += 1
            landmark = self.landmarks.observe("door", xy)
            pose = observation.pose
            first = self.first_view.setdefault(landmark.id, (pose.x, pose.y, pose.yaw))
            turn = abs(math.atan2(math.sin(pose.yaw - first[2]), math.cos(pose.yaw - first[2])))
            if (math.dist((pose.x, pose.y), first[:2]) >= self.settings.min_view_translation_m
                    or turn >= math.radians(self.settings.min_view_turn_deg)):
                self.separated.add(landmark.id)
            self.widths[landmark.id] = width
        if before != {door.id for door in self.confirmed()}:
            self.revision += 1

    def confirmed(self):
        return [door for door in self.landmarks.confirmed() if door.id in self.separated]

    def diagnostics(self):
        return {"detections": self.detections, "depth_accepted": self.depth_accepted,
                "confirmed": len(self.confirmed()),
                "landmarks": [{"id": door.id, "xy": list(door.xy), "count": door.count,
                               "width_m": self.widths[door.id]} for door in self.confirmed()]}

