"""Where to put the camera around an object. numpy only; habitat's yaw: 0 faces -z, positive turns toward -x."""
import hashlib
import math

import numpy as np

APPROACH_DISTANCES_M = (4.5, 3.5, 2.5, 1.8, 1.2, 0.8)
HELD_OUT_SALT = "object-views-1"


def yaw_toward(from_xz, to_xz):
    dx, dz = to_xz[0] - from_xz[0], to_xz[1] - from_xz[1]
    return math.atan2(-dx, -dz)


def habitat_position(view):
    """The stored pose is the public ENU one: x = -habitat z, y = -habitat x, z = habitat y."""
    return np.array([-view["y"], view["z"], -view["x"]])


def forward_xz(yaw):
    return (-math.sin(yaw), -math.cos(yaw))


def approach_positions(center_xz, azimuth_rad, distances=APPROACH_DISTANCES_M):
    """Camera (x, z) on a straight line toward the object, one point per distance, from far to near."""
    return [(center_xz[0] + d * math.cos(azimuth_rad), center_xz[1] + d * math.sin(azimuth_rad)) for d in distances]


def jittered_yaws(yaw_to_object, count, max_jitter_rad, rng):
    """The first view looks straight at the object, the others turn away by a random angle up to ``max_jitter_rad``."""
    return [(yaw_to_object, 0.0)] + [(yaw_to_object + j, j) for j in rng.uniform(-max_jitter_rad, max_jitter_rad, count - 1)]


def is_held_out(scene, fraction):
    """A scene-level split that never changes: the same scene is held out in every build."""
    digest = hashlib.sha256((HELD_OUT_SALT + scene).encode()).digest()
    return int.from_bytes(digest[:4], "big") / 2 ** 32 < fraction


def touches_border(xyxy, width, height):
    return xyxy[0] <= 0 or xyxy[1] <= 0 or xyxy[2] >= width or xyxy[3] >= height
