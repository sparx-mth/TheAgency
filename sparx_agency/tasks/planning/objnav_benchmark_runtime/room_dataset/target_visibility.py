"""Whether an annotated object is in a camera's picture and not hidden: the geometric test used to leave rooms where the target is in view out of the prior test. numpy only; habitat axes."""
import math

import numpy as np

MIN_FORWARD_M = 0.3


def target_visible(camera_xyz_yaw, depth_m, center_xyz, width, height, hfov_deg, margin_px, tolerance_m, max_depth_m):
    """The centre projects into the picture (grown by ``margin_px``) and, within sensor range, no surface in front of it hides it; beyond range or without depth it cannot be checked and counts as visible."""
    cam_x, cam_y, cam_z, yaw = camera_xyz_yaw
    dx, dz = center_xyz[0] - cam_x, center_xyz[2] - cam_z
    forward = dx * -math.sin(yaw) + dz * -math.cos(yaw)
    sideways = dx * math.cos(yaw) + dz * -math.sin(yaw)
    if forward < MIN_FORWARD_M:
        return False
    focal = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    column, row = width / 2 + focal * sideways / forward, height / 2 - focal * (center_xyz[1] - cam_y) / forward
    if not (-margin_px <= column < width + margin_px and -margin_px <= row < height + margin_px):
        return False
    if forward > max_depth_m:
        return True
    surface = float(np.asarray(depth_m, dtype=float)[min(height - 1, max(0, int(row))), min(width - 1, max(0, int(column)))])
    return not np.isfinite(surface) or surface <= 0 or surface >= forward - tolerance_m


def steps_in_index(poses, forward_steps):
    """Index of the pose after the ``forward_steps``-th forward step of a walk (turns in place do not count); the last pose if the walk has fewer."""
    moved = 0
    for i in range(1, len(poses)):
        if float(np.linalg.norm(np.asarray(poses[i][0], dtype=float) - np.asarray(poses[i - 1][0], dtype=float))) > 1e-9:
            moved += 1
            if moved == forward_steps:
                return i
    return len(poses) - 1
