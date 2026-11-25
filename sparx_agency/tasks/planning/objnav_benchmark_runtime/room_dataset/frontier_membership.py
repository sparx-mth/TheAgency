"""Which frames of a planner recording see each frontier cluster: a frame sees a cluster when the cluster's goal cell is inside the camera's horizontal field of view and range and nothing known blocks the line to it on the map."""
import math
from dataclasses import dataclass

from sparx_agency.core.planning.planners.common.grid_geometry_2d import line_of_sight_clear

HFOV_DEG = 79.0
MAX_RANGE_M = 5.0
FREE = 0


@dataclass(frozen=True)
class MapFrame:
    origin: tuple
    resolution: float

    def cell_of(self, x, y):
        return int(math.floor((x - self.origin[0]) / self.resolution)), int(math.floor((y - self.origin[1]) / self.resolution))


def in_view(pose, point, hfov_deg=HFOV_DEG, max_range_m=MAX_RANGE_M):
    """Whether ``point`` (world x, y) is within range and the horizontal field of view of ``pose`` (x, y, yaw counter-clockwise from +x)."""
    dx, dy = point[0] - pose[0], point[1] - pose[1]
    if math.hypot(dx, dy) > max_range_m:
        return False
    turn = (math.atan2(dy, dx) - pose[2] + math.pi) % (2 * math.pi) - math.pi
    return abs(turn) <= math.radians(hfov_deg) / 2


def sees(pose, point, grid, frame, hfov_deg=HFOV_DEG, max_range_m=MAX_RANGE_M):
    """``in_view`` and every map cell on the line from the pose to the point is known free."""
    if not in_view(pose, point, hfov_deg, max_range_m):
        return False
    x0, y0 = frame.cell_of(pose[0], pose[1])
    x1, y1 = frame.cell_of(*point)
    height, width = grid.shape
    if not (0 <= x0 < width and 0 <= x1 < width and 0 <= y0 < height and 0 <= y1 < height):
        return False
    return line_of_sight_clear(grid != FREE, x0, y0, x1, y1)


def plan_steps(states):
    """The steps at which the planner's state becomes ``select``: where it picks the next goal."""
    return [i for i, state in enumerate(states) if state == "select" and (i == 0 or states[i - 1] != "select")]


def cluster_membership(poses, goals, grid, frame):
    """For each goal cell, the indices into ``poses`` of the frames that see it."""
    return [[i for i, pose in enumerate(poses) if sees(pose, goal, grid, frame)] for goal in goals]
