"""Room views taken from what a robot passes anyway: the poses along a walk, and an online rule that keeps the best few as the walk goes. numpy only."""
import math

import numpy as np

from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.grid_views import FORWARD_STEP_M, TURN_DEG, heading_of
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.geometry import greedy_views
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.view_selection import MIN_GAIN, REPEAT_CREDIT, usable


def toward(yaw, target):
    """``yaw`` turned one step of ``TURN_DEG`` toward ``target`` by the short way, landing on it when it is within a step."""
    difference = (target - yaw + math.pi) % (2 * math.pi) - math.pi
    step = math.radians(TURN_DEG)
    return target if abs(difference) <= step + 1e-9 else yaw + math.copysign(step, difference)


def walk_poses(path, start_yaw):
    """``(point, yaw)`` at the start and after every action of walking the 3-D ``path`` from ``start_yaw``: turn in place to each segment's heading, then forward steps of a quarter metre."""
    points = [np.asarray(path[0], dtype=float)]
    for p in path[1:]:
        if math.dist((p[0], p[2]), (points[-1][0], points[-1][2])) > 1e-6:
            points.append(np.asarray(p, dtype=float))
    poses, yaw = [(points[0], start_yaw)], start_yaw
    for a, b in zip(points, points[1:]):
        heading = heading_of((a[0], a[2]), (b[0], b[2]))
        while abs((heading - yaw + math.pi) % (2 * math.pi) - math.pi) > 1e-9:
            yaw = toward(yaw, heading)
            poses.append((a, yaw))
        length = math.dist((a[0], a[2]), (b[0], b[2]))
        steps = math.ceil(length / FORWARD_STEP_M - 1e-6)
        for k in range(1, steps + 1):
            poses.append((a + (b - a) * min(1.0, k * FORWARD_STEP_M / length), yaw))
    return poses


def even_picks(count_frames, count):
    """Indices spread evenly through ``count_frames`` frames, none when there are none; every frame when there are no more than ``count``."""
    if count_frames <= count:
        return list(range(count_frames))
    return [round((j + 1) * (count_frames - 1) / (count + 1)) for j in range(count)]


def keep_best(frames, count, min_new=MIN_GAIN, repeat_credit=REPEAT_CREDIT):
    """Indices of the frames an online keeper holds after the walk: each new usable frame joins those kept and the best ``count`` by added floor stay."""
    kept = []
    for i, frame in enumerate(frames):
        if not usable([frame]):
            continue
        pool = kept + [i]
        cells = len(frame["cells"])
        chosen = greedy_views([frames[j]["cells"] for j in pool], count, min_new, np.full(cells, 1.0 / cells), repeat_credit)
        kept = [pool[c] for c in chosen]
    return kept
