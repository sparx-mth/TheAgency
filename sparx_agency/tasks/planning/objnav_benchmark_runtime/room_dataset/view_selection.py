"""Which views of a room to keep: those that together see the most of its floor and of its detected objects, or random ones for comparison."""
import numpy as np

from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.geometry import habitat_position
from sparx_agency.tasks.planning.objnav_benchmark_runtime.object_dataset.projection import box_floor_point
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.geometry import (
    greedy_views, near_room, near_share, object_coverage, visible_cells, wall_hits)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_dataset.review_sheets import far_fraction

NEAR_M = 1.0
MAX_NEAR_SHARE = 0.4
MIN_GAIN = 0.02
REPEAT_CREDIT = 0.5
MIN_CONF = 0.3
SAME_OBJECT_M = 0.6
ROOM_GAP_M = 0.5
WALL_TOLERANCE_M = 0.5


def objects_in_room(detector, rgb, depth, camera, room_cells, protocol):
    placed = []
    for d in detector.detect(rgb):
        point = box_floor_point(d.xyxy, depth, (camera[0], camera[2], camera[3]), protocol.width, protocol.height, protocol.hfov_deg) if d.conf >= MIN_CONF else None
        if point is not None and near_room(point, room_cells, ROOM_GAP_M):
            placed.append({"cls": d.cls, "point": point})
    return placed


def describe_candidates(poses, render, room_cells, protocol, detector, perimeter=None):
    """What each renderable pose sees: ``cells`` (room floor in front, bool per room cell), ``objects`` in the room, ``near`` share of the picture, plus the pose."""
    found = []
    for cell, yaw in poses:
        try:
            rgb, depth, pose = render(cell, yaw)
        except EnvContractError:
            continue
        camera = (*habitat_position(dict(zip("xyz", pose))), yaw)
        found.append({"cell": cell, "yaw": yaw, "near": near_share(depth, NEAR_M), "far_fraction": far_fraction(depth),
                      "cells": visible_cells(depth, camera, room_cells, protocol.width, protocol.height, protocol.hfov_deg),
                      "objects": objects_in_room(detector, rgb, depth, camera, room_cells, protocol) if detector and near_share(depth, NEAR_M) <= MAX_NEAR_SHARE else [],
                      "walls": None if perimeter is None else wall_hits(depth, (camera[0], camera[2], yaw), perimeter, protocol.width, protocol.hfov_deg, WALL_TOLERANCE_M)})
    return found


def usable(candidates):
    return [i for i, c in enumerate(candidates) if c["near"] <= MAX_NEAR_SHARE]


def select(candidates, count, with_objects):
    """Indices of up to ``count`` candidates, best first: each adds the most room floor (and objects) the earlier ones did not show."""
    pool = usable(candidates)
    if not pool:
        return []
    objects, object_masks = object_coverage([candidates[i]["objects"] if with_objects else [] for i in pool], SAME_OBJECT_M)
    cells = len(candidates[pool[0]]["cells"])
    masks = [np.concatenate([candidates[i]["cells"], o]) for i, o in zip(pool, object_masks)]
    weights = np.concatenate([np.full(cells, 1 / cells), np.full(len(objects), 1 / max(1, len(objects)))])
    return [pool[j] for j in greedy_views(masks, count, MIN_GAIN, weights, REPEAT_CREDIT)]


def random_pick(candidates, count, rng, filtered):
    pool = usable(candidates) if filtered else list(range(len(candidates)))
    return [int(i) for i in rng.permutation(pool)[:count]]


def select_walls(candidates, count):
    """Indices of up to ``count`` candidates, best first: each hits the most room perimeter the earlier ones did not."""
    pool = usable(candidates)
    if not pool:
        return []
    size = len(candidates[pool[0]]["walls"])
    return [pool[j] for j in greedy_views([candidates[i]["walls"] for i in pool], count, MIN_GAIN, np.full(size, 1 / size), REPEAT_CREDIT)]


def sample_names(names, count, seed):
    """``count`` of the names, chosen at random with ``seed``, in sorted order; all of them when there are fewer."""
    names = sorted(names)
    chosen = np.random.default_rng(seed).permutation(len(names))[:count]
    return [names[i] for i in sorted(int(i) for i in chosen)]
