"""Greedy grouping and persistent floor-qualified identities for observed map locations."""
import math


class SpatialIds:
    def __init__(self, match_m):
        self.match_m = float(match_m)
        self._known = {}
        self._next = 0

    def identify(self, floor, xy):
        known = self._known.setdefault(int(floor), {})
        nearest = min(known, key=lambda i: math.dist(known[i], xy), default=None)
        if nearest is None or math.dist(known[nearest], xy) > self.match_m:
            nearest = self._next
            self._next += 1
        known[nearest] = (float(xy[0]), float(xy[1]))
        return nearest


def merge_near_seed(items, radius_m, xy=None):
    """Each item joins the first nearby seed in input order, or starts a new group."""
    where = xy if xy is not None else (lambda item: item.xy)
    groups, seeds = [], []
    for item in items:
        point = where(item)
        for group, seed in zip(groups, seeds):
            if math.dist(seed, point) <= radius_m:
                group.append(item)
                break
        else:
            groups.append([item])
            seeds.append(point)
    return groups
