"""The BEV map lane: occupancy snapshots on disk, resolved at an instant.

The recorder saves a grid only when the map actually changed, so a run holds
hundreds of snapshots at an irregular rate. They are indexed here by stamp and
answered by an as-of join -- the map the drone was planning on at that moment,
not the newest one -- and the last few grids are kept in an LRU cache so
stepping back and forth through a replay does not re-read from disk.
"""
from __future__ import annotations

import glob
import json
import os
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np

from sparx_agency.tasks.planning.nav_debug.frame import BevMap
from sparx_agency.tasks.planning.nav_debug.schema import BEV_CONF_DIR, BEV_DIR
from sparx_agency.tasks.planning.nav_debug.sources import as_of_index

_CACHE_SIZE = 4


class BevSource:
    """Every recorded BEV snapshot of one run, resolvable by timestamp."""

    def __init__(self, run_dir: str, manifest: Optional[dict] = None) -> None:
        self.run_dir = run_dir
        self.manifest = manifest or {}
        self.index = self._index(run_dir)          # [(t, npy_path, geometry)]
        self.stamps = [t for t, _, _ in self.index]
        self._cache = OrderedDict()                # npy_path -> BevMap

    def __len__(self) -> int:
        return len(self.index)

    def at(self, t: float) -> Tuple[Optional[BevMap], Optional[np.ndarray]]:
        """The map (and its confidence grid) latest at or before ``t``."""
        j = as_of_index(self.stamps, t)
        if j is None:
            return None, None
        _, npy, meta = self.index[j]
        return self._load(npy, meta), self._confidence(npy)

    @staticmethod
    def _index(run_dir: str) -> List[Tuple[float, str, dict]]:
        """Stamp every ``bev/<ms>.npy``, from its sidecar or its filename."""
        index = []
        for npy in sorted(glob.glob(os.path.join(run_dir, BEV_DIR, "*.npy"))):
            meta = _sidecar(npy[:-4] + ".json")
            t = meta.get("t")
            if t is None:
                try:
                    t = float(os.path.splitext(os.path.basename(npy))[0]) / 1000.0
                except ValueError:
                    continue
            index.append((float(t), npy, meta))
        index.sort(key=lambda e: e[0])
        return index

    def _load(self, npy: str, meta: dict) -> Optional[BevMap]:
        """One grid + its geometry, from the cache when it is still there."""
        if npy in self._cache:
            self._cache.move_to_end(npy)
            return self._cache[npy]
        try:
            grid = np.load(npy)
        except (OSError, ValueError):
            return None
        geo = meta or self.manifest.get("bev", {})
        bev = BevMap(grid=grid,
                     resolution=float(geo.get("resolution", 0.15)),
                     origin_x=float(geo.get("origin_x", 0.0)),
                     origin_y=float(geo.get("origin_y", 0.0)),
                     frame_id=str(geo.get("frame_id", "world")),
                     stamp=float(geo.get("t", 0.0)))
        self._cache[npy] = bev
        if len(self._cache) > _CACHE_SIZE:
            self._cache.popitem(last=False)
        return bev

    def _confidence(self, npy: str) -> Optional[np.ndarray]:
        """The per-cell confidence grid co-registered with ``npy``, if recorded."""
        path = os.path.join(self.run_dir, BEV_CONF_DIR, os.path.basename(npy))
        if not os.path.isfile(path):
            return None
        try:
            return np.load(path)
        except (OSError, ValueError):
            return None


def _sidecar(path: str) -> dict:
    """A snapshot's geometry sidecar; ``{}`` when absent or unreadable."""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as fh:
            meta = json.load(fh)
    except (OSError, IOError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}
