"""How much of each building the campaign has actually seen.

The question a long campaign has to answer is *when to stop*, and neither disk
nor flight count answers it. A thousand flights in a corridor already flown a
hundred times add almost nothing; two hundred in an unvisited wing add a great
deal. What matters is how much of each surveyed building the aircraft has been
through, and from how many directions.

Two numbers per scene, both measured against the same map the flights were
planned on:

**Spatial coverage** — the share of *reachable* one-metre cells the aircraft has
passed through. Reachable means the largest connected free component, which is
what ``free_space_sampler`` draws goals from; free space outside it is space no
flight can ever visit, and counting it would put a ceiling below 100 % that
looks like a plateau.

**Heading coverage** — the mean number of distinct 45° heading bins each visited
cell has been seen from, out of eight. This is the one people forget. The policy
is goal-conditioned, so it has to fly *any* direction from *any* place; a
corridor flown north ten times has been seen once as far as the camera is
concerned. Early in a campaign this sits near 2 and it is usually the slower of
the two to saturate.

Reading it: both curves are roughly logarithmic in flight count, so the useful
signal is the *increment* between successive samples. When another hour of
flying moves neither number, the campaign is producing near-duplicates and can
be stopped whatever the disk says.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

CELL_M = 1.0
"""Side of a coverage cell, metres.

Coarser than the map's 10 cm so that "has the aircraft been here" means what it
sounds like. At 10 cm a flight would have to retrace its own line to count a
cell twice, and coverage would read as a function of path length rather than of
where the aircraft has been.
"""

HEADING_BINS = 8
"""45° sectors. Fine enough that a corridor flown both ways scores two."""


@dataclass
class SceneCoverage:
    """One building's coverage, and what it cost to get there."""

    scene: str
    flights: int
    cells_reachable: int
    cells_seen: int
    mean_headings: float

    @property
    def fraction(self) -> float:
        """Share of reachable cells the aircraft has passed through."""
        return self.cells_seen / self.cells_reachable if self.cells_reachable else 0.0

    @property
    def heading_fraction(self) -> float:
        """Mean heading bins per visited cell, as a share of all eight."""
        return self.mean_headings / HEADING_BINS


def reachable_mask(map_path: Path):
    """The largest connected free component of a surveyed map, plus its geometry.

    Returns:
        ``(mask, resolution_m, origin_xy)``. The mask is what a flight *could*
        reach; everything else is unreachable and must not count against
        coverage.
    """
    from scipy import ndimage

    data = np.load(map_path)
    grid = data["grid"]
    free = grid == 0
    labels, count = ndimage.label(free)
    if count == 0:
        return np.zeros_like(free), float(data["resolution"]), data["origin"]
    sizes = ndimage.sum(free, labels, range(1, count + 1))
    biggest = int(np.argmax(sizes)) + 1
    return labels == biggest, float(data["resolution"]), data["origin"]


def measure(recordings_root: Path, scenes: Dict[str, Path],
            pose_stride: int = 5) -> List[SceneCoverage]:
    """Coverage of every scene, from the recordings written so far.

    Args:
        recordings_root: The campaign directory.
        scenes: ``{scene name: 2D map path}``.
        pose_stride: Sample every N-th pose. At 10 Hz and cruise speed
            consecutive poses are ~12 cm apart, far finer than a coverage cell,
            so striding costs no accuracy and most of the runtime.

    Returns:
        One :class:`SceneCoverage` per scene that has recordings, in the order
        the scenes were given.
    """
    prepared = {}
    for name, map_path in scenes.items():
        if not Path(map_path).is_file():
            continue
        mask, resolution, origin = reachable_mask(Path(map_path))
        step = max(1, int(round(CELL_M / resolution)))
        shape = (mask.shape[0] // step + 1, mask.shape[1] // step + 1)
        rows, cols = np.nonzero(mask)
        reachable = np.zeros(shape, bool)
        reachable[rows // step, cols // step] = True
        prepared[name] = {
            "reachable": reachable, "resolution": resolution, "origin": origin,
            "step": step, "seen": np.zeros(shape, bool), "headings": {},
            "flights": 0,
        }

    for meta_path in sorted(Path(recordings_root).rglob("meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            continue
        state = prepared.get(str(meta.get("scene")))
        poses_path = meta_path.parent / "poses.npy"
        if state is None or not poses_path.is_file():
            continue
        try:
            poses = np.load(poses_path)[::pose_stride]
        except (OSError, ValueError):
            continue
        if poses.size == 0:
            continue
        shape = state["seen"].shape
        origin, resolution, step = state["origin"], state["resolution"], state["step"]
        rows = np.clip(((poses[:, 2] - origin[1]) / resolution).astype(int) // step,
                       0, shape[0] - 1)
        cols = np.clip(((poses[:, 1] - origin[0]) / resolution).astype(int) // step,
                       0, shape[1] - 1)
        state["seen"][rows, cols] = True
        sector = (((poses[:, 3] + np.pi) / (2 * np.pi / HEADING_BINS))
                  .astype(int) % HEADING_BINS)
        for row, col, bin_index in zip(rows, cols, sector):
            state["headings"].setdefault((row, col), set()).add(int(bin_index))
        state["flights"] += 1

    out: List[SceneCoverage] = []
    for name, state in prepared.items():
        if not state["flights"]:
            continue
        reachable = state["reachable"]
        visited = [len(bins) for cell, bins in state["headings"].items()
                   if reachable[cell]]
        out.append(SceneCoverage(
            scene=name,
            flights=state["flights"],
            cells_reachable=int(reachable.sum()),
            cells_seen=int((state["seen"] & reachable).sum()),
            mean_headings=float(np.mean(visited)) if visited else 0.0,
        ))
    return out


def default_scene_maps(map_dir: Optional[Path] = None,
                       altitude_m: float = 1.5) -> Dict[str, Path]:
    """Every surveyed 2D map, keyed by scene name.

    Reads the directory rather than a hardcoded list, so a newly surveyed
    building appears without touching this file.
    """
    if map_dir is None:
        map_dir = (Path(__file__).resolve().parents[2]
                   / "robots" / "PEGASUS" / "maps")
    suffix = f"_alt{int(round(altitude_m * 100)):04d}cm.npz"
    return {path.name[: -len(suffix)]: path
            for path in sorted(Path(map_dir).glob(f"*{suffix}"))}
