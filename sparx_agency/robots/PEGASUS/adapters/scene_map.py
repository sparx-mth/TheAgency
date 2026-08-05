"""Where a scene's surveyed free-space map lives, and how to read it.

A map is expensive to produce (a raycast + overlap sweep of a whole building,
minutes of work) and completely deterministic, so it is surveyed once and
committed next to the platform that flies it. Everything that plans a flight
reads it from here; only ``tasks/planning/sim_flight_recording/survey_scene.py``
writes it.

Deliberately free of any Isaac Sim import, so a map can be loaded, inspected and
planned against on a laptop with no simulator — which is what makes the whole
episode planner unit-testable. The surveying itself, which does need a live
simulation, is :mod:`voxel_survey`.

**A map is only valid at the altitude it was surveyed at.** Clearance at head
height and clearance at desk height are different buildings, so the altitude is
part of the filename rather than a footnote.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

from sparx_agency.core.planning.environment import (
    OccupancyGrid2D, load_occupancy_grid, save_occupancy_grid,
)

MAP_DIR = Path(__file__).resolve().parent.parent / "maps"


def map_path(scene: str, altitude_m: float, map_dir: Path = None) -> Path:
    """Canonical file for one scene surveyed at one altitude.

    Args:
        scene: Scene key, see
            :data:`~sparx_agency.robots.PEGASUS.adapters.scene.INDOOR_SCENES`.
        altitude_m: Survey altitude, metres. Rounded to the centimetre in the
            name, so 1.5 and 1.500001 do not produce two files.
        map_dir: Override the directory. Defaults to ``robots/PEGASUS/maps``.

    Returns:
        The ``.npz`` path (which may not exist yet).
    """
    directory = Path(map_dir) if map_dir is not None else MAP_DIR
    return directory / f"{scene}_alt{round(altitude_m * 100):04d}cm.npz"


LANDABLE_LAYER = "landable"
"""Name of the layer marking cells the aircraft can be *put down* on.

Distinct from being flyable, and the distinction is not academic: a cell can be
wide open at 1.5 m and have a desk 70 cm below it. An episode that ends there
lands on the desk, tips, and every later flight is refused with ``Preflight
Fail: Attitude failure (roll)`` -- which is exactly how one campaign lost four
of its six episodes.
"""


def voxel_map_path(scene: str, map_dir: Path = None) -> Path:
    """Canonical file for a scene's ground-truth 3D voxel map.

    Unlike the 2D maps there is no altitude in the name: the voxel grid covers
    the whole building, and every 2D map is a slice of it.

    Args:
        scene: Scene key.
        map_dir: Override the directory. Defaults to ``robots/PEGASUS/maps``.

    Returns:
        The ``.npz`` path (which may not exist yet).
    """
    directory = Path(map_dir) if map_dir is not None else MAP_DIR
    return directory / f"{scene}_voxels.npz"


def load_voxel_map(scene: str, map_dir: Path = None):
    """Read a scene's ground-truth 3D voxel map.

    Args:
        scene: Scene key.
        map_dir: Override the directory.

    Returns:
        ``(VoxelGrid3D, metadata)``.

    Raises:
        FileNotFoundError: With the exact command to produce the missing map.
    """
    from sparx_agency.core.planning.environment.voxel_grid_3d import load_voxel_grid

    path = voxel_map_path(scene, map_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"no 3D voxel map for scene {scene!r} ({path}). Produce it with:\n"
            f"  tasks/planning/sim_flight_recording/survey_scene.py --scene {scene}"
        )
    return load_voxel_grid(path)


def save_scene_map(
    scene: str, altitude_m: float, grid: OccupancyGrid2D, metadata: Dict,
    map_dir: Path = None, layers: Dict = None,
) -> Path:
    """Write a surveyed map to its canonical location.

    Args:
        scene: Scene key.
        altitude_m: Survey altitude, metres.
        grid: The surveyed occupancy grid.
        metadata: Survey provenance (robot radius, resolution, cell counts, ...).
            ``scene`` and ``altitude_m`` are added automatically.
        map_dir: Override the directory.
        layers: Extra boolean masks, notably :data:`LANDABLE_LAYER`.

    Returns:
        The path written.
    """
    record = dict(metadata)
    record.update({"scene": scene, "altitude_m": float(altitude_m)})
    return save_occupancy_grid(map_path(scene, altitude_m, map_dir), grid, record,
                               layers=layers)


def load_scene_map(
    scene: str, altitude_m: float, map_dir: Path = None,
) -> Tuple[OccupancyGrid2D, Dict, Dict]:
    """Read a surveyed map back.

    Args:
        scene: Scene key.
        altitude_m: The altitude the map was surveyed at, metres.
        map_dir: Override the directory.

    Returns:
        ``(grid, metadata, layers)``.

    Raises:
        FileNotFoundError: With the exact command to produce the missing map.
    """
    path = map_path(scene, altitude_m, map_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"no surveyed map for scene {scene!r} at {altitude_m:.2f} m ({path}). "
            f"Produce it with:\n"
            f"  tasks/planning/sim_flight_recording/survey_scene.py "
            f"--scene {scene} --altitude {altitude_m}"
        )
    return load_occupancy_grid(path)


def available_maps(map_dir: Path = None):
    """Every surveyed map on disk, as ``(scene, altitude_m, path)`` tuples."""
    directory = Path(map_dir) if map_dir is not None else MAP_DIR
    if not directory.exists():
        return []
    found = []
    for path in sorted(directory.glob("*_alt*cm.npz")):
        scene, _, altitude = path.stem.rpartition("_alt")
        found.append((scene, int(altitude[:-2]) / 100.0, path))
    return found
