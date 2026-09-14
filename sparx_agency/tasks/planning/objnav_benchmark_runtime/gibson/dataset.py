"""Read the published Gibson v1.1 episodes and evaluator-only semantic maps.

Never pass this object, its rows, or its maps to a SearchPolicy. Episode ids
are scene-qualified row indices, stable even when upstream omits episode_id.
Only local files are read; this loader neither downloads nor generates data.
"""
from __future__ import annotations

import bz2
import gzip
import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sparx_agency.core.planning.objnav.labels.datasets.gibson import CATEGORIES
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import SCENES


class _ArrayUnpickler(pickle.Unpickler):
    """Allow numpy containers, never arbitrary globals from a downloaded pickle."""

    def find_class(self, module, name):
        allowed = {
            ("numpy", "ndarray"), ("numpy", "dtype"),
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy.core.multiarray", "scalar"),
            ("numpy.core.numeric", "_frombuffer"),
        }
        module = module.replace("numpy._core", "numpy.core")
        if (module, name) not in allowed:
            raise ValueError("Unapproved global in Gibson map pickle: %s.%s"
                             % (module, name))
        return super().find_class(module, name)


@dataclass(frozen=True)
class GibsonEpisode:
    """Private evaluator/simulator input. Quaternion order is WXYZ, not XYZW."""

    episode_id: str
    scene: str
    category: str
    category_index: int
    floor_id: int
    start_position: tuple
    start_rotation: tuple


def _vector(value, size, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError("%s must contain %d finite numbers" % (name, size))
    return tuple(float(v) for v in array)


def _episode(row, scene, index):
    if "object_categories" in row or "object_ids" in row:
        raise ValueError("PONI multi-goal episodes are not SemExp v1.1; "
                         "do not silently take their first goal")
    category = row["object_category"]
    category_index = row["object_id"]
    floor_id = row["floor_id"]
    if (type(category_index) is not int or category_index not in range(6)
            or CATEGORIES[category_index] != category):
        raise ValueError("Unknown/mismatched Gibson object_category/object_id")
    if type(floor_id) is not int or floor_id < 0:
        raise ValueError("floor_id must be a non-negative integer")
    rotation = _vector(row["start_rotation"], 4, "start_rotation (WXYZ)")
    if not np.isclose(np.linalg.norm(rotation), 1.0, atol=1e-4):
        raise ValueError("start_rotation must be a unit WXYZ quaternion")
    return GibsonEpisode(
        "%s/%06d" % (scene, index), scene, category, category_index, floor_id,
        _vector(row["start_position"], 3, "start_position"), rotation)


class GibsonDataset:
    """Validated local release; full validation is the default, never a subset.

    Args:
        episodes_dir: v1.1/val directory, containing content/ and val_info.pbz2.
        scenes_dir: Directory containing <scene>.glb and <scene>.navmesh.
        full: Require exactly 1,000 episodes (200 in each of the five scenes).
            False is reserved for explicitly labelled smoke/subset runs.
    """

    def __init__(self, episodes_dir, scenes_dir, full=True, scene=None):
        import json

        if scene is not None and (scene not in SCENES or full):
            raise ValueError("Select one known scene with full=False for a demo")
        self.selected_scene = scene
        self.episodes_dir = Path(episodes_dir).expanduser().resolve()
        self.scenes_dir = Path(scenes_dir).expanduser().resolve()
        info_file = self.episodes_dir / "val_info.pbz2"
        with bz2.open(info_file, "rb") as stream:
            self._maps = _ArrayUnpickler(stream).load()
        if not isinstance(self._maps, dict):
            raise ValueError("val_info.pbz2 must hold a scene dictionary")
        files = sorted((self.episodes_dir / "content").glob("*_episodes.json.gz"))
        if scene is not None:
            files = [p for p in files if p.name == scene + "_episodes.json.gz"]
        found = {p.name[:-len("_episodes.json.gz")] for p in files}
        if not found or not found.issubset(SCENES) or (full and found != set(SCENES)):
            raise ValueError("Expected Gibson tiny val scenes %r; found %r"
                             % (SCENES, sorted(found)))
        self.episodes = {}
        self._validated_floors = set()
        self._assets = [info_file]
        self.scene_counts = {}
        for path in files:
            scene = path.name[:-len("_episodes.json.gz")]
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                rows = json.load(stream)["episodes"]
            if not isinstance(rows, list) or not rows or (full and len(rows) != 200):
                raise ValueError("%s must contain 200 episodes for a full run" % path)
            self.scene_counts[scene] = len(rows)
            self._assets.append(path)
            for suffix in (".glb", ".navmesh"):
                asset = self.scenes_dir / (scene + suffix)
                if not asset.is_file() or asset.stat().st_size == 0:
                    raise FileNotFoundError("Missing/non-empty Gibson asset: %s" % asset)
                self._assets.append(asset)
            for index, row in enumerate(rows):
                episode = _episode(row, scene, index)
                self.floor(episode)  # validate every referenced map before a run
                self.episodes[episode.episode_id] = episode

    def floor(self, episode):
        """Evaluator-only map and origin, checked rather than guessed or snapped."""
        try:
            floor = self._maps[episode.scene][episode.floor_id]
            semantic = np.asarray(floor["sem_map"])
            origin = np.asarray(floor["origin"], dtype=np.float64)
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Missing floor map for %s" % episode.episode_id) from exc
        key = (episode.scene, episode.floor_id)
        if key not in self._validated_floors and (semantic.ndim != 3 or semantic.shape[0] < 7
                or min(semantic.shape[1:]) < 2 or semantic.dtype.kind not in "buif"
                or not np.isfinite(semantic).all()
                or not np.isin(semantic, (0, 1)).all()):
            raise ValueError("sem_map must be binary [free, six goals, ...] x H x W")
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("Floor origin must contain two finite centimetre values")
        self._validated_floors.add(key)
        if not semantic[episode.category_index + 1].any():
            raise ValueError("Empty target map for %s" % episode.episode_id)
        return semantic, origin

    def manifest(self):
        """Hash episodes, GT maps, meshes and navmeshes for run/resume provenance."""
        files = []
        for path in self._assets:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            files.append({"path": str(path), "bytes": path.stat().st_size,
                          "sha256": digest.hexdigest()})
        return {"release": "SemExp ObjectNav Gibson v1.1",
                "episode_count": len(self.episodes),
                "selected_scene": self.selected_scene,
                "scene_counts": dict(self.scene_counts), "files": files}

