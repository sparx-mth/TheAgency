"""Read the published MP3D ObjectNav v1 release: episodes and evaluator-only goals.

Never pass this object, its rows, or its view points to a SearchPolicy. Only
local files are read; this loader neither downloads nor generates data, and it
never regenerates goals, view points or a navmesh.

Layout, as habitat-lab's ``ObjectNavDatasetV1`` reads it::

    <episodes_dir>/<split>.json.gz        category_to_task_category_id, ...
    <episodes_dir>/<split>/content/<scene>.json.gz   episodes + goals_by_category

Episode ids are scene-qualified row indices. habitat-lab renumbers episodes
from 0 in every content file (``episode.episode_id = str(i)``), so the raw ids
repeat across a split's scenes and cannot key results on their own; the file's
own id is kept alongside as metadata.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, Tuple

import numpy as np

from sparx_agency.core.planning.objnav.labels.datasets.mp3d import CATEGORIES
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.protocol import (
    PROTOCOL, PUBLISHED_VAL_EPISODES, PUBLISHED_VAL_SCENES,
)

#: habitat-lab strips this prefix from a serialized ``scene_id``.
SCENE_PATH_PREFIX = "data/scene_datasets/"


@dataclass(frozen=True)
class MP3DEpisode:
    """Private evaluator/simulator input.

    Quaternion order is the published XYZW, converted at the simulator edge
    (habitat-sim's ``quaternion.from_float_array`` wants WXYZ).
    """

    episode_id: str
    published_episode_id: str
    scene: str
    scene_glb: str
    category: str
    goals_key: str
    start_position: Tuple[float, ...]
    start_rotation_xyzw: Tuple[float, ...]
    published_geodesic_m: float

    def start_rotation_wxyz(self) -> Tuple[float, ...]:
        """The published rotation in habitat-sim's coefficient order."""
        x, y, z, w = self.start_rotation_xyzw
        return (w, x, y, z)


def _vector(value, size, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError("%s must contain %d finite numbers" % (name, size))
    return tuple(float(v) for v in array)


def _view_points(goals, goals_key):
    """Every published view-point position of one category, as habitat-lab reads them.

    The evaluator's distance is measured to these, not to the object centres:
    ``DistanceToGoal`` with ``distance_to: VIEW_POINTS`` collects
    ``[view_point.agent_state.position for goal in episode.goals
    for view_point in goal.view_points]``.
    """
    points = []
    for goal in goals:
        for view in goal.get("view_points") or ():
            state = view.get("agent_state") or {}
            points.append(_vector(state.get("position"), 3, "view point position"))
    if not points:
        raise ValueError("No published view points for %s" % goals_key)
    return np.asarray(points, dtype=np.float32)


class MP3DDataset:
    """Validated local release; the full split is the default, never a subset.

    Args:
        episodes_dir: The split directory holding ``content/`` and, beside it
            or inside it, ``<split>.json.gz`` with the category tables.
        scenes_dir: ``data/scene_datasets`` -- the parent of ``mp3d/``.
        full: Require the complete published validation split (11 scenes,
            2,195 episodes). False is reserved for explicitly labelled
            smoke/subset runs.
        scene: Restrict loading to one scene, for a demo. Requires full=False.
    """

    def __init__(self, episodes_dir, scenes_dir, full=True, scene=None):
        self.episodes_dir = Path(episodes_dir).expanduser().resolve()
        self.scenes_dir = Path(scenes_dir).expanduser().resolve()
        self.split = PROTOCOL.split
        self.selected_scene = scene
        if scene is not None and full:
            raise ValueError("Select one scene with full=False for a demo")
        self._assets = []
        self.category_ids = self._categories()
        files = sorted((self.episodes_dir / "content").glob("*.json.gz"))
        if scene is not None:
            files = [p for p in files if p.name == scene + ".json.gz"]
        if not files:
            raise FileNotFoundError(
                "No %s content files under %s. Point --episodes-dir at the "
                "split directory of the extracted objectnav_mp3d_v1 release."
                % ("scene " + scene if scene else "episode", self.episodes_dir))
        self.episodes: Dict[str, MP3DEpisode] = {}
        self.scene_counts: Dict[str, int] = {}
        self._view_points: Dict[str, np.ndarray] = {}
        self._scene_files: Dict[str, Path] = {}
        for path in files:
            self._read_scene(path)
        if full and (len(self.episodes) != PUBLISHED_VAL_EPISODES
                     or len(self.scene_counts) != PUBLISHED_VAL_SCENES):
            raise ValueError(
                "A full MP3D %s run needs the published %d episodes in %d "
                "scenes; this release has %d in %d. Use full=False (a --scene "
                "or --limit run) to evaluate a labelled subset."
                % (self.split, PUBLISHED_VAL_EPISODES, PUBLISHED_VAL_SCENES,
                   len(self.episodes), len(self.scene_counts)))

    def _categories(self):
        """The release's own category table, checked against our label table."""
        candidates = [self.episodes_dir / (self.split + ".json.gz"),
                      self.episodes_dir.parent / (self.split + ".json.gz"),
                      self.episodes_dir / (self.episodes_dir.name + ".json.gz")]
        for path in candidates:
            if path.is_file():
                break
        else:
            raise FileNotFoundError(
                "Missing the split's category table (%s.json.gz) beside or "
                "inside %s" % (self.split, self.episodes_dir))
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        table = payload.get("category_to_task_category_id")
        scene_table = (payload.get("category_to_mp3d_category_id")
                       or payload.get("category_to_scene_annotation_category_id"))
        if not isinstance(table, dict) or not isinstance(scene_table, dict):
            raise ValueError("%s lacks the ObjectNav category tables" % path)
        if set(table) != set(CATEGORIES):
            missing = sorted(set(CATEGORIES) - set(table))
            extra = sorted(set(table) - set(CATEGORIES))
            raise ValueError(
                "This release's goal categories are not MP3D ObjectNav v1's "
                "21: missing %r, unexpected %r. Our label table would score "
                "the wrong vocabulary." % (missing, extra))
        if set(table) != set(scene_table):
            raise ValueError("category_to_task and category_to_mp3d disagree")
        self._assets.append(path)
        return {name: int(index) for name, index in table.items()}

    def _read_scene(self, path):
        """Parse one content file, keep compact arrays, and drop the raw goals."""
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        rows = payload.get("episodes")
        goals_by_category = payload.get("goals_by_category")
        if not isinstance(rows, list) or not rows or not isinstance(goals_by_category, dict):
            raise ValueError("%s is not an ObjectNav content file" % path)
        scene = path.name[:-len(".json.gz")]
        self._scene_files[scene] = path
        self._assets.append(path)
        glb = None
        for index, row in enumerate(rows):
            episode = self._episode(row, scene, index)
            if glb is None:
                glb = episode.scene_glb
                for asset in self._scene_assets(glb):
                    self._assets.append(asset)
            elif episode.scene_glb != glb:
                raise ValueError("%s mixes scenes: %r and %r" % (path, glb, episode.scene_glb))
            if episode.goals_key not in self._view_points:
                goals = goals_by_category.get(episode.goals_key)
                if not isinstance(goals, list) or not goals:
                    raise ValueError("Missing goals for %s" % episode.goals_key)
                self._view_points[episode.goals_key] = _view_points(goals, episode.goals_key)
            self.episodes[episode.episode_id] = episode
        self.scene_counts[scene] = len(rows)

    def _episode(self, row, scene, index):
        category = row.get("object_category")
        if category is None and row.get("goals"):
            category = row["goals"][0].get("object_category")
        if category not in CATEGORIES:
            raise ValueError("Unknown MP3D goal category %r in %s" % (category, scene))
        scene_id = row.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id.endswith(".glb"):
            raise ValueError("Episode %s/%d has no scene mesh" % (scene, index))
        if scene_id.startswith(SCENE_PATH_PREFIX):
            scene_id = scene_id[len(SCENE_PATH_PREFIX):]
        parts = PurePosixPath(scene_id).parts
        if PurePosixPath(scene_id).is_absolute() or ".." in parts:
            raise ValueError("Refusing an escaping scene path: %r" % scene_id)
        rotation = _vector(row.get("start_rotation"), 4, "start_rotation (XYZW)")
        if not np.isclose(np.linalg.norm(rotation), 1.0, atol=1e-4):
            raise ValueError("start_rotation must be a unit XYZW quaternion")
        info = row.get("info") or {}
        geodesic = info.get("geodesic_distance")
        geodesic = float(geodesic) if isinstance(geodesic, (int, float)) else float("nan")
        return MP3DEpisode(
            episode_id="%s/%06d" % (scene, index),
            published_episode_id=str(row.get("episode_id", index)),
            scene=scene, scene_glb=scene_id, category=category,
            goals_key="%s_%s" % (PurePosixPath(scene_id).name, category),
            start_position=_vector(row.get("start_position"), 3, "start_position"),
            start_rotation_xyzw=rotation, published_geodesic_m=geodesic)

    def _scene_assets(self, scene_glb):
        """The mesh and the published navmesh; neither is generated here."""
        mesh = self.scenes_dir / scene_glb
        navmesh = mesh.with_suffix(".navmesh")
        for asset in (mesh, navmesh):
            if not asset.is_file() or asset.stat().st_size == 0:
                raise FileNotFoundError(
                    "Missing or empty MP3D asset: %s. Matterport3D's habitat "
                    "task download ships <scene>.glb beside <scene>.navmesh; "
                    "this evaluator never recomputes a published navmesh."
                    % asset)
        return mesh, navmesh

    def scene_paths(self, episode):
        """(mesh, navmesh) for one episode, both checked at load."""
        mesh = self.scenes_dir / episode.scene_glb
        return mesh, mesh.with_suffix(".navmesh")

    def view_points(self, episode):
        """Evaluator-only view points of this episode's goal category, in Habitat XYZ."""
        return self._view_points[episode.goals_key]

    def manifest(self):
        """Hash episodes, category tables, meshes and navmeshes for run/resume provenance."""
        files = []
        for path in self._assets:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            files.append({"path": str(path), "bytes": path.stat().st_size,
                          "sha256": digest.hexdigest()})
        return {"release": "Habitat ObjectNav MP3D v1",
                "split": self.split,
                "episode_count": len(self.episodes),
                "selected_scene": self.selected_scene,
                "scenes": sorted(self.scene_counts),
                "scene_counts": dict(self.scene_counts),
                "category_to_task_category_id": dict(self.category_ids),
                "view_point_counts": {key: int(len(points))
                                      for key, points in sorted(self._view_points.items())},
                "files": files}
