"""Read the published MP3D ObjectNav v1 episodes and their evaluator-only goals.

Never pass this object, its rows, or its goals to a ``SearchPolicy``: the goal
view points are exactly the privileged information a zero-shot benchmark
exists to withhold.

Layout, as habitat-lab's ``ObjectNavDatasetV1`` reads it (``objectnav_mp3d_v1.zip``)::

    <episodes_dir>/<split>.json.gz                 category tables, no episodes
    <episodes_dir>/content/<scene>.json.gz         one scene's episodes and goals

where ``<episodes_dir>`` is the split directory, e.g.
``.../objectnav/mp3d/v1/val``. ``val_mini`` ships its episodes and
``goals_by_category`` in ``<split>.json.gz`` itself, with no ``content/``;
that file is read as one shard per scene, numbered the same way. A shard whose episodes carry their ``goals``
inline (the release predates habitat-lab's ``goals_by_category``
de-duplication, and ``ObjectNavDatasetV1.dedup_goals`` folds them at load) is
read exactly as one that already has ``goals_by_category``.

Three habitat-lab conventions are reproduced rather than guessed at, because
each would otherwise be an invisible bug:

* ``ObjectNavDatasetV1.from_json`` **renumbers** every episode
  ``episode_id = str(i)`` within its own content shard, so ids collide across
  scenes. Ours are scene-qualified (``2azQ1b91cZZ/000042``) and stable.
* ``start_rotation`` is a quaternion in **XYZW** order. The shared Habitat
  bridge takes WXYZ, so the order is converted here, once.
* ``goals_key`` is ``f"{basename(scene_id)}_{object_category}"``
  (``2azQ1b91cZZ.glb_chair``), the key the goals live under.

Scenes: every row's ``scene_id`` is ``mp3d/<scene>/<scene>.glb`` (optionally
behind habitat-lab's ``data/scene_datasets/`` prefix, which is stripped), and
habitat-sim derives the sibling navmesh as ``<scene>.navmesh``. Both files are
the publisher's; this loader never generates, snaps or downloads anything.

Python 3.8 syntax; numpy only, no simulator import.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from sparx_agency.core.planning.objnav.labels.datasets.mp3d import CATEGORIES
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.protocol import PUBLISHED_VAL


def published_counts(protocol):
    """What this protocol's split is documented to contain, or ``{}``.

    Only the validation split is a published benchmark. A development run on
    ``train`` or ``val_mini`` has no count to be complete against, and
    pretending otherwise is how a subset gets quoted as a split.
    """
    if protocol.split != "val":
        return {}
    return PUBLISHED_VAL.get(protocol.dataset_version, {})


#: habitat-lab strips this prefix from a row's ``scene_id`` before joining it
#: to the scenes directory. The MP3D v1 rows do not carry it -- they start
#: straight at ``mp3d/`` -- but the strip is what habitat-lab does.
SCENE_PATH_PREFIX = "data/scene_datasets/"

#: The publisher's scene dataset config for the Habitat MP3D release, installed
#: as ``<scenes_dir>/mp3d/<this name>`` (habitat-sim's DATASETS.md). It is what
#: registers each scene's ``_semantic.ply`` to its ``.glb``: the meshes are in
#: different frames (``up``/``front`` against ``semantic_up``/``semantic_front``),
#: and habitat-sim's built-in ``"default"`` dataset leaves the semantic mesh
#: misplaced (measured 2026-11-04, habitat-sim 0.2.4: 8 of 418 rendered ids on
#: their objects in X7HyMhZNoso, against 604 of 630 under this config).
SEMANTIC_SCENE_DATASET_CONFIG = "mp3d.scene_dataset_config.json"
SEMANTIC_SCENE_DATASET_CONFIG_URL = "http://dl.fbaipublicfiles.com/habitat/mp3d/config_v1/mp3d.scene_dataset_config.json"


def _vector(value, size, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError("%s must contain %d finite numbers, got %r"
                         % (name, size, value))
    return tuple(float(v) for v in array)


def xyzw_to_wxyz(rotation):
    """Reorder a published XYZW start rotation into the WXYZ the bridge wants.

    Raises:
        ValueError: Not four finite numbers, or not a unit quaternion.
    """
    x, y, z, w = _vector(rotation, 4, "start_rotation (XYZW)")
    if not np.isclose(np.linalg.norm((x, y, z, w)), 1.0, atol=1e-4):
        raise ValueError("start_rotation must be a unit quaternion, got %r"
                         % (rotation,))
    return (w, x, y, z)


@dataclass(frozen=True)
class MP3DEpisode:
    """One published episode. Private evaluator/simulator input.

    Attributes:
        episode_id: ``"<scene>/<index within the shard>"``, unique within
            (benchmark, split) despite habitat-lab's per-shard renumbering.
        scene_key: The Matterport scene hash, e.g. ``"2azQ1b91cZZ"`` -- the
            scene's folder name and its shard's name.
        scene_relative_path: The row's ``scene_id`` with habitat-lab's
            ``data/scene_datasets/`` prefix stripped (``mp3d/<scene>/<scene>.glb``).
        goals_key: The ``goals_by_category`` key this episode's goals live under.
        category: ``object_category``, in the dataset's own spelling.
        start_position: Habitat-frame ``(x, y, z)``, +Y up.
        start_rotation_wxyz: The start rotation, already reordered.
        published_geodesic_m: The row's ``info.geodesic_distance`` -- the
            publisher's own geodesic from this start to the nearest goal view
            point, the same quantity as ``l``. A cross-check, never the SPL
            reference; a disagreement is a defect, not a difference of
            definition.
        source_episode_id: The row's own ``episode_id``, kept for provenance.
    """

    episode_id: str
    scene_key: str
    scene_relative_path: str
    goals_key: str
    category: str
    start_position: Tuple[float, float, float]
    start_rotation_wxyz: Tuple[float, float, float, float]
    published_geodesic_m: Optional[float]
    source_episode_id: str


def _read_json_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _inline_shards(index, index_file):
    """An index file that carries the episodes itself (val_mini), cut into one shard per scene.

    Returns:
        ``[(scene, index_file, payload)]`` in first-appearance order; each
        payload holds that scene's rows, in file order, and its goal lists.
    """
    rows_by_scene = {}
    for row in index["episodes"]:
        scene = Path(str(row.get("scene_id", ""))).name[:-len(".glb")]
        rows_by_scene.setdefault(scene, []).append(row)
    goals = index.get("goals_by_category")
    shards = []
    for scene, rows in rows_by_scene.items():
        payload = {"episodes": rows}
        if goals is not None:
            payload["goals_by_category"] = {k: v for k, v in goals.items() if k.startswith(scene + ".glb_")}
        shards.append((scene, index_file, payload))
    return shards


def _episode(row, index, stem, scene_root_name):
    """One validated row. ``stem`` is the content shard's own scene name."""
    category = row.get("object_category")
    if category is None and row.get("goals"):
        category = (row["goals"][0] or {}).get("object_category")
    if category not in CATEGORIES:
        raise ValueError("Unknown MP3D object_category %r; expected one of %r"
                         % (category, CATEGORIES))
    scene_id = row.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id.strip():
        raise ValueError("Episode row has no scene_id")
    relative = (scene_id[len(SCENE_PATH_PREFIX):]
                if scene_id.startswith(SCENE_PATH_PREFIX) else scene_id)
    parts = Path(relative).parts
    if Path(relative).is_absolute() or ".." in parts or len(parts) != 3:
        raise ValueError("Unexpected MP3D scene_id %r; expected "
                         "<release>/<scene>/<scene>.glb" % (scene_id,))
    if parts[0] != scene_root_name:
        raise ValueError(
            "Episode row points at scene release %r, but this protocol's scenes "
            "live under %r" % (parts[0], scene_root_name))
    scene_key = parts[1]
    if scene_key != stem or Path(relative).name != stem + ".glb":
        raise ValueError("The %s shard holds an episode of scene %r (%s)"
                         % (stem, scene_key, relative))
    geodesic = (row.get("info") or {}).get("geodesic_distance")
    if geodesic is not None:
        geodesic = float(geodesic)
        if not np.isfinite(geodesic) or geodesic < 0:
            raise ValueError("Published geodesic_distance is not a length: %r" % (geodesic,))
    return MP3DEpisode(
        episode_id="%s/%06d" % (scene_key, index),
        scene_key=scene_key,
        scene_relative_path=relative,
        goals_key="%s_%s" % (Path(relative).name, category),
        category=category,
        start_position=_vector(row.get("start_position"), 3, "start_position"),
        start_rotation_wxyz=xyzw_to_wxyz(row.get("start_rotation")),
        published_geodesic_m=geodesic,
        source_episode_id=str(row.get("episode_id", index)))


class MP3DDataset:
    """A validated local MP3D ObjectNav release, episodes and goals together.

    Args:
        episodes_dir: The split directory, e.g. ``.../objectnav/mp3d/v1/val``,
            containing ``val.json.gz`` and ``content/<scene>.json.gz``.
        scenes_dir: The directory ``data/scene_datasets/`` points at, i.e. the
            one containing ``mp3d/<scene>/``. ``None`` is allowed only with
            ``require_scenes=False``.
        protocol: The :class:`~...mp3d.protocol.MP3DProtocol` being run; its
            ``split`` names the file to read.
        full: Require the documented scene and episode counts of the published
            split. ``False`` is only for an explicitly labelled subset run.
        scenes: Restrict to these scene hashes (requires ``full=False``).
        require_scenes: Check that every referenced mesh and navmesh is on
            disk. Only an episode-level check may turn it off; a run may not.
        episodes_per_scene: Keep only the first ``N`` rows of each shard. The
            validation split is 2,195 episodes, but MP3D's **train** split is
            millions of rows, so a development run has to be bounded to be
            expressible at all. A prefix is taken, so an episode keeps the
            identity it would have had in the whole shard.

    Raises:
        FileNotFoundError: A required episode file or scene asset is missing.
        ValueError: The release, a row, or a goal fails validation.
    """

    def __init__(self, episodes_dir, scenes_dir, protocol, *, full=True, scenes=None,
                 require_scenes=True, episodes_per_scene=None):
        if scenes is not None and full:
            raise ValueError("Selecting scenes is a subset run; pass full=False")
        if episodes_per_scene is not None and (type(episodes_per_scene) is not int
                                               or episodes_per_scene <= 0):
            raise ValueError("episodes_per_scene must be a positive integer")
        if episodes_per_scene is not None and full:
            raise ValueError("Capping episodes per scene is a subset run; pass full=False")
        self.episodes_per_scene = episodes_per_scene
        if scenes_dir is None and require_scenes:
            raise ValueError("A run needs --scenes-dir; only an episode-only "
                             "check may omit it")
        self.protocol = protocol
        self.require_scenes = require_scenes
        self.episodes_dir = Path(episodes_dir).expanduser().resolve()
        self.scenes_dir = (Path(scenes_dir).expanduser().resolve()
                           if scenes_dir is not None else None)
        self.selected_scenes = tuple(scenes) if scenes is not None else None
        split = protocol.split
        index_file = self.episodes_dir / ("%s.json.gz" % split)
        if not index_file.is_file():
            raise FileNotFoundError(
                "Missing %s; point --episodes-dir at the split directory that "
                "holds it and content/ (the objectnav_mp3d_v1 release)" % index_file)
        index = _read_json_gz(index_file)
        self.category_to_task_category_id = dict(
            index.get("category_to_task_category_id") or {})
        if self.category_to_task_category_id and set(
                self.category_to_task_category_id) != set(CATEGORIES):
            missing = sorted(set(CATEGORIES) - set(self.category_to_task_category_id))
            extra = sorted(set(self.category_to_task_category_id) - set(CATEGORIES))
            raise ValueError(
                "Published categories are not the 21 MP3D ObjectNav v1 goals: "
                "missing %r, unexpected %r. Our label table would score the wrong "
                "vocabulary." % (missing, extra))
        scene_table = (index.get("category_to_mp3d_category_id")
                       or index.get("category_to_scene_annotation_category_id"))
        if scene_table is not None and set(scene_table) != set(self.category_to_task_category_id):
            raise ValueError("category_to_task_category_id and category_to_mp3d_category_id "
                             "name different categories")
        #: The release's goal -> mpcat40 class index table; the ground-truth oracle checks it (mp3d_categories).
        self.category_to_mp3d_category_id = dict(scene_table or {})
        shards = [(p.name[:-len(".json.gz")], p, None)
                  for p in sorted((self.episodes_dir / "content").glob("*.json.gz"))]
        if not shards and index.get("episodes"):
            # val_mini ships its episodes and goals in <split>.json.gz itself, with no content/.
            shards = _inline_shards(index, index_file)
        if scenes is not None:
            wanted = {str(s) for s in scenes}
            shards = [s for s in shards if s[0] in wanted]
            missing = sorted(wanted - {s[0] for s in shards})
            if missing:
                raise FileNotFoundError("No content shard for: %s" % ", ".join(missing))
        if not shards:
            raise FileNotFoundError("No content/*.json.gz under %s" % self.episodes_dir)
        files = shards
        expected = published_counts(protocol)
        if full and not expected:
            raise ValueError(
                "Only the val split has published counts to check a complete run "
                "against; load %r with full=False and label the result a subset."
                % (protocol.split,))
        if full and expected and len(files) != expected["scenes"]:
            raise ValueError(
                "MP3D %s %s is documented as %d scenes; found %d content shards. "
                "Provision the whole split, or run a labelled subset."
                % (protocol.dataset_version, split, expected["scenes"], len(files)))
        self.episodes: Dict[str, MP3DEpisode] = {}
        self.goals: Dict[str, list] = {}
        self.scene_counts: Dict[str, int] = {}
        self._assets = [index_file]
        for stem, path, payload in files:
            self._load_shard(stem, path, payload)
        if full and expected and len(self.episodes) != expected["episodes"]:
            raise ValueError(
                "MP3D %s %s is documented as %d episodes; found %d."
                % (protocol.dataset_version, split, expected["episodes"],
                   len(self.episodes)))

    def _load_shard(self, stem, path, payload=None):
        """One scene's rows and goals: a ``content/`` file, or ``payload`` cut from an inline index."""
        if payload is None:
            payload = _read_json_gz(path)
        rows = payload.get("episodes")
        if not isinstance(rows, list) or not rows:
            raise ValueError("%s holds no episodes" % path)
        if self.episodes_per_scene is not None:
            rows = rows[:self.episodes_per_scene]
        goals_by_category = payload.get("goals_by_category")
        inline = goals_by_category is None
        if inline:
            goals_by_category = {}
        if path not in self._assets:
            self._assets.append(path)
        for index, row in enumerate(rows):
            episode = _episode(row, index, stem, self.protocol.scene_root_name)
            self.scene_counts[episode.scene_key] = index + 1
            if inline:
                # ObjectNavDatasetV1.dedup_goals: the first row's goals stand
                # for the (scene, category); every later row carries the same.
                goals_by_category.setdefault(episode.goals_key, row.get("goals") or [])
            if episode.goals_key not in goals_by_category:
                raise ValueError("No goals for %s (%s)"
                                 % (episode.episode_id, episode.goals_key))
            if episode.episode_id in self.episodes:
                raise ValueError("Duplicate episode identity %s" % episode.episode_id)
            self.episodes[episode.episode_id] = episode
            if self.require_scenes:
                # Fail on a missing mesh before the run, never at episode 900.
                self.scene_assets(episode)
        for key, goals in goals_by_category.items():
            if not isinstance(goals, list) or not goals:
                raise ValueError("Goal list %r in %s is empty" % (key, path))
            if key in self.goals and self.goals[key] != goals:
                raise ValueError("Two shards disagree about the goals of %r" % key)
            self.goals[key] = goals

    def goals_for(self, episode):
        """The evaluator-only goal rows of ``episode``. Never given to a policy."""
        return self.goals[episode.goals_key]

    def scene_assets(self, episode):
        """The scene mesh and its **published** navmesh, both checked to exist.

        Returns:
            ``(glb_path, navmesh_path)``: ``mp3d/<scene>/<scene>.glb`` and the
            sibling ``<scene>.navmesh`` habitat-sim derives from it.

        Note:
            Whether the *shipped* navmesh is the one actually navigated on is
            the protocol's ``navmesh`` choice, not this method's. This method
            only proves the publisher's files are here.
        """
        if self.scenes_dir is None:
            raise ValueError("This dataset was loaded without scene assets")
        glb = self.scenes_dir / episode.scene_relative_path
        navmesh = glb.with_name(glb.name[:-len(".glb")] + ".navmesh")
        for path in (glb, navmesh):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(
                    "Missing or empty MP3D asset: %s. Matterport3D's Habitat task "
                    "download (mp3d_habitat.zip) ships <scene>.glb beside "
                    "<scene>.navmesh; this loader never generates one." % path)
        if glb not in self._assets:
            self._assets.extend((glb, navmesh))
        return glb, navmesh

    def semantic_assets(self, episode):
        """The scene's annotation files, both checked to exist: ``(<scene>.house, <scene>_semantic.ply)``.

        habitat-sim reads them beside ``<scene>.glb`` for the semantic sensor,
        and renders instance 0 everywhere without saying so when they are
        absent -- so the ground-truth perception checks them before a renderer
        exists. Both ship in ``mp3d_habitat.zip``.

        Raises:
            FileNotFoundError: Either is missing or empty.
            ValueError: This dataset was loaded without a scenes directory.
        """
        glb, _ = self.scene_assets(episode)
        stem = glb.name[:-len(".glb")]
        house, ply = glb.with_name(stem + ".house"), glb.with_name(stem + "_semantic.ply")
        for path in (house, ply):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError("Missing or empty MP3D annotation: %s. mp3d_habitat.zip ships <scene>.house "
                                        "and <scene>_semantic.ply beside <scene>.glb" % path)
        return house, ply

    def semantic_scene_dataset_config(self):
        """The publisher's MP3D scene dataset config, checked to exist (see :data:`SEMANTIC_SCENE_DATASET_CONFIG`).

        Raises:
            FileNotFoundError: Not installed; the message names the URL and the fetch command.
            ValueError: This dataset was loaded without a scenes directory.
        """
        if self.scenes_dir is None:
            raise ValueError("This dataset was loaded without scene assets")
        path = self.scenes_dir / self.protocol.scene_root_name / SEMANTIC_SCENE_DATASET_CONFIG
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                "Missing %s: the semantic sensor needs the publisher's MP3D scene dataset config to register "
                "<scene>_semantic.ply to <scene>.glb (%s; python -m "
                "sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.assets --scenes-dir %s "
                "--fetch-semantic-config)" % (path, SEMANTIC_SCENE_DATASET_CONFIG_URL, self.scenes_dir))
        return path

    def goal_instances(self, scene_key):
        """``{category: frozenset(object_id)}``: the instances the publisher counts as goals in a scene.

        Every goal row's ``object_id`` is the scene's ``.house`` object index,
        the id the semantic sensor renders (``object_name`` is habitat-sim's
        ``<level>_<region>_<object>`` for the same object). Only categories
        with loaded episodes in the scene are listed. The same contract as
        ``HM3DDataset.goal_instances``, over this loader's goal keys.

        Privileged: feeds the ground-truth perception upper bound only, never
        a method's observation.

        Raises:
            KeyError: No episode of ``scene_key`` is loaded.
        """
        keys = {e.goals_key: e.category for e in self.episodes.values() if e.scene_key == scene_key}
        if not keys:
            raise KeyError("No episodes of scene %r are loaded" % (scene_key,))
        return {category: frozenset(int(row["object_id"]) for row in self.goals[key])
                for key, category in sorted(keys.items(), key=lambda item: item[1])}

    def episode_ids(self):
        """Every loaded episode identity, in publisher order."""
        return tuple(self.episodes)

    def manifest(self, *, hash_scenes=True):
        """Hash the episode files and scene assets for run/resume provenance.

        Args:
            hash_scenes: Hash the meshes and navmeshes too. Leaving it on is
                the point -- a swapped scene release is otherwise invisible --
                and costs one pass over a few gigabytes at preflight.
        """
        files = []
        for path in sorted(set(self._assets), key=str):
            entry = {"path": str(path), "bytes": path.stat().st_size}
            if hash_scenes or path.suffix == ".gz":
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                        digest.update(chunk)
                entry["sha256"] = digest.hexdigest()
            files.append(entry)
        return {"release": "MP3D ObjectNav %s %s on %s"
                           % (self.protocol.dataset_version, self.protocol.split,
                              self.protocol.scene_release),
                "dataset_version": self.protocol.dataset_version,
                "split": self.protocol.split,
                "episode_count": len(self.episodes),
                "scene_count": len(self.scene_counts),
                "selected_scenes": list(self.selected_scenes or ()),
                "episodes_per_scene": self.episodes_per_scene,
                "scene_counts": dict(sorted(self.scene_counts.items())),
                "category_to_task_category_id": dict(sorted(
                    self.category_to_task_category_id.items())),
                "scene_assets_present": bool(self.require_scenes),
                "scene_assets_hashed": bool(hash_scenes and self.require_scenes),
                "files": files}
