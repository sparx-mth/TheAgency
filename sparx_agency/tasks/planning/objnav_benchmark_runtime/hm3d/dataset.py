"""Read the published HM3D ObjectNav episodes and their evaluator-only goals.

Never pass this object, its rows, or its goals to a ``SearchPolicy``: the goal
view points are exactly the privileged information a zero-shot benchmark
exists to withhold.

Two habitat-lab conventions are reproduced here rather than guessed at, because
each would otherwise be an invisible bug:

* ``ObjectNavDatasetV1.from_json`` **renumbers** every episode
  ``episode_id = str(i)`` within its own content shard, so ids collide across
  scenes. Ours are scene-qualified (``00800-TEEsavR23oF/000042``) and stable.
* ``start_rotation`` is a quaternion in **XYZW** order
  (``habitat.utils.geometry_utils.quaternion_from_coeff`` reads ``coeffs[3]``
  as the real part). The shared Habitat bridge takes WXYZ, so the order is
  converted here, once. Feeding XYZW straight through would start every
  episode facing somewhere else -- and nothing downstream could tell.

Goals live in ``goals_by_category`` keyed by
``f"{basename(scene_id)}_{object_category}"`` (``ObjectGoalNavEpisode.goals_key``),
with the per-episode ``goals`` list emptied by the publisher. A shard that
predates that de-duplication (goals inline on every episode) is read too.

Only local files are read. Nothing is downloaded, generated or snapped.

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

from sparx_agency.core.planning.objnav.labels.datasets.hm3d import CATEGORIES

#: What each published validation split is documented to contain. The loader
#: refuses a release that disagrees rather than quietly scoring a different
#: number of episodes; if a future release genuinely differs, change it here
#: deliberately, and say so in the run's notes.
PUBLISHED_VAL = {
    "v1": {"scenes": 20, "episodes": 2000},
    "v2": {"scenes": 36, "episodes": 1000},
}

def published_counts(protocol):
    """What this protocol's split is documented to contain, or ``{}``.

    Only the validation split is a published benchmark. A development run on
    ``train`` has no count to be complete against, and pretending otherwise is
    how a subset gets quoted as a split.
    """
    if protocol.split != "val":
        return {}
    return PUBLISHED_VAL.get(protocol.dataset_version, {})


#: habitat-lab strips this prefix from a row's ``scene_id`` before joining it
#: to the scenes directory. HM3D rows do not carry it -- they start straight at
#: ``hm3d/`` or ``hm3d_v0.2/`` -- but the strip is what habitat-lab does, so it
#: is reproduced rather than assumed away.
SCENE_PATH_PREFIX = "data/scene_datasets/"


def _vector(value, size, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError("%s must contain %d finite numbers, got %r"
                         % (name, size, value))
    return tuple(float(v) for v in array)


def xyzw_to_wxyz(rotation):
    """Reorder a published XYZW start rotation into the WXYZ the bridge wants.

    Args:
        rotation: Four finite numbers, ``(x, y, z, w)``, as habitat-lab stores
            them.

    Returns:
        ``(w, x, y, z)``.

    Raises:
        ValueError: Not four finite numbers, or not a unit quaternion (which
            would mean the row is not a rotation at all).
    """
    x, y, z, w = _vector(rotation, 4, "start_rotation (XYZW)")
    if not np.isclose(np.linalg.norm((x, y, z, w)), 1.0, atol=1e-4):
        raise ValueError("start_rotation must be a unit quaternion, got %r"
                         % (rotation,))
    return (w, x, y, z)


@dataclass(frozen=True)
class HM3DEpisode:
    """One published episode. Private evaluator/simulator input.

    Attributes:
        episode_id: ``"<scene folder>/<index within the shard>"``, unique
            within (benchmark, split) despite habitat-lab's per-shard
            renumbering.
        scene_key: The scene's folder name, e.g. ``"00800-TEEsavR23oF"``.
        scene_relative_path: The row's ``scene_id`` with habitat-lab's
            ``data/scene_datasets/`` prefix stripped.
        goals_key: The ``goals_by_category`` key this episode's goals live under.
        category: ``object_category``, in the dataset's own spelling.
        start_position: Habitat-frame ``(x, y, z)``, +Y up.
        start_rotation_wxyz: The start rotation, already reordered.
        published_geodesic_m: The row's ``info.geodesic_distance`` -- the
            publisher's own geodesic from this start to the nearest goal **view
            point**, which is the same quantity as ``l``. It is a cross-check,
            never the SPL reference (``l`` is measured at reset on the navmesh
            the episode is actually navigated on), but because the two are
            defined alike, a disagreement is a **defect**, not a difference of
            definition. Verified on the installed splits: a row's
            ``info.geodesic_distance`` is below the straight-line distance to
            the nearest goal object for 731 of 2000 v1 rows -- geodesically
            impossible if it were measured to the object -- and never below the
            straight line to the nearest view point.
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


def _episode(row, index, stem, scene_root_name):
    """One validated row. ``stem`` is the content shard's own scene stem.

    The shard is named for the scene **stem** (``TEEsavR23oF.json.gz``) while
    the scene folder is ``00800-TEEsavR23oF``, so the folder name comes from the
    row's own ``scene_id`` and the stem is used to catch a shard whose rows
    belong to another scene.
    """
    category = row.get("object_category")
    if category not in CATEGORIES:
        raise ValueError("Unknown HM3D object_category %r; expected one of %r"
                         % (category, CATEGORIES))
    scene_id = row.get("scene_id")
    if not isinstance(scene_id, str) or not scene_id.strip():
        raise ValueError("Episode row has no scene_id")
    relative = (scene_id[len(SCENE_PATH_PREFIX):]
                if scene_id.startswith(SCENE_PATH_PREFIX) else scene_id)
    parts = Path(relative).parts
    if Path(relative).is_absolute() or ".." in parts or len(parts) != 4:
        raise ValueError("Unexpected HM3D scene_id %r; expected "
                         "<release>/<split>/<folder>/<stem>.basis.glb" % (scene_id,))
    if parts[0] != scene_root_name:
        raise ValueError(
            "Episode row points at scene release %r, but this protocol is the "
            "one whose scenes live under %r. Running v1 episodes against v0.2 "
            "geometry (or the reverse) silently changes the scenes."
            % (parts[0], scene_root_name))
    scene_key = parts[2]
    if not scene_key.endswith("-" + stem) or Path(relative).name != stem + ".basis.glb":
        raise ValueError("The %s shard holds an episode of scene %r"
                         % (stem, scene_key))
    geodesic = (row.get("info") or {}).get("geodesic_distance")
    if geodesic is not None:
        geodesic = float(geodesic)
        if not np.isfinite(geodesic) or geodesic < 0:
            raise ValueError("Published geodesic_distance is not a length: %r" % (geodesic,))
    return HM3DEpisode(
        episode_id="%s/%06d" % (scene_key, index),
        scene_key=scene_key,
        scene_relative_path=relative,
        goals_key="%s_%s" % (Path(relative).name, category),
        category=category,
        start_position=_vector(row.get("start_position"), 3, "start_position"),
        start_rotation_wxyz=xyzw_to_wxyz(row.get("start_rotation")),
        published_geodesic_m=geodesic,
        source_episode_id=str(row.get("episode_id", index)))


class HM3DDataset:
    """A validated local HM3D ObjectNav release, episodes and goals together.

    Args:
        episodes_dir: The split directory, e.g.
            ``.../objectnav/hm3d/v2/val``, containing ``val.json.gz`` and
            ``content/<scene>.json.gz``.
        scenes_dir: The directory ``data/scene_datasets/`` points at, i.e. the
            one containing ``hm3d/val/<scene folder>/``. ``None`` is allowed
            only with ``require_scenes=False``.
        protocol: The :class:`~...hm3d.protocol.HM3DProtocol` being run; its
            ``dataset_version`` selects the documented published counts and its
            ``split`` names the file to read.
        full: Require the documented scene and episode counts of the published
            split. ``False`` is only for an explicitly labelled subset run.
        scenes: Restrict to these scenes (requires ``full=False``). Either
            spelling works: the folder ``00800-TEEsavR23oF`` or the stem
            ``TEEsavR23oF``.
        require_scenes: Check that every referenced mesh and navmesh is on
            disk. Only an episode-level check (validating the published rows,
            or a unit test) may turn it off; a run may not.
        episodes_per_scene: Keep only the first ``N`` rows of each shard. The
            validation splits are 2000 and 1000 episodes, but HM3D's **train**
            split is roughly seven million -- some 48,000 per scene -- so a
            development run has to be bounded to be expressible at all. A
            prefix is taken, so an episode keeps the identity it would have had
            in the whole shard.

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
                "holds it and content/" % index_file)
        index = _read_json_gz(index_file)
        self.category_to_task_category_id = dict(
            index.get("category_to_task_category_id") or {})
        if self.category_to_task_category_id and set(
                self.category_to_task_category_id) != set(CATEGORIES):
            raise ValueError(
                "Published categories %r are not the six HM3D ObjectNav goals %r"
                % (sorted(self.category_to_task_category_id), sorted(CATEGORIES)))
        files = sorted((self.episodes_dir / "content").glob("*.json.gz"))
        if scenes is not None:
            # A scene may be named either way round: the folder "00800-TEEsavR23oF"
            # a person reads off the disk, or the stem "TEEsavR23oF" the shard is
            # named for.
            stems = {s.split("-", 1)[1] if "-" in s else s for s in scenes}
            files = [p for p in files if p.name[:-len(".json.gz")] in stems]
            missing = sorted(stems - {p.name[:-len(".json.gz")] for p in files})
            if missing:
                raise FileNotFoundError("No content shard for: %s" % ", ".join(missing))
        if not files:
            raise FileNotFoundError("No content/*.json.gz under %s" % self.episodes_dir)
        expected = published_counts(protocol)
        if full and not expected:
            raise ValueError(
                "Only the val split has published counts to check a complete run "
                "against; load %r with full=False and label the result a subset."
                % (protocol.split,))
        if full and expected and len(files) != expected["scenes"]:
            raise ValueError(
                "HM3D %s %s is documented as %d scenes; found %d content shards. "
                "Provision the whole split, or run a labelled subset."
                % (protocol.dataset_version, split, expected["scenes"], len(files)))
        self.episodes: Dict[str, HM3DEpisode] = {}
        self.goals: Dict[str, list] = {}
        self.scene_counts: Dict[str, int] = {}
        self._assets = [index_file]
        for path in files:
            self._load_shard(path)
        if full and expected and len(self.episodes) != expected["episodes"]:
            raise ValueError(
                "HM3D %s %s is documented as %d episodes; found %d."
                % (protocol.dataset_version, split, expected["episodes"],
                   len(self.episodes)))

    def _load_shard(self, path):
        stem = path.name[:-len(".json.gz")]
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
        self._assets.append(path)
        for index, row in enumerate(rows):
            episode = _episode(row, index, stem, self.protocol.scene_root_name)
            self.scene_counts[episode.scene_key] = index + 1
            if inline:
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
            ``(glb_path, navmesh_path)``.

        Raises:
            FileNotFoundError: Either is missing or empty.
            ValueError: This dataset was loaded without a scenes directory.

        Note:
            Whether the *shipped* navmesh is the one actually navigated on is
            the protocol's ``navmesh`` choice, not this method's: habitat-sim
            recomputes it at the agent's radius and height unless told
            otherwise. This method only proves the publisher's files are here.
        """
        if self.scenes_dir is None:
            raise ValueError("This dataset was loaded without scene assets")
        glb = self.scenes_dir / episode.scene_relative_path
        # habitat-sim derives the sibling navmesh as splitext(scene_id)[0] +
        # ".navmesh", which for "<stem>.basis.glb" is "<stem>.basis.navmesh".
        navmesh = glb.with_name(glb.name[:-len(".glb")] + ".navmesh")
        for path in (glb, navmesh):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(
                    "Missing or empty HM3D asset: %s. Provision the published "
                    "scene release; this loader never generates one." % path)
        if glb not in self._assets:
            self._assets.extend((glb, navmesh))
        return glb, navmesh

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
        return {"release": "HM3D ObjectNav %s %s on %s"
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
