"""The published RoboTHOR episode files, loaded, checked and fingerprinted.

One gzipped JSON list per scene under ``<split>/episodes/<Scene>.json.gz``,
exactly as ``allenai/robothor-challenge`` ships them in-tree and as AllenAct's
``robothor-objectnav-challenge-2021.tar.gz`` extracts them. Each row is a
dict with ``id``, ``scene``, ``object_type``, ``initial_position``,
``initial_orientation``, ``initial_horizon``, ``shortest_path`` and
``shortest_path_length``.

``shortest_path`` and ``shortest_path_length`` are **privileged**: they are
the SPL numerator and they name the goal. They live on the row and reach the
scorer through
:class:`~sparx_agency.core.planning.objnav.types.measurement.EpisodeMeasurement`;
nothing here ever copies a raw dataset row into
:attr:`~sparx_agency.core.planning.objnav.types.episode.ObjNavEpisode.metadata`,
which is how a dataset leaks its own answer.

Everything is checked at load, before the simulator is started: a missing key,
a category the label table does not cover, a scene file whose rows belong to
another scene, a repeated episode id, or a ``shortest_path_length`` that
disagrees with its own corners. The last one is worth the arithmetic -- it is
the published SPL numerator, and a file that has been regenerated or
half-copied is otherwise indistinguishable from the real one.

Python 3.8 syntax, standard library only.
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

from sparx_agency.core.planning.objnav.labels.datasets.robothor import CATEGORIES
from sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.protocol import (
    PROTOCOL, SCENES, VAL_EPISODES, VAL_EPISODES_PER_SCENE,
)

_REQUIRED = ("id", "scene", "object_type", "initial_position",
             "initial_orientation", "initial_horizon")
#: How far a shipped ``shortest_path_length`` may differ from its own corners.
#: The published files agree to about 2e-15; this is generous by 1e10 and
#: still refuses a regenerated or truncated path.
_LENGTH_TOLERANCE_M = 1e-5


def vector_distance(a, b) -> float:
    """AI2-THOR's own 3-D point distance, ``ai2thor.util.metrics``."""
    return math.sqrt(sum((float(a[axis]) - float(b[axis])) ** 2
                         for axis in ("x", "y", "z")))


def path_distance(path) -> float:
    """AI2-THOR's own polyline length, ``ai2thor.util.metrics``."""
    return sum(vector_distance(path[i], path[i + 1])
               for i in range(len(path) - 1))


@dataclass(frozen=True)
class RobothorEpisodeRow:
    """One published episode. ``shortest_path*`` is evaluator-only.

    Attributes:
        episode_id: The publisher's own ``id``, unique across the split and
            the key every result, resume and comparison is filed under.
        scene: The AI2-THOR scene to load.
        category: The goal ``objectType``, verbatim.
        start_position: ``{"x", "y", "z"}``, the published spawn point.
        start_rotation_deg: ``initial_orientation``, AI2-THOR's bearing.
        start_horizon_deg: ``initial_horizon`` -- 30 in every published
            episode, i.e. the camera starts fully pitched down.
        shortest_path: The precomputed ``GetShortestPath`` corners.
            **Privileged.**
        shortest_path_length_m: ``l``, the SPL numerator. **Privileged.**
    """

    episode_id: str
    scene: str
    category: str
    start_position: Dict[str, float]
    start_rotation_deg: float
    start_horizon_deg: float
    shortest_path: Tuple[Dict[str, float], ...]
    shortest_path_length_m: float


class RobothorDataset:
    """Every episode of one published split, in the publisher's order.

    Args:
        episodes_dir: The split directory holding ``episodes/*.json.gz``, or
            the dataset root holding ``<split>/episodes/``.
        split: Which split the directory is expected to be.
        scene: Load one scene only, for a smoke run. A single-scene dataset
            can never be presented as the complete split.
        require_full_split: Check the published scene list and episode counts.
            Left on for a real run; a single-scene load turns it off.

    Raises:
        FileNotFoundError: No episode files under that directory.
        ValueError: A malformed, mislabelled or incomplete split.
    """

    def __init__(self, episodes_dir, *, split=PROTOCOL.split, scene=None,
                 require_full_split=True):
        self.split = split
        self.scene = scene
        self.root = self._resolve(Path(episodes_dir).expanduser(), split)
        paths = sorted(self.root.glob("*.json.gz"))
        if scene is not None:
            paths = [p for p in paths if p.name == scene + ".json.gz"]
            if not paths:
                raise FileNotFoundError(
                    "No episode file for scene %r under %s" % (scene, self.root))
        if not paths:
            raise FileNotFoundError(
                "No <Scene>.json.gz episode files under %s. Clone "
                "allenai/robothor-challenge, or extract "
                "robothor-objectnav-challenge-2021.tar.gz" % (self.root,))
        self.files = tuple(paths)
        self.episodes = collections.OrderedDict()
        self._digests = {}
        for path in paths:
            self._load_scene(path)
        self.zero_length_episode_ids = tuple(
            row.episode_id for row in self.episodes.values()
            if row.shortest_path_length_m == 0.0)
        if require_full_split and scene is None:
            self._check_full_split()

    @staticmethod
    def _resolve(directory, split) -> Path:
        """Accept either the split directory or the dataset root."""
        for candidate in (directory / "episodes", directory / split / "episodes",
                          directory):
            if candidate.is_dir() and any(candidate.glob("*.json.gz")):
                return candidate
        raise FileNotFoundError(
            "Expected <root>/%s/episodes/*.json.gz under %s" % (split, directory))

    def _load_scene(self, path) -> None:
        raw = path.read_bytes()
        self._digests[path.name] = hashlib.sha256(raw).hexdigest()
        rows = json.loads(gzip.decompress(raw).decode("utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError("%s does not hold a non-empty list of episodes" % path)
        expected_scene = path.name[: -len(".json.gz")]
        for row in rows:
            self._add(path, expected_scene, row)

    def _add(self, path, expected_scene, row) -> None:
        if not isinstance(row, dict):
            raise ValueError("%s holds a %s where an episode was expected"
                             % (path, type(row).__name__))
        missing = [key for key in _REQUIRED if key not in row]
        if missing:
            raise ValueError("%s: episode is missing %s"
                             % (path, ", ".join(missing)))
        if row["scene"] != expected_scene:
            raise ValueError(
                "%s holds an episode of scene %r; an episode file must hold "
                "only its own scene" % (path, row["scene"]))
        if row["object_type"] not in CATEGORIES:
            raise ValueError(
                "%s: %r is not one of the 12 RoboTHOR challenge targets"
                % (path, row["object_type"]))
        episode_id = row["id"]
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("%s: episode id must be a non-blank string" % path)
        if episode_id in self.episodes:
            raise ValueError(
                "Episode id %r appears twice; ids must be unique within the "
                "split or results, resumes and paired comparisons collide"
                % (episode_id,))
        corners = tuple(dict(point) for point in row.get("shortest_path", ()))
        length = self._checked_length(path, episode_id, corners, row)
        self.episodes[episode_id] = RobothorEpisodeRow(
            episode_id=episode_id, scene=row["scene"],
            category=row["object_type"],
            start_position={axis: float(row["initial_position"][axis])
                            for axis in ("x", "y", "z")},
            start_rotation_deg=float(row["initial_orientation"]),
            start_horizon_deg=float(row["initial_horizon"]),
            shortest_path=corners, shortest_path_length_m=length)

    @staticmethod
    def _checked_length(path, episode_id, corners, row) -> float:
        """``l``, checked against the corners it is supposed to measure."""
        if "shortest_path_length" not in row:
            raise ValueError(
                "%s: %s has no shortest_path_length. The test split ships "
                "without it and cannot be scored here" % (path, episode_id))
        if not corners:
            raise ValueError("%s: %s has an empty shortest_path"
                             % (path, episode_id))
        length = float(row["shortest_path_length"])
        if not math.isfinite(length) or length < 0:
            raise ValueError("%s: %s has shortest_path_length %r"
                             % (path, episode_id, length))
        measured = path_distance(corners)
        if abs(measured - length) > _LENGTH_TOLERANCE_M:
            raise ValueError(
                "%s: %s ships shortest_path_length %.6f but its own corners "
                "measure %.6f. This is the published SPL numerator; the file "
                "has been regenerated or truncated"
                % (path, episode_id, length, measured))
        return length

    def _check_full_split(self) -> None:
        """Refuse to call an incomplete directory the published split."""
        scenes = collections.Counter(row.scene for row in self.episodes.values())
        if self.split == PROTOCOL.split:
            problems = []
            if tuple(sorted(scenes)) != tuple(sorted(SCENES)):
                problems.append("scenes %s" % ", ".join(sorted(scenes)))
            if len(self.episodes) != VAL_EPISODES:
                problems.append("%d episodes" % len(self.episodes))
            odd = {s: n for s, n in scenes.items() if n != VAL_EPISODES_PER_SCENE}
            if odd:
                problems.append("per-scene counts %r" % (odd,))
            if problems:
                raise ValueError(
                    "This is not the published RoboTHOR validation split "
                    "(%d scenes x %d episodes = %d): %s"
                    % (len(SCENES), VAL_EPISODES_PER_SCENE, VAL_EPISODES,
                       "; ".join(problems)))

    def categories(self) -> Tuple[str, ...]:
        """The goal categories present, in the published category order."""
        present = {row.category for row in self.episodes.values()}
        return tuple(c for c in CATEGORIES if c in present)

    def manifest(self) -> dict:
        """What was actually loaded, for the run manifest. No privileged values."""
        scenes = collections.Counter(row.scene for row in self.episodes.values())
        categories = collections.Counter(row.category
                                         for row in self.episodes.values())
        return {
            "dataset": "robothor-objectnav-challenge-2021",
            "split": self.split,
            "episodes_dir": str(self.root),
            "episode_files_sha256": dict(sorted(self._digests.items())),
            "n_episodes": len(self.episodes),
            "scenes": dict(sorted(scenes.items())),
            "categories": dict(sorted(categories.items())),
            "single_scene": self.scene,
            "zero_shortest_path_episodes": len(self.zero_length_episode_ids),
        }
