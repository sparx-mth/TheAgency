"""Explicit generated training-development data, never published Gibson validation.

Ground-truth maps and generated starts are evaluator-only. Reuse the existing
floor validation, FMM scorer, RGB-D bridge and action protocol without passing
this dataset object to the policy.
"""
from __future__ import annotations

import bz2
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.dataset import GibsonDataset, _ArrayUnpickler, _episode
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.env import GibsonEnv
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL, SCENES

SCHEMA = "sparx-gibson-generated-train/1"
SPLIT = "train-development"
DEVELOPMENT_PROTOCOL = replace(PROTOCOL, protocol_id=SCHEMA, split=SPLIT)


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def load_training_maps(path):
    with bz2.open(Path(path).expanduser(), "rb") as stream:
        maps = _ArrayUnpickler(stream).load()
    if not isinstance(maps, dict) or not maps:
        raise ValueError("Training semantic maps must contain scene dictionaries")
    for scene, floors in maps.items():
        if not isinstance(scene, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", scene) or scene in SCENES:
            raise ValueError("Not a separate Gibson training scene: %r" % scene)
        if not isinstance(floors, dict) or not floors:
            raise ValueError("Training scene has no floor maps: " + scene)
    return maps


class DevelopmentDataset(GibsonDataset):
    """Load an immutable generated manifest; never interpret PointNav placeholders."""

    def __init__(self, manifest, scene=None):
        self.manifest_path = Path(manifest).expanduser().resolve()
        data = json.loads(self.manifest_path.read_text())
        if data.get("schema") != SCHEMA or data.get("split") != SPLIT:
            raise ValueError("Not a generated Gibson training-development manifest")
        scenes = data["scenes"]
        if len(scenes) != len(set(scenes)) or (scene is not None and scene not in scenes):
            raise ValueError("Duplicate or unlisted development scene")
        map_path = Path(data["train_info"]["path"])
        if file_identity(map_path) != data["train_info"]:
            raise ValueError("Training maps changed since episode generation")
        self._maps = load_training_maps(map_path)
        if not set(scenes).issubset(self._maps):
            raise ValueError("Manifest contains scenes absent from training maps")
        self.scenes_dir = Path(data["scenes_dir"])
        self.episodes_dir = self.manifest_path.parent
        self.selected_scene = scene
        self.episodes, self.scene_counts = {}, {}
        self._validated_floors = set()
        self._assets = [map_path, self.manifest_path]
        identities = {item["path"]: item for item in data["assets"]}
        for name in scenes:
            if scene is not None and scene != name:
                continue
            rows = [row for row in data["episodes"] if row["scene"] == name]
            if not rows:
                raise ValueError("No generated ObjectNav episodes for " + name)
            for suffix in (".glb", ".navmesh"):
                path = (self.scenes_dir / (name + suffix)).resolve()
                if file_identity(path) != identities.get(str(path)):
                    raise ValueError("Scene asset changed since generation: " + str(path))
                self._assets.append(path)
            self.scene_counts[name] = len(rows)
            for index, row in enumerate(rows):
                episode = _episode(row, name, index)
                self.floor(episode)
                self.episodes[episode.episode_id] = episode
        self.definition = data

    def manifest(self):
        result = super().manifest()
        result.update(release="SemExp Gibson TRAIN maps; generated ObjectNav development starts",
                      split=SPLIT, episode_source="generated-not-published-validation",
                      generation=self.definition["generation"], held_out_claim=False)
        return result


class DevelopmentEnv(GibsonEnv):
    """Same evaluator/action behavior, explicitly separate public split identity."""

    name = "habitat/gibson-generated-training-development"

    def reset(self, episode_id):
        episode, observation = super().reset(episode_id)
        self._episode = replace(episode, split=SPLIT)
        return self._episode, observation

