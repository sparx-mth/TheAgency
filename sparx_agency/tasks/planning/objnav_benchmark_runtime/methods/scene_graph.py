"""Observed-only room geometry, confirmed doors and revisable LLM room labels."""
from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np

from sparx_agency.core.mapping.topology.room_adjacency import room_adjacency
from sparx_agency.core.mapping.topology.room_registry import RoomRegistry
from sparx_agency.core.mapping.topology.room_stats import count_frontier_clusters, link_doors, door_room_pairs
from sparx_agency.core.mapping.topology.room_watershed import segment_rooms_watershed, WatershedRoomParams
from sparx_agency.core.mapping.topology.search_oracle import OracleRoom, SearchOracle
from sparx_agency.core.planning.exploration.object_search_supervisor import RoomFacts
from sparx_agency.core.planning.exploration.room_search_policy import RoomOption
from sparx_agency.core.planning.objnav.errors import ObjNavInternalError
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doors import DOOR_LABELS
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_labels import RevisableRoomLabels

DEFAULT_SEGMENTATION = WatershedRoomParams(
    min_room_separation_m=1.0, min_clearance_m=0.3, min_room_cells=50,
    door_cut_m=0.75, merge_dynamics_m=0.2)


class ObservedSceneGraph:
    """No surveyed door positions, GT semantic labels or privileged map inputs."""

    def __init__(self, client, segmentation=None, label_settings=None):
        self.registry = RoomRegistry(iou_threshold=0.15)
        self.label_tracker = RevisableRoomLabels(client, label_settings)
        self.classifier = self.label_tracker.classifier
        self.oracle = SearchOracle(client)
        self.segmentation = segmentation or DEFAULT_SEGMENTATION
        self.options = []
        self.facts = {}
        self.probs = {}
        self.searched = {}
        self.queries = 0
        self.last_reasoning = {}
        self.doors = []
        self._partition = None
        self.partition_revision = 0
        self.max_rooms = 0

    def credit_time(self, world, pose, seconds):
        gx, gy = world.world_to_grid(pose.x, pose.y)
        if world.in_bounds(gx, gy):
            for pid, room in self.registry.rooms.items():
                if room.mask[gy, gx]:
                    self.searched[pid] = self.searched.get(pid, 0.0) + seconds
                    break

    def update(self, world, landmarks, target, doors=(), step=0):
        doors = tuple(doors)
        cells = [world.world_to_grid(*door.xy) for door in doors]
        _, _, stats = segment_rooms_watershed(
            world.grid == world.values.free, world.resolution, self.segmentation,
            door_cells=cells)
        rooms = self.registry.update(stats, world.grid_to_world)
        partition = (tuple(sorted(rooms)), tuple(sorted(door.id for door in doors)))
        changed = self._partition is not None and self._partition != partition
        if changed:
            self.partition_revision += 1
        self._partition = partition
        self.max_rooms = max(self.max_rooms, len(rooms))
        pid_labels = np.zeros(world.grid.shape, dtype=np.int32)
        for pid, room in rooms.items():
            pid_labels[room.mask] = pid + 1
        self._door_links(world, pid_labels, doors, cells)
        counts = count_frontier_clusters(world.grid, pid_labels, min_cluster_cells=4)
        self.facts = {pid: RoomFacts(pid, counts.get(pid + 1, 0),
                                    self.searched.get(pid, 0.0), room.n_cells)
                      for pid, room in rooms.items()}
        objects = self._room_objects(world, pid_labels, landmarks)
        if not rooms:
            self.label_tracker.update({}, step, changed)
            self.last_reasoning = {}
            self.options, self.probs = [], {}
            return
        self._reason(world, objects, target, step, changed)

    def _door_links(self, world, pid_labels, doors, cells):
        cut = int(round(self.segmentation.door_cut_m / world.resolution))
        links = link_doors(pid_labels, cells, cut, cut + 4)
        pairs = door_room_pairs(links, room_adjacency(pid_labels))
        self.doors = [{"index": door.id, "xy": list(door.xy), "discovered": True,
                       "observations": door.count,
                       "rooms": sorted({pid - 1 for pair in adjacent for pid in pair}),
                       "room_pairs": [[a - 1, b - 1] for a, b in adjacent]}
                      for door, adjacent in zip(doors, pairs)]

    def _room_objects(self, world, pid_labels, landmarks):
        objects = {pid: [] for pid in self.registry.rooms}
        for landmark in landmarks:
            if landmark.class_name in DOOR_LABELS:
                continue
            gx, gy = world.world_to_grid(*landmark.xy)
            if world.in_bounds(gx, gy) and pid_labels[gy, gx] > 0:
                objects[int(pid_labels[gy, gx]) - 1].append(landmark.class_name)
        return objects

    def _reason(self, world, objects, target, step, changed):
        rooms = self.registry.rooms
        try:
            labels = self.label_tracker.update(objects, step, changed)
            oracle_rooms = [OracleRoom(
                pid, labels[pid].label, self.searched.get(pid, 0.0),
                self.facts[pid].frontier_clusters, tuple(objects[pid]),
                room.n_cells * world.resolution ** 2) for pid, room in rooms.items()]
            result = self.oracle.probabilities(target.query, oracle_rooms)
        except Exception as exc:
            raise ObjNavInternalError("Room LLM failed: %s" % exc) from exc
        if result.source != "llm" or not all(
                math.isfinite(p) and 0 <= p <= 1 for p in result.probs.values()):
            raise ObjNavInternalError("Room oracle failed; refusing a uniform fallback run")
        self.queries += 1
        self.probs = {pid: prob * result.p_present for pid, prob in result.probs.items()}
        self.options = [RoomOption(room_id=pid, label=labels[pid].label,
                                   prob=result.probs[pid], xy=room.centroid)
                        for pid, room in rooms.items()]
        self.last_reasoning = {"labels": {str(pid): item for pid, item in self.label_tracker.metadata.items()},
                               "oracle": asdict(result), "doors": self.doors,
                               "label_history": list(self.label_tracker.history),
                               "partition_revision": self.partition_revision}
