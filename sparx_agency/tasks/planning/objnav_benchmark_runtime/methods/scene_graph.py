"""Observed-only room geometry, confirmed doors, revisable LLM room labels and the node oracle."""
from __future__ import annotations

from dataclasses import asdict, replace
import math
import numpy as np

from sparx_agency.core.mapping.topology.room_adjacency import room_adjacency
from sparx_agency.core.mapping.topology.room_registry import RoomRegistry
from sparx_agency.core.mapping.topology.room_stats import count_frontier_clusters, link_doors, door_room_pairs
from sparx_agency.core.mapping.topology.room_watershed import segment_rooms_watershed, WatershedRoomParams
from sparx_agency.core.mapping.topology.search_node_oracle import ROOM, SearchNode, SearchNodeOracle
from sparx_agency.core.planning.exploration.frontier_ranking import accessible_frontiers
from sparx_agency.core.planning.exploration.object_search_supervisor import RoomFacts
from sparx_agency.core.planning.exploration.room_search_policy import RoomOption
from sparx_agency.core.planning.objnav.errors import ObjNavInternalError
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doors import DOOR_LABELS
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_labels import RevisableRoomLabels

DEFAULT_SEGMENTATION = WatershedRoomParams(
    min_room_separation_m=1.0, min_clearance_m=0.3, min_room_cells=50,
    door_cut_m=0.75, merge_dynamics_m=0.2)


class ObservedSceneGraph:
    """No surveyed door positions, GT semantic labels or privileged map inputs.

    Attributes:
        labels: ``(H, W)`` int32 room label image from the latest
            :meth:`update` -- ``pid + 1`` inside a room, 0 elsewhere -- or
            None before the first update. The room-search loop reads it every
            action: to confine in-room routes to the room in force and to
            credit frontier clusters to rooms by the same majority vote
            :attr:`facts` counts them with.
        probs: Independent per-room search-success estimates from the last
            oracle round, never renormalised across the candidate nodes.
        stair_probs: The same for the staircase nodes the loop handed the
            oracle beside the rooms, keyed by their node ids.
        p_present: Chance of some node succeeding under the RPT* independence
            approximation; ``1 - p_present`` is the legacy ``elsewhere`` field.
    """

    def __init__(self, client, segmentation=None, label_settings=None):
        self.registry = RoomRegistry(iou_threshold=0.15)
        self.label_tracker = RevisableRoomLabels(client, label_settings)
        self.classifier = self.label_tracker.classifier
        self.oracle = SearchNodeOracle(client)
        self.segmentation = segmentation or DEFAULT_SEGMENTATION
        self.options, self.facts, self.probs, self.searched = [], {}, {}, {}
        self.stair_probs = {}
        self.queries = 0
        self.oracle_reuses = 0
        self.last_reasoning = {}
        self.doors = []
        self.labels = None
        self.frontier_inventory = None
        self._partition = None
        self.partition_revision = 0
        self.max_rooms = 0
        self._objects = {}
        self.p_present = 0.0
        #: ``{pid: step}`` -- the last action the agent's cell lay in the room.
        self.last_inside = {}

    def credit_time(self, world, pose, seconds, step=None):
        gx, gy = world.world_to_grid(pose.x, pose.y)
        if world.in_bounds(gx, gy):
            for pid, room in self.registry.rooms.items():
                if room.mask[gy, gx]:
                    self.searched[pid] = self.searched.get(pid, 0.0) + seconds
                    if step is not None:
                        self.last_inside[pid] = int(step)
                    break

    def room_at(self, world, xy):
        """The pid of the room whose mask holds world ``xy``, or None."""
        gx, gy = world.world_to_grid(*xy)
        if self.labels is None or not world.in_bounds(gx, gy):
            return None
        label = int(self.labels[gy, gx])
        return label - 1 if label > 0 else None

    def objects_in(self, pid):
        """Class names of the confirmed landmarks the latest update placed in room ``pid``."""
        return list(self._objects.get(pid, ()))

    def label_info(self, pid):
        """The room's label record: ``label``, ``confidence``, ``strength``, ``provisional`` -- or None."""
        item = self.label_tracker.metadata.get(pid)
        if item is not None:
            return item
        label = self.label_tracker.labels.get(pid)
        if label is None:
            return None
        return {"label": label.label, "confidence": label.confidence, "reasoning": label.reasoning,
                "strength": "weak", "provisional": True}

    def relabel(self, pid, step):
        """Re-classify ONE room from its latest evidence, if that evidence changed.

        The clue the loop acts on mid-visit: a bed seen from the doorway
        names the room now, not at the next loop point. One bounded model
        call for the one room it matters to; the option list is kept in
        step so the recording shows the new name at once.

        Returns:
            True when the room's label changed.
        """
        before = self.label_tracker.labels.get(pid)
        after = self.label_tracker.query_room(pid, step)
        if after is None:
            return False
        self.options = [replace(o, label=after.label) if o.room_id == pid else o for o in self.options]
        if "labels" in self.last_reasoning:
            self.last_reasoning["labels"][str(pid)] = self.label_tracker.metadata.get(pid, {})
        return before is None or before.label != after.label

    def update(self, world, landmarks, target, doors=(), step=0, reason=True,
               cost=None, here_xy=None, yaw=0.0, ranking=None):
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
        self.labels = pid_labels
        self._door_links(world, pid_labels, doors, cells)
        counts = count_frontier_clusters(world.grid, pid_labels, min_cluster_cells=4)
        self.facts = {pid: RoomFacts(pid, counts.get(pid + 1, 0), self.searched.get(pid, 0.0), room.n_cells)
                      for pid, room in rooms.items()}
        self.frontier_inventory = None
        if cost is not None and here_xy is not None:
            self.refresh_accessibility(world, cost, here_xy, yaw, ranking)
        self._objects = self._room_objects(world, pid_labels, landmarks)
        if not rooms:
            self.label_tracker.update({}, step, changed)
            self.last_reasoning = {}
            self.options, self.probs, self.stair_probs = [], {}, {}
            return
        if reason:
            self._reason(world, self._objects, target, step, changed)
        else:
            labels = self.label_tracker.update(self._objects, step, changed, allow_query=False)
            self.options = [RoomOption(room_id=pid, label=labels[pid].label if pid in labels else "unknown",
                                       prob=self.probs.get(pid, 0.0), xy=room.centroid)
                            for pid, room in rooms.items()]

    def refresh_accessibility(self, world, cost, here_xy, yaw=0.0, ranking=None):
        """One source of truth for accessible counts, planning goals and map markers."""
        if self.labels is None:
            return
        self.frontier_inventory = accessible_frontiers(world, cost, self.labels, here_xy, yaw, ranking)
        self.facts = {pid: replace(fact, frontier_clusters=len(self.frontier_inventory.by_room.get(pid + 1, ())))
                      for pid, fact in self.facts.items()}

    def reason(self, world, target, step, extra_nodes=(), context=None, here_xy=None, action_time_s=1.0):
        """Value every node from the observations so far, without inventing a partition change.

        Args:
            world: The floor's occupancy grid.
            target: The target labels (``query`` is what the model is asked about).
            step: The current action.
            extra_nodes: :class:`SearchNode` s beyond the rooms -- the
                staircases the loop offers -- valued in the same call.
            context: :class:`~search_node_oracle.SearchContext` on the storeys.
            here_xy: The agent's position, to mark the room it stands in.
            action_time_s: Seconds per action, to phrase "last inside N ago".
        """
        if self.registry.rooms:
            self._reason(world, self._objects, target, step, False, extra_nodes, context, here_xy, action_time_s)

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

    def nodes(self, world, step, here_xy=None, action_time_s=1.0):
        """Every room as a :class:`SearchNode`, with the facts the oracle values it by."""
        labels = self.label_tracker.labels
        metadata = self.label_tracker.metadata
        here = self.room_at(world, here_xy) if here_xy is not None else None
        out = []
        for pid, room in self.registry.rooms.items():
            fact = self.facts.get(pid)
            inside = self.last_inside.get(pid)
            out.append(SearchNode(
                id=pid, kind=ROOM, label=labels[pid].label if pid in labels else "unknown",
                tentative=metadata.get(pid, {}).get("strength") == "weak",
                area_m2=room.n_cells * world.resolution ** 2,
                frontier_clusters=0 if fact is None else int(fact.frontier_clusters),
                searched_s=float(self.searched.get(pid, 0.0)),
                last_inside_ago_s=None if inside is None else max(0.0, (int(step) - inside) * action_time_s),
                here=(pid == here), objects=tuple(self._objects.get(pid, ()))))
        return out

    def _reason(self, world, objects, target, step, changed, extra_nodes=(), context=None, here_xy=None,
                action_time_s=1.0):
        rooms = self.registry.rooms
        try:
            labels = self.label_tracker.update(objects, step, changed)
            nodes = self.nodes(world, step, here_xy, action_time_s) + list(extra_nodes)
            result = self.oracle.probabilities(target.query, nodes, context)
        except Exception as exc:
            raise ObjNavInternalError("Room LLM failed: %s" % exc) from exc
        if result.source != "llm" or not all(math.isfinite(p) and 0 <= p <= 1 for p in result.probs.values()):
            raise ObjNavInternalError("Room oracle failed; refusing a uniform fallback run")
        if result.reused:
            self.oracle_reuses += 1
        else:
            self.queries += 1
        self.p_present = float(result.p_present)
        self.probs = {pid: float(result.probs.get(pid, 0.0)) for pid in rooms}
        self.stair_probs = {nid: float(p) for nid, p in result.probs.items() if nid not in rooms}
        self.options = [RoomOption(room_id=pid, label=labels[pid].label if pid in labels else "unknown",
                                   prob=self.probs[pid], xy=room.centroid) for pid, room in rooms.items()]
        self.last_reasoning = {"labels": {str(pid): item for pid, item in self.label_tracker.metadata.items()},
                               "oracle": asdict(result), "doors": self.doors,
                               "nodes": [asdict(n) for n in nodes],
                               "label_history": list(self.label_tracker.history),
                               "partition_revision": self.partition_revision}
