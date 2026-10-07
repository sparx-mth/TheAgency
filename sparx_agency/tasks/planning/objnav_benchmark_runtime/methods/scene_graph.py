"""Observed-only room geometry, confirmed doors, revisable LLM room labels and the node oracle."""
from __future__ import annotations

from dataclasses import asdict, replace
import math
import numpy as np

from sparx_agency.core.mapping.topology.room_adjacency import room_adjacency
from sparx_agency.core.mapping.topology.room_registry import RoomRegistry
from sparx_agency.core.mapping.topology.room_stats import count_frontier_clusters, link_doors, door_room_pairs
from sparx_agency.core.mapping.topology.room_watershed import segment_rooms_watershed, trail_thresholds, WatershedRoomParams
from sparx_agency.core.mapping.topology.search_node_oracle import ROOM, SearchNode, SearchNodeOracle
from sparx_agency.core.planning.environment import OccupancyGrid2D
from sparx_agency.core.planning.exploration.frontier_ranking import accessible_frontiers
from sparx_agency.core.planning.exploration.object_search_supervisor import RoomFacts
from sparx_agency.core.planning.exploration.room_search_policy import RoomOption
from sparx_agency.core.planning.objnav.errors import ObjNavInternalError
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doors import DOOR_LABELS
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_labels import RevisableRoomLabels

#: How far beyond its footprint an object may stand from room floor and still be the room's.
OBJECT_ROOM_REACH_M = 0.6


def room_near_cell(labels, world, xy, reach_m):
    """The pid of the nearest labelled cell to world ``xy`` within ``reach_m``, or None."""
    gx, gy = world.world_to_grid(*xy)
    radius = max(1, int(math.ceil(float(reach_m) / world.resolution)))
    h, w = labels.shape
    x0, x1 = max(0, gx - radius), min(w, gx + radius + 1)
    y0, y1 = max(0, gy - radius), min(h, gy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    window = labels[y0:y1, x0:x1]
    ys, xs = np.nonzero(window > 0)
    if not len(xs):
        return None
    d2 = (xs + x0 - gx) ** 2 + (ys + y0 - gy) ** 2
    nearest = int(np.argmin(d2))
    if d2[nearest] > radius * radius:
        return None
    return int(window[ys[nearest], xs[nearest]]) - 1


DEFAULT_SEGMENTATION = WatershedRoomParams(
    min_room_separation_m=1.0, min_clearance_m=0.3, min_room_cells=50,
    door_cut_m=0.75, merge_dynamics_m=0.2,
    # A detected door's position is an estimate along the camera ray; the cut
    # goes to the nearest choke within this reach (Hanson 2026-10-04: a door
    # 0.6 m off its doorway split the bedroom around its bed into R0 and R6).
    door_snap_reach_m=0.9,
    # A clearance dip the agent WALKED THROUGH is a threshold (room_watershed.
    # trail_thresholds): carved at the severing choke within this reach, so the
    # room beyond it separates the tick the agent is through, not when its floor
    # has grown a clearance peak of its own (since 2026-10-07).
    threshold_snap_reach_m=0.5)

#: The registry remembers a vanished room's mask for this many updates -- the
#: episode, in effect (500 actions plus the warm-ups) -- bounded by
#: ``REGISTRY_MEMORY_ROOMS`` masks: the Allensville spawn room merged into the
#: hallway at step 10 and re-split at step 44, 34 ticks past the former 10-tick
#: memory, under a new number.
REGISTRY_MEMORY_TICKS = 1000
REGISTRY_MEMORY_ROOMS = 48
#: Trail points closer than this to the last kept one are not kept (turning in
#: place adds nothing to the clearance profile); the trail keeps this much travel.
TRAIL_SPACING_M = 0.15
TRAIL_LENGTH_M = 6.0
#: A threshold already on record within this distance is the same threshold.
THRESHOLD_MATCH_M = 0.6


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

    def __init__(self, client, segmentation=None, label_settings=None, first_pid=0):
        self.registry = RoomRegistry(iou_threshold=0.15, first_pid=first_pid,
                                     memory_ticks=REGISTRY_MEMORY_TICKS, memory_rooms=REGISTRY_MEMORY_ROOMS)
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
        #: The agent's recent positions (world xy, ``TRAIL_SPACING_M`` apart, the
        #: last ``TRAIL_LENGTH_M`` of travel) and the thresholds -- doorways it
        #: walked through, world xy -- read off the clearance along them; sticky
        #: for the floor, carved as cuts on every update (``threshold_events``).
        self.trail = []
        self.thresholds = []
        self.threshold_events = []
        self._trail_grown = False
        #: Optional ``class_name -> bool``: whether an object class is a home object of the
        #: episode's target (``room_priors.home_object``); sets :attr:`SearchNode.home` on the
        #: nodes handed to the oracle, which reads such a room at its ``home_floor`` at least.
        self.home_object = None
        #: Optional ``world -> (H, W) bool`` naming the unknown cells the search has
        #: settled (:class:`~sightlines.SightLedger.resolved`): looked through without
        #: a return, or enclosed pockets. The frontier is read with them written
        #: OCCUPIED, so no room counts a boundary toward them and no goal is made of
        #: it. None reads the frontier off the map as it is.
        self.resolved_provider = None

    def note_pose(self, xy):
        """Remember where the agent stands, for the walked-through thresholds."""
        point = (float(xy[0]), float(xy[1]))
        if self.trail and math.dist(self.trail[-1], point) < TRAIL_SPACING_M:
            return
        self.trail.append(point)
        self._trail_grown = True
        travelled = 0.0
        keep = len(self.trail)
        for i in range(len(self.trail) - 1, 0, -1):
            travelled += math.dist(self.trail[i - 1], self.trail[i])
            if travelled > TRAIL_LENGTH_M:
                keep = len(self.trail) - i + 1
                break
        if keep < len(self.trail):
            del self.trail[:len(self.trail) - keep]

    def _refresh_thresholds(self, world, free, step):
        """Add every new clearance dip the trail walked through to the floor's thresholds."""
        if not self._trail_grown or self.segmentation.threshold_snap_reach_m <= 0.0 or len(self.trail) < 3:
            return
        self._trail_grown = False
        cells = [world.world_to_grid(x, y) for x, y in self.trail]
        for cx, cy in trail_thresholds(free, world.resolution, cells, self.segmentation):
            xy = world.grid_to_world(int(cx), int(cy))
            if any(math.dist(xy, known) <= THRESHOLD_MATCH_M for known in self.thresholds):
                continue
            self.thresholds.append((float(xy[0]), float(xy[1])))
            self.threshold_events.append({"step": int(step), "xy": [round(float(v), 2) for v in xy]})

    def frontier_world(self, world):
        """The map the frontier is read from: ``world`` with the settled unknown written OCCUPIED, or ``world``."""
        if self.resolved_provider is None:
            return world
        resolved = self.resolved_provider(world)
        if resolved is None or not np.any(resolved):
            return world
        data = world.grid.copy()
        data[np.asarray(resolved, dtype=bool)] = world.values.occupied
        return OccupancyGrid2D(data, world.params, values=world.values)

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

    def room_near(self, world, xy, reach_m):
        """The pid of the room at ``xy``, else of the nearest room floor within ``reach_m``, else None.

        For objects, not the agent: a bed, a wardrobe or a desk stands on
        cells the map reads OCCUPIED, which the watershed never labels, so
        the room's own furniture read as belonging to no room (Hanson
        2026-10-04: the bed and the desks of R0/R6 were ``room: null`` and
        the bedroom was named by its one chair). The nearest labelled cell
        within the object's footprint plus a step is the room it stands in.
        """
        pid = self.room_at(world, xy)
        if pid is not None or self.labels is None:
            return pid
        return room_near_cell(self.labels, world, xy, reach_m)

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
               cost=None, here_xy=None, yaw=0.0, ranking=None, exclude=None, preferred_cost=None):
        """Re-segment the floor into rooms and refresh every per-room fact.

        Args:
            exclude: Optional ``(H, W)`` bool -- cells that are NOT room floor
                whatever the map says: the footprint of a seen staircase. A
                stair is a connector between storeys, not a room; left in,
                the watershed carved a "room" on the landing that the search
                then valued, peeked and waited on. Excluded cells belong to
                no room, carry no label and credit no frontier to anybody.
            preferred_cost: Optional cost array at the planner's preferred
                standoff, for distances measured the way A* will fly them
                (:func:`~frontier_ranking.accessible_frontiers`).
        """
        doors = tuple(doors)
        cells = [world.world_to_grid(*door.xy) for door in doors]
        free = world.grid == world.values.free
        if exclude is not None:
            free = free & ~np.asarray(exclude, dtype=bool)
        if here_xy is not None:
            self.note_pose(here_xy)
        self._refresh_thresholds(world, free, step)
        threshold_cells = [world.world_to_grid(x, y) for x, y in self.thresholds]
        _, _, stats = segment_rooms_watershed(free, world.resolution, self.segmentation, door_cells=cells,
                                              threshold_cells=threshold_cells)
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
        frontier_world = self.frontier_world(world)
        counts = count_frontier_clusters(frontier_world.grid, pid_labels, min_cluster_cells=4)
        self.facts = {pid: RoomFacts(pid, counts.get(pid + 1, 0), self.searched.get(pid, 0.0), room.n_cells)
                      for pid, room in rooms.items()}
        self.frontier_inventory = None
        if cost is not None and here_xy is not None:
            self.refresh_accessibility(world, cost, here_xy, yaw, ranking, preferred_cost, frontier_world=frontier_world)
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

    def refresh_accessibility(self, world, cost, here_xy, yaw=0.0, ranking=None, preferred_cost=None,
                              frontier_world=None):
        """One source of truth for accessible counts, planning goals and map markers."""
        if self.labels is None:
            return
        if frontier_world is None:
            frontier_world = self.frontier_world(world)
        self.frontier_inventory = accessible_frontiers(world, cost, self.labels, here_xy, yaw, ranking,
                                                       preferred_cost=preferred_cost,
                                                       frontier_world=None if frontier_world is world else frontier_world)
        self.facts = {pid: replace(fact, frontier_clusters=len(self.frontier_inventory.by_room.get(pid + 1, ())))
                      for pid, fact in self.facts.items()}

    def refresh_labels(self, step):
        """Re-read every room's label from its evidence now, before the nodes are chosen.

        The label tracker is otherwise updated inside :meth:`reason`, AFTER
        the loop has decided which rooms are nodes: a room whose evidence
        made it a strong bedroom on this very action was shown to the
        oracle and kept in the order for one more loop point (Hanson
        2026-10-05, action 150: a "bedroom, fully seen, no toilet" valued
        at 0.10 instead of excluded). One bounded classifier call per room
        with a new kind of object, as the mid-visit clue rule spends.
        """
        if not self.registry.rooms or not self._objects:
            return dict(self.label_tracker.labels)
        labels = self.label_tracker.update(self._objects, step, False)
        self.options = [replace(o, label=labels[o.room_id].label if o.room_id in labels else o.label)
                        for o in self.options]
        return labels

    def reason(self, world, target, step, extra_nodes=(), context=None, here_xy=None, action_time_s=1.0,
               exclude=()):
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
            exclude: Room pids NOT shown to the model -- rooms the search has
                finished or ruled out by type. They keep a probability of 0.0
                so nothing downstream reads them as unvalued; the storey
                summary still names their types.
        """
        if self.registry.rooms:
            self._reason(world, self._objects, target, step, False, extra_nodes, context, here_xy, action_time_s,
                         exclude=exclude)

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
        """``{pid: [class, ...]}`` -- every confirmed landmark credited to the room it stands in.

        The landmark's cell when it is room floor; otherwise the nearest
        room floor within its footprint radius plus :data:`OBJECT_ROOM_REACH_M`
        (furniture occupies its own cells, see :meth:`room_near`).
        """
        objects = {pid: [] for pid in self.registry.rooms}
        for landmark in landmarks:
            if landmark.class_name in DOOR_LABELS:
                continue
            pid = self.object_room(world, landmark, pid_labels)
            if pid is not None and pid in objects:
                objects[pid].append(landmark.class_name)
        return objects

    def object_room(self, world, landmark, pid_labels=None):
        """The pid of the room a landmark stands in (its cell, else the nearest floor within its footprint + reach)."""
        labels = self.labels if pid_labels is None else pid_labels
        if labels is None:
            return None
        gx, gy = world.world_to_grid(*landmark.xy)
        if world.in_bounds(gx, gy) and labels[gy, gx] > 0:
            return int(labels[gy, gx]) - 1
        radius = float(getattr(landmark, "radius_m", None) or 0.0)
        return room_near_cell(labels, world, landmark.xy, radius + OBJECT_ROOM_REACH_M)

    def nodes(self, world, step, here_xy=None, action_time_s=1.0, exclude=()):
        """Every room as a :class:`SearchNode`, with the facts the oracle values it by."""
        labels = self.label_tracker.labels
        metadata = self.label_tracker.metadata
        here = self.room_at(world, here_xy) if here_xy is not None else None
        excluded = set(int(pid) for pid in exclude)
        is_home = getattr(self, "home_object", None)
        out = []
        for pid, room in self.registry.rooms.items():
            if pid in excluded:
                continue
            fact = self.facts.get(pid)
            inside = self.last_inside.get(pid)
            objects = tuple(self._objects.get(pid, ()))
            out.append(SearchNode(
                id=pid, kind=ROOM, label=labels[pid].label if pid in labels else "unknown",
                tentative=metadata.get(pid, {}).get("strength") == "weak",
                area_m2=room.n_cells * world.resolution ** 2,
                frontier_clusters=0 if fact is None else int(fact.frontier_clusters),
                searched_s=float(self.searched.get(pid, 0.0)),
                last_inside_ago_s=None if inside is None else max(0.0, (int(step) - inside) * action_time_s),
                here=(pid == here), objects=objects,
                home=bool(is_home is not None and any(is_home(name) for name in objects))))
        return out

    def _reason(self, world, objects, target, step, changed, extra_nodes=(), context=None, here_xy=None,
                action_time_s=1.0, exclude=()):
        rooms = self.registry.rooms
        try:
            labels = self.label_tracker.update(objects, step, changed)
            nodes = self.nodes(world, step, here_xy, action_time_s, exclude=exclude) + list(extra_nodes)
            if not nodes:
                # Every room finished or ruled out and no staircase to offer:
                # nothing to ask. The probabilities read zero and the loop's
                # fallback carries the search (the stairs, by the explicit rule).
                self.probs = {pid: 0.0 for pid in rooms}
                self.stair_probs = {}
                self.options = [RoomOption(room_id=pid, label=labels[pid].label if pid in labels else "unknown",
                                           prob=0.0, xy=room.centroid) for pid, room in rooms.items()]
                self.last_reasoning = {"labels": {str(pid): item for pid, item in self.label_tracker.metadata.items()},
                                       "oracle": {"probs": {}, "reasons": {}, "p_present": 0.0, "elsewhere": 1.0,
                                                  "source": "no_nodes", "reused": False, "omitted": [], "reading": {}},
                                       "doors": self.doors, "nodes": [], "excluded": sorted(int(p) for p in exclude),
                                       "label_history": list(self.label_tracker.history),
                                       "partition_revision": self.partition_revision}
                self.p_present = 0.0
                return
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
                               "nodes": [asdict(n) for n in nodes], "excluded": sorted(int(p) for p in exclude),
                               "label_history": list(self.label_tracker.history),
                               "partition_revision": self.partition_revision}
