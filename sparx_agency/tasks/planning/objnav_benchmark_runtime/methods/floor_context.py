"""Persistent floor-local policy state; global action/solver ledgers stay shared."""
from __future__ import annotations

from sparx_agency.core.mapping.objects.landmarks import ObjectLandmarkMap
from sparx_agency.core.planning.exploration.object_search_supervisor import ObjectSearchSupervisor
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doors import ObservedDoors
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.object_evidence import TargetEvidence
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.oracle_retry import RepairingNodeOracle
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_priors import home_object
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.room_search_loop import RoomSearchLoop
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.scene_graph import ObservedSceneGraph


class FloorContextBank:
    """Floor-qualified object/room IDs and an action clock that pauses off-floor."""

    FIELDS = ("graph", "doors", "landmarks", "target_evidence", "supervisor", "loop",
              "_target_xy", "_target_id", "_target_step", "_last_door_revision",
              "_visited_frontiers", "_last_graph_step", "_last_plan_s",
              "_blocked_since", "_floor_time", "_object_geometry", "_target_floor_id")

    def __init__(self, policy):
        self.policy = policy
        self.active = 0
        self.contexts = {}

    def new(self):
        p = self.policy
        # Room numbers are unique across the building: a new storey's registry
        # starts after the highest pid any storey (including the one just left,
        # saved by ``activate``) has handed out, so "R0" names one room.
        first_pid = max([0] + [state["graph"].registry.next_pid for state in self.contexts.values()
                               if getattr(state.get("graph"), "registry", None) is not None])
        p.graph = ObservedSceneGraph(p.llm_client, label_settings=p.room_label_settings, first_pid=first_pid)
        p.graph.oracle = RepairingNodeOracle(p.llm_client, unexplored_floor=p.loop_settings.unexplored_floor,
                                             unexplored_elsewhere=p.loop_settings.unexplored_elsewhere,
                                             home_floor=p.loop_settings.home_floor)
        # The oracle's ``home`` flag per room: a confirmed home object of THIS target.
        p.graph.home_object = lambda class_name, target=p.target: home_object(target, class_name)
        # The frontier is read with the settled unknown written occupied: what the
        # camera looked through without a return, and the enclosed pockets.
        sight = getattr(p, "sight", None)
        if sight is not None:
            p.graph.resolved_provider = sight.resolved
        p.doors = ObservedDoors(p.door_settings)
        # Classes never merge (since 2026-10-07): a cup on a table, a toilet beside a
        # bathtub, a vase on a cabinet are separate instances whatever their proximity
        # or footprint overlap. Same-class instances are told apart by size: a tight
        # centroid floor plus footprint-disc overlap (``RPTSettings.landmark_*``).
        # The class vote the map used to take (a sofa box on a confirmed bed voting
        # on the bed) stays in the library behind ``class_votes=True``; the room
        # objects, the LLM and the target evidence see each class's own landmarks.
        p.landmarks = ObjectLandmarkMap(dedupe_radius_m=p.settings.landmark_dedupe_radius_m, nearest_match=True,
                                        class_votes=False, footprint_iou=p.settings.landmark_footprint_iou)
        p.target_evidence = TargetEvidence(p.target_settings)
        p.supervisor = ObjectSearchSupervisor(p.supervisor_params, solver=p.solver)
        # The loop mirrors the supervisor: the room in force and the local
        # step counter are facts about THIS floor's search, restored with it.
        p.loop = RoomSearchLoop(p, p.loop_settings)
        p._target_xy = p._target_id = p._target_floor_id = None
        p._object_geometry = {}
        p._target_step = -p.settings.target_memory_steps - 1
        p._last_door_revision = 0
        p._visited_frontiers = []
        p._last_graph_step = -p.settings.graph_period_steps
        p._last_plan_s = p._blocked_since = None
        p._floor_time = 0.0

    def save(self):
        self.contexts[self.active] = {key: getattr(self.policy, key) for key in self.FIELDS}
        return self.contexts

    def activate(self, floor_id):
        if floor_id == self.active:
            return
        self.save()
        self.active = floor_id
        if floor_id in self.contexts:
            for key, value in self.contexts[floor_id].items():
                setattr(self.policy, key, value)
        else:
            self.new()
        # Geometry is revalidated at the new measured pose. Evidence remains,
        # but no old XY route or stale target latch may cross a floor boundary.
        p = self.policy
        p.route_memory.clear("floor_context_changed")
        p._route = p._goal = p._target_id = p._target_xy = p._target_floor_id = None
        p._blocked_since = p._last_pose = None
        p._last_graph_step = -p.settings.graph_period_steps

    def diagnostics(self):
        out = []
        for floor_id, state in self.save().items():
            loop = state["loop"].diagnostics()
            out.append({"floor_id": floor_id, "rooms": len(state["graph"].registry.rooms),
                        "landmarks": len(state["landmarks"]), "llm_queries": state["graph"].queries,
                        "search_time_s": state["_floor_time"], "supervisor": dict(state["supervisor"].stats),
                        "target_evidence": state["target_evidence"].diagnostics(),
                        "room_ids": ["f%d/r%d" % (floor_id, pid) for pid in state["graph"].registry.rooms],
                        # The loop is floor-local; without this the storey the agent left
                        # keeps no record of what it decided there.
                        "room_search_loop": {k: loop[k] for k in ("stats", "events", "estimate_events", "excluded")}})
        return out

