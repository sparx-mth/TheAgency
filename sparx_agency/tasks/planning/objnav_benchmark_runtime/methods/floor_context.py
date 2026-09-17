"""Persistent floor-local policy state; global action/solver ledgers stay shared."""
from __future__ import annotations

from sparx_agency.core.mapping.objects.landmarks import ObjectLandmarkMap
from sparx_agency.core.planning.exploration.object_search_supervisor import ObjectSearchSupervisor
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.doors import ObservedDoors
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.object_evidence import TargetEvidence
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.oracle_retry import RepairingSearchOracle
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.scene_graph import ObservedSceneGraph


class FloorContextBank:
    """Floor-qualified object/room IDs and an action clock that pauses off-floor."""

    FIELDS = ("graph", "doors", "landmarks", "target_evidence", "supervisor",
              "_target_xy", "_target_id", "_target_step", "_last_door_revision",
              "_visited_frontiers", "_last_graph_step", "_last_plan_s",
              "_blocked_since", "_floor_time")

    def __init__(self, policy):
        self.policy = policy
        self.active = 0
        self.contexts = {}

    def new(self):
        p = self.policy
        p.graph = ObservedSceneGraph(p.llm_client, label_settings=p.room_label_settings)
        p.graph.oracle = RepairingSearchOracle(p.llm_client)
        p.doors = ObservedDoors(p.door_settings)
        p.landmarks = ObjectLandmarkMap(nearest_match=True)
        p.target_evidence = TargetEvidence(p.target_settings)
        p.supervisor = ObjectSearchSupervisor(p.supervisor_params, solver=p.solver)
        p._target_xy = p._target_id = None
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
        p._route = p._goal = p._target_id = p._target_xy = None
        p._blocked_since = p._last_pose = None
        p._last_graph_step = -p.settings.graph_period_steps

    def diagnostics(self):
        return [{"floor_id": floor_id, "rooms": len(state["graph"].registry.rooms),
                 "landmarks": len(state["landmarks"]), "llm_queries": state["graph"].queries,
                 "search_time_s": state["_floor_time"], "supervisor": dict(state["supervisor"].stats),
                 "target_evidence": state["target_evidence"].diagnostics(),
                 "room_ids": ["f%d/r%d" % (floor_id, pid) for pid in state["graph"].registry.rooms]}
                for floor_id, state in self.save().items()]

