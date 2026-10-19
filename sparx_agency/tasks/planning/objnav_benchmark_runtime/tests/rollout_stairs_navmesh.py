"""Headless climb of every ground-truth staircase on a scene's own navmesh, by the real traversal and converter.

Not a unit test: a development probe (and, with ``habitat_sim`` present, a
regression) for the one promise the stair work has to keep -- **once the
decision to take a staircase is made, the climb completes**. For each
connector of each scene, both ways, a point agent is put at the storey
anchor, a :class:`GroundTruthTraversal` is committed on it, and every action
goes through the same :class:`DiscreteActionConverter` the benchmark agent
uses. MOVE_FORWARD is the simulator's own collision model --
``PathFinder.try_step`` with sliding, exactly what ``habitat_sim`` does to
the agent -- so a banister the route grazes blocks the step here as it does
in the recording. The climb has succeeded when the atlas confirms the other
storey; the probe reports the actions it took, the blocked steps, the
recoveries and any retreat.

Run from the repo root in the Habitat environment::

    ~/miniconda3/envs/habitat/bin/python -m \
        sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.rollout_stairs_navmesh \
        --scenes-dir ~/datasets/gibson/development-15-20260915/scenes [--scene Pomaria] [--verbose]

Exit status 1 when any climb failed, so a campaign script can gate on it.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional

import numpy as np

from sparx_agency.core.planning.exploration.floor_atlas import FloorAtlas, MultiFloorParams
from sparx_agency.core.planning.objnav.action_converter.converter import DiscreteActionConverter
from sparx_agency.core.planning.objnav.action_converter.params import ActionConverterParams
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.stair_connectors import (
    enu_to_habitat, habitat_to_enu, scene_structure_from_pathfinder)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.ground_truth_traversal import GroundTruthTraversal
from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.stair_ground_truth import GroundTruthStairs

#: Actions a climb may take before the probe calls it a failure (a flight is ~30 actions).
BUDGET = 160


class PointAgent:
    """A body of the benchmark's geometry moved by the navmesh's own collision model."""

    def __init__(self, pathfinder, spec, position_enu, yaw):
        self.pathfinder, self.spec = pathfinder, spec
        self.position = np.asarray(enu_to_habitat(position_enu), dtype=np.float32)
        self.yaw = float(yaw)

    @property
    def pose(self) -> AgentPose:
        x, y, z = habitat_to_enu(self.position)
        return AgentPose(x, y, z, self.yaw, 0.0)

    def act(self, action: DiscreteAction) -> None:
        if action == DiscreteAction.TURN_LEFT:
            self.yaw = math.atan2(math.sin(self.yaw + self.spec.turn_angle_rad), math.cos(self.yaw + self.spec.turn_angle_rad))
        elif action == DiscreteAction.TURN_RIGHT:
            self.yaw = math.atan2(math.sin(self.yaw - self.spec.turn_angle_rad), math.cos(self.yaw - self.spec.turn_angle_rad))
        elif action == DiscreteAction.MOVE_FORWARD:
            # ENU forward (cos yaw, sin yaw) is Habitat (-sin yaw, 0, -cos yaw).
            step = self.spec.forward_step_m * np.asarray([-math.sin(self.yaw), 0.0, -math.cos(self.yaw)], dtype=np.float32)
            self.position = np.asarray(self.pathfinder.try_step(self.position, self.position + step), dtype=np.float32)


def fake_coordinator(portal: Dict, params: MultiFloorParams, atlas: FloorAtlas):
    policy = SimpleNamespace(mapping=SimpleNamespace(atlas=atlas), _route=None, _goal=None)
    return SimpleNamespace(policy=policy, params=params, active=portal, floor_id=0, events=[])


def climb(pathfinder, connector, direction: int, params: MultiFloorParams, verbose=False) -> Dict:
    """One committed climb of ``connector`` in ``direction`` (+1 up, -1 down); the outcome as a dict."""
    spec = PROTOCOL.actions()
    height = connector.bottom_z if direction > 0 else connector.top_z
    direction, entry, _, destination_z, path = connector.oriented(height, params.floor_match_m)
    # Face along the first leg, as an approach that arrived at the anchor would.
    yaw = math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0])
    agent = PointAgent(pathfinder, spec, entry, yaw)
    atlas = FloorAtlas(params)
    atlas.update(agent.pose)
    portal = {"id": connector.id, "floor_id": 0, "entry": list(entry), "direction": direction,
              "path": [tuple(p) for p in path], "destination_z": destination_z, "connector_id": connector.id}
    coordinator = fake_coordinator(portal, params, atlas)
    obs = SimpleNamespace(pose=agent.pose, step=0)
    traversal = GroundTruthTraversal(coordinator, obs)
    converter = DiscreteActionConverter(spec, ActionConverterParams())
    outcome = {"scene": None, "connector": connector.id, "direction": "up" if direction > 0 else "down",
               "actions": 0, "blocked": 0, "recoveries": 0, "retreat": False, "arrived": False, "forced": False,
               "final_height": None, "destination_z": destination_z, "phases": []}
    for step in range(1, BUDGET + 1):
        pose = agent.pose
        obs = SimpleNamespace(pose=pose, step=step)
        if converter.forward_blocked(pose):
            traversal.blocked(obs)
            outcome["blocked"] += 1
        traversal.observe(obs)
        atlas.update(pose, arrival_allowed=traversal.arrival_allowed)
        if atlas.active_id != 0:
            outcome.update(arrived=True, actions=step, final_height=pose.z, forced=traversal.forced)
            break
        command = traversal.plan(obs)
        if traversal.forced and atlas.active_id != 0:
            outcome.update(arrived=True, actions=step, final_height=pose.z, forced=True)
            break
        result = converter.step(pose, command)
        action = result.action if result.action is not None else DiscreteAction.TURN_LEFT
        if verbose:
            print("   %3d %-12s blk=%d xyz=(%.2f,%.2f,%.2f) yaw=%6.1f cur=%d/%d tight=%d rec=%d %s" % (
                step, action.name, int(result.forward_blocked), pose.x, pose.y, pose.z, math.degrees(pose.yaw),
                traversal.cursor, len(traversal.route), int(traversal.tight), int(traversal._recovery is not None),
                traversal.phase))
        agent.act(action)
        outcome["actions"] = step
        if traversal.phase == "RETREAT":
            outcome["retreat"] = True
    outcome["recoveries"] = traversal.recoveries
    outcome["final_height"] = agent.pose.z
    outcome["failures"] = traversal.failures
    outcome["phase"] = traversal.phase
    return outcome


def run_scene(navmesh_path: str, verbose=False, only_direction: Optional[int] = None) -> List[Dict]:
    import habitat_sim

    pathfinder = habitat_sim.PathFinder()
    if not pathfinder.load_nav_mesh(navmesh_path):
        raise RuntimeError("cannot load %s" % navmesh_path)
    structure = scene_structure_from_pathfinder(pathfinder)
    stairs = GroundTruthStairs.from_metadata(structure)
    params = MultiFloorParams()
    scene = os.path.basename(navmesh_path).rsplit(".", 1)[0]
    outcomes = []
    for connector in stairs.connectors:
        for direction in (1, -1):
            if only_direction is not None and direction != only_direction:
                continue
            if verbose:
                print("== %s connector %d %s" % (scene, connector.id, "up" if direction > 0 else "down"))
            outcome = climb(pathfinder, connector, direction, params, verbose)
            outcome["scene"] = scene
            outcome["walkability"] = structure["stair_connectors"][connector.id].get("walkability")
            outcome["traversable"] = connector.traversable
            outcomes.append(outcome)
    return outcomes


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenes-dir", default=os.environ.get("GIBSON_SCENES_DIR"), required=False)
    parser.add_argument("--scene", default=None, help="one scene name; every scene of the directory by default")
    parser.add_argument("--direction", type=int, choices=(1, -1), default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if not args.scenes_dir:
        parser.error("--scenes-dir (or GIBSON_SCENES_DIR) is required")
    paths = sorted(glob.glob(os.path.join(args.scenes_dir, (args.scene or "*") + ".navmesh")))
    failed = skipped = 0
    for path in paths:
        for o in run_scene(path, args.verbose, args.direction):
            ok = o["arrived"] and not o["retreat"]
            if not o["traversable"]:
                # The flight joins two navmesh islands: no agent crosses it in this simulator, and the policy
                # never makes a portal of it. Reported, not counted -- it is the mesh's failure, not the climb's.
                verdict = "SKIP"
                skipped += 1
            else:
                verdict = "OK" if ok else "FAIL"
                failed += 0 if ok else 1
            print("%-12s c%d %-4s %-5s actions=%3d blocked=%2d recoveries=%d retreat=%d forced=%d z=%.2f->%.2f walk=%s" % (
                o["scene"], o["connector"], o["direction"], verdict, o["actions"], o["blocked"],
                o["recoveries"], int(o["retreat"]), int(o["forced"]), o["final_height"], o["destination_z"],
                o["walkability"]))
    print("failed climbs: %d (island-split connectors skipped: %d)" % (failed, skipped))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

