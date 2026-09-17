"""Replay an observed stair-entry pose for a bounded component test, NOT a score.

Uses the actual detector, LLM, policy, converter and Habitat collision checks.
No ground-truth goal or pathfinder output reaches navigation. The explicit
spent-floor-budget option isolates stair handling; it is never a campaign flag.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path

from sparx_agency.core.planning.objnav.agent.headless_agent import HeadlessObjNavAgent
from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.tasks.planning.objnav_benchmark.kinematics import check_motion
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.multifloor_dataset import MULTIFLOOR_PROTOCOL as PROTOCOL
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run import _gpu_gate, _method
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.simulator import HabitatRGBDSimulator
from sparx_agency.tasks.planning.objnav_benchmark_runtime.recording import EpisodeRecorder, PolicyProbe, RecordingAgent


def run(args):
    _gpu_gate(args)
    rows = list(csv.DictReader(args.trajectory.open()))
    matches = [row for row in rows if int(row["step"]) == args.start_step]
    if len(matches) != 1:
        raise ValueError("Need exactly one recorded start-step pose")
    row = matches[0]
    x, y, z, yaw = (float(row[key]) for key in ("x", "y", "z", "yaw"))
    policy, config = _method(args)
    episode = ObjNavEpisode("diagnostic/" + args.scene, args.scene, "gibson", "stair-component-diagnostic",
                            args.target, PROTOCOL.camera(), PROTOCOL.actions(), args.steps)
    probe = PolicyProbe(policy)
    recorder = EpisodeRecorder(args.output, policy)
    agent = RecordingAgent(HeadlessObjNavAgent(probe, gibson_label_mapper()), probe, recorder)
    bridge = HabitatRGBDSimulator(PROTOCOL.camera(), PROTOCOL.actions(), PROTOCOL.agent_radius_m,
                                  PROTOCOL.allow_sliding, args.gpu_device)
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        frame = bridge.reset(args.scenes_dir / (args.scene + ".glb"), args.scenes_dir / (args.scene + ".navmesh"),
                             (-y, z, -x), (math.cos(yaw / 2), 0, math.sin(yaw / 2), 0), args.seed)
        obs = ObjNavObservation(*frame, PROTOCOL.camera(), args.target, 0)
        agent.reset(episode)
        if args.spent_floor_budget:
            policy.building.entered_step = -policy.settings.multifloor.floor_search_actions
        recorder.begin(episode, obs, {})
        for step in range(args.steps):
            decision = agent.act(obs)
            before = obs.pose
            frame = bridge.step(decision.action)
            obs = ObjNavObservation(*frame, PROTOCOL.camera(), args.target, step + 1)
            check_motion(decision.action, before, obs.pose, PROTOCOL.actions(), PROTOCOL.kinematics())
            terminal = decision.action.name == "STOP" or len(policy.mapping.atlas.floors) > 1 or step == args.steps - 1
            recorder.after_step(obs, {}, terminal)
            if step % 5 == 0 or terminal:
                terrain = policy.building.terrain
                print(step, decision.action.name, "z", round(obs.pose.z, 3), policy.building.phase,
                      "candidates", len(terrain.candidates), "portals", len(policy.building.portals),
                      "safe cells", int(terrain.safe.sum()), "decision", decision.info.get("policy"), flush=True)
            if terminal:
                break
        result = {"benchmark_result": False, "start_source": str(args.trajectory), "start_step": args.start_step,
                  "spent_floor_budget": args.spent_floor_budget, "configuration": config,
                  "transition_completed": bool(policy.mapping.atlas.connections),
                  "policy": policy.episode_info(), "final_pose": asdict(obs.pose)}
        (args.output / "diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print("Transition completed:", result["transition_completed"], flush=True)
        return result
    finally:
        recorder.close()
        bridge.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scenes-dir", type=Path, required=True)
    parser.add_argument("--target", default="toilet")
    parser.add_argument("--steps", type=int, default=140)
    parser.add_argument("--spent-floor-budget", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--explorer", choices=("frontier", "falcon"), default="frontier")
    parser.add_argument("--detector-url", required=True)
    parser.add_argument("--detector-backend", choices=("yolo_world", "llmdet"), default="yolo_world")
    parser.add_argument("--policy-config", type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.set_defaults(agent="rpt", allow_shared_gpu=False)
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 500 or args.start_step < 0:
        raise ValueError("Invalid bounded diagnostic selection")
    run(args)


if __name__ == "__main__":
    main()

