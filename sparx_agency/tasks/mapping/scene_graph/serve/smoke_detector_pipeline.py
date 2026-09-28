"""Bounded TRAIN-scene detector -> unchanged ObjectNav -> real-action smoke.

Run with PYTHONPATH pointing to an existing RoboTHOR adapter checkout. This file
is shared detector test tooling, not a fork of the adapter or navigation policy.
The room-language service is explicitly a deterministic fixture: loading a cold
GPU LLM alongside the simulator is unsafe on this host. No benchmark SR/SPL is
computed, and zero false STOPs in a short run is not a precision claim.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import re
from types import SimpleNamespace


class RoomServiceFixture:
    """Same schema as the existing test_method FakeLLM, without importing pytest."""

    def chat_json(self, system, user):
        if "Room observed objects" in user:
            return {"label": "living_room", "confidence": .8, "reasoning": "smoke fixture"}
        return {"rooms": [{"id": int(pid), "score": 50, "why": "smoke fixture"}
                          for pid in re.findall(r"id=(\d+)", user)]}


def run_smoke(args):
    from sparx_agency.core.planning.objnav.agent.headless_agent import HeadlessObjNavAgent
    from sparx_agency.core.planning.objnav.labels.datasets.robothor import robothor_label_mapper
    from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
    from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.protocol import PROTOCOL
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.run import _simulator, _gpu_gate, embodiment
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception import HttpDetector
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_policy import RPTSearchPolicy, RPTSettings
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.provenance import source_fingerprint

    if not args.scene.startswith("FloorPlan_Train") or not 1 <= args.steps <= 50:
        raise ValueError("Only bounded (1..50 step) training-scene checks are allowed")
    if args.output.exists():
        raise FileExistsError("Use a new smoke output file")
    with gzip.open(args.episodes / "train" / "episodes" / (args.scene + ".json.gz"), "rt") as stream:
        episodes = json.load(stream)
    if isinstance(episodes, dict):
        episodes = episodes["episodes"]
    start = episodes[0]
    category = start["object_type"]
    mapper = robothor_label_mapper()
    detector = HttpDetector(args.url, mapper.vocabulary(), timeout_s=180)
    identity = detector.health()
    if identity["metadata"].get("backend") != args.backend:
        raise ValueError("Smoke endpoint is not the requested backend")
    settings = RPTSettings(**dict(embodiment(), detection_confidence=args.confidence))
    policy = RPTSearchPolicy(detector, RoomServiceFixture(), settings)
    agent = HeadlessObjNavAgent(policy, mapper)
    episode = ObjNavEpisode("detector-smoke/" + args.scene, args.scene,
                            "detector-development", "train", category,
                            PROTOCOL.camera(), PROTOCOL.actions(), args.steps)
    sim_args = SimpleNamespace(thor_build=PROTOCOL.thor_build_id,
                               platform="reference-linux64", allow_shared_gpu=False)
    _gpu_gate(sim_args)
    sim = _simulator(sim_args)
    trace = []
    try:
        rgb, depth, pose = sim.reset(args.scene, start["initial_position"],
                                     start["initial_orientation"], start["initial_horizon"])
        agent.reset(episode)
        for step in range(args.steps):
            observation = ObjNavObservation(rgb, depth, pose, episode.camera, category, step)
            decision = agent.act(observation)
            # Privileged metadata is read AFTER the decision and only by this scorer.
            visible = any(obj["objectType"] == category and obj.get("visible")
                          for obj in sim.metadata["objects"])
            stopped = decision.action.name == "STOP"
            trace.append({"step": step, "action": decision.action.name,
                          "pose": asdict(pose), "target_visible_for_stop": visible,
                          "false_stop": bool(stopped and not visible),
                          "evidence": policy.target_evidence.diagnostics(),
                          "landmarks": len(policy.landmarks)})
            if stopped:
                break
            rgb, depth, pose = sim.step(decision.action)
    finally:
        sim.close()
        detector.session.close()
    result = {"purpose": "bounded training plumbing, NOT benchmark results",
              "room_language_service": "deterministic fixture, not a real LLM",
              "policy_source_sha256": source_fingerprint(), "detector": identity,
              "configuration": policy.configuration(), "scene": args.scene, "target": category,
              "trace": trace, "episode_info": policy.episode_info(),
              "stops": sum(t["action"] == "STOP" for t in trace),
              "false_stops": sum(t["false_stop"] for t in trace)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Smoke complete:", len(trace), "actions;", result["stops"], "STOPs;",
          result["false_stops"], "false STOPs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--scene", default="FloorPlan_Train3_1")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--url", required=True)
    parser.add_argument("--backend", choices=("llmdet", "yolo_world"), required=True)
    parser.add_argument("--confidence", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run_smoke(parser.parse_args())


if __name__ == "__main__":
    main()
