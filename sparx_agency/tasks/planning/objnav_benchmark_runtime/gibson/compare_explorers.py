"""Paired explorer comparison on disclosed, previously inspected development starts.

Runs the existing Gibson evaluator in sequential fresh processes. Never changes
episode caps, camera/action settings, detector/LLM services or scoring. Curves
are observed-map proxies. This is deliberately not a held-out evaluation tool.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import SCENES
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run import source_fingerprint


def summarize(root, *, role="previously_inspected_development_starts", require_falcon_activity=True):
    rows, curves = [], []
    identities = {}
    for backend in ("frontier", "falcon"):
        for path in sorted((root / backend).glob("*/episodes.jsonl")):
            config = json.loads((path.parent / "run.json").read_text())["config"]
            key = path.parent.name
            method = dict(config["method"])
            adaptation = dict(method["adaptation"])
            adaptation.pop("local_exploration", None)
            identity = {"protocol": config["protocol"], "seed": config["seed"],
                        "source": config["source_sha256"], "episodes": config["selected_episode_ids"],
                        "detector": method["detector"], "llm": method["llm"],
                        "llm_identity": method["llm_identity"], "adaptation": adaptation,
                        "kinematics": config["kinematics"], "dataset": config["dataset"]}
            if key in identities and identity != identities[key]:
                raise ValueError("Paired configuration mismatch for %s" % key)
            identities[key] = identity
            sidecar = path.parent / "evaluation_diagnostics.jsonl"
            evaluation = {item["episode_id"]: item for item in
                          (json.loads(line) for line in sidecar.read_text().splitlines())} if sidecar.exists() else {}
            for line in path.read_text().splitlines():
                episode = json.loads(line)
                native = evaluation.get(episode["episode_id"], {})
                policy = episode["agent_info"]["policy"]
                metrics = policy["exploration_metrics"]
                hierarchy = policy.get("hierarchy") or {}
                plans = hierarchy.get("falcon_plans", [])
                executed = [plan for plan in plans if plan["status"] == "planned"]
                if backend == "falcon" and not executed and require_falcon_activity:
                    raise ValueError("FALCON did not produce an executable CP/SOP plan in %s" % path)
                rows.append({"explorer": backend, "scene": episode["scene_id"], "episode": episode["episode_id"],
                             "success": episode["success"], "spl": episode["spl"],
                             "distance_to_goal_m": episode["distance_to_goal_m"], "actions": episode["steps"],
                             "actions_to_success": native.get("first_success_action", episode["steps"]) if episode["success"] else None,
                             "observed_gain_m2": metrics["observed_gain_m2"],
                             "coverage_gain_m2_per_action": metrics["coverage_gain_m2_per_action"],
                             "revisits": metrics["revisits_0_5m_cells"], "repeated_turns": metrics["consecutive_same_turns"],
                             "turn_reversals": metrics["immediate_turn_reversals"], "collisions_proxy": policy["blocked"],
                             "native_collisions": native.get("collisions"),
                             "no_progress_actions": metrics["no_translation_or_coverage_actions"],
                             "no_progress_decision_wall_s": metrics.get("no_progress_decision_wall_s"),
                             "wall_s": episode["wall_s"], "latency": metrics["latency"],
                             "actions_by_phase": metrics["actions_by_phase"],
                             "burst_reasons": [b["reason"] for b in hierarchy.get("bursts", [])],
                             "falcon_completed_plans": len(executed), "falcon_exercised": bool(executed),
                             "agent_error": episode["agent_error"]})
                curves.extend(dict(point, explorer=backend, episode=episode["episode_id"])
                              for point in metrics["coverage_curve"])
    report = {"role": role, "held_out": False,
              "coverage": "ever-observed occupancy area proxy, not GT area; initial observation excluded from gain",
              "observations": "identical sensor protocol/starts, not identical post-action frames after policies diverge",
              "collisions": "native Habitat contacts when available; failed-forward notifications reported separately as proxy",
              "rows": rows}
    (root / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if curves:
        fields = sorted({key for row in curves for key in row})
        with (root / "coverage.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(curves)
    lines = ["# Paired Gibson development comparison", "", "Role: %s; no held-out claim. Coverage is an observed-map proxy." % role, "",
             "| Explorer | Scene | Success | SPL | DTG m | Actions | New m²/action |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| {explorer} | {scene} | {success} | {spl:.4f} | {distance_to_goal_m} | {actions} | {coverage_gain_m2_per_action:.4f} |".format(**row))
    lines += ["", "See comparison.json for latency tails, repeated turns, revisits, blocked-motion proxy, burst reasons and stage allocations.",
              "Coverage curves: coverage.csv. Higher proxy coverage is not proof of better ObjectNav."]
    (root / "COMPARISON.md").write_text("\n".join(lines) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=["Collierville", "Corozal"])
    parser.add_argument("--detector-url", required=True)
    parser.add_argument("--detector-backend", choices=("yolo_world", "llmdet"), required=True)
    parser.add_argument("--episodes-dir", required=True)
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-config", type=Path)
    parser.add_argument("--allow-sim-version-mismatch", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.summarize_only:
        args.output.mkdir(parents=True, exist_ok=False)
        fingerprint = source_fingerprint()
        for backend in ("frontier", "falcon"):
            for scene in args.scenes:
                if fingerprint != source_fingerprint():
                    raise RuntimeError("Source changed during controlled comparison")
                destination = args.output / backend / scene
                destination.mkdir(parents=True)
                command = [sys.executable, "-u", "-m", "sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run",
                           "--scene", scene, "--limit", "1", "--explorer", backend, "--output", str(destination),
                           "--detector-url", args.detector_url, "--detector-backend", args.detector_backend,
                           "--episodes-dir", args.episodes_dir, "--scenes-dir", args.scenes_dir, "--seed", str(args.seed)]
                if args.policy_config:
                    command += ["--policy-config", str(args.policy_config)]
                if args.allow_sim_version_mismatch:
                    command += ["--allow-sim-version-mismatch"]
                print("Running", backend, scene, flush=True)
                with (destination / "console.log").open("w") as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=1800)
    summarize(args.output)
    print(args.output / "COMPARISON.md")


if __name__ == "__main__":
    main()



