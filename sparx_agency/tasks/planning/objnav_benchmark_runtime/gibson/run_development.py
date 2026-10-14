"""Frozen generated Gibson development runs through the shared benchmark loop."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

from sparx_agency.core.planning.objnav.agent.headless_agent import HeadlessObjNavAgent
from sparx_agency.core.planning.objnav.agent.params import HeadlessAgentParams
from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark.runner import run_benchmark
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.development_dataset import DEVELOPMENT_PROTOCOL, DevelopmentDataset, DevelopmentEnv
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.detector_options import add_detector_options
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.multifloor_dataset import MULTIFLOOR_SCHEMA, MultiFloorDataset, MultiFloorEnv
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.run import _gpu_gate, _method, _runtime, select_episodes, source_fingerprint


def prepare(args):
    _gpu_gate(args)
    issues = []
    runtime = _runtime(issues)
    if issues:
        raise RuntimeError("; ".join(issues))
    if runtime.get("habitat-sim") != DEVELOPMENT_PROTOCOL.reference_sim_version and not args.allow_sim_version_mismatch:
        raise RuntimeError("Acknowledge Habitat 0.2.4 versus reference 0.1.5")
    multistory = json.loads(args.manifest.read_text()).get("schema") == MULTIFLOOR_SCHEMA
    dataset = (MultiFloorDataset if multistory else DevelopmentDataset)(args.manifest, args.scene)
    env = (MultiFloorEnv if multistory else DevelopmentEnv)(dataset, args.seed, args.gpu_device)
    try:
        ids = list(select_episodes(env.episode_ids(), getattr(args, "limit", None)))
        env.validate_starts(ids)
        # Multi-storey development episodes are generated to need a floor change: the
        # benchmark default (no stair traversal) is lifted here, visibly in ``method``,
        # unless --policy-config says otherwise.
        policy, method = _method(args, defaults={"allow_stair_traversal": True} if multistory else None)
        policy.target_projector = getattr(env, "target_projector", None)
        method["target_navmesh_projection"] = policy.target_projector is not None
        protocol = env.protocol if multistory else DEVELOPMENT_PROTOCOL
        recorded = ids[:1] if getattr(args, "record_first", False) else ids
        display_counts = {scene: len(audit["levels"]) for scene, audit in
                          dataset.definition.get("generation", {}).get("scene_audits", {}).items()} if multistory else {}
        config = {"protocol": asdict(protocol), "runtime": runtime,
                  "source_sha256": source_fingerprint(), "method": method, "seed": args.seed,
                  "gpu_device": args.gpu_device, "allow_shared_gpu": args.allow_shared_gpu,
                  "selected_episode_ids": ids, "dataset": dataset.manifest(),
                  "kinematics": asdict(protocol.kinematics()),
                  "recording": {"enabled": args.record, "fps": args.video_fps, "display_floor_counts": display_counts,
                                "episode_ids": recorded if args.record else []}, "full_split": False,
                  "reference_sim_version_match": runtime.get("habitat-sim") == protocol.reference_sim_version,
                  "evaluation_role": "frozen_generated_training_development", "held_out_claim": False,
                  "navigation_tuning_during_batch": False, "on_agent_error": "record"}
        return env, policy, json.loads(json.dumps(config, allow_nan=False))
    except BaseException:
        env.close()
        raise


def execute(args, env, policy, config):
    if args.expect_config is not None:
        expected = json.loads(args.expect_config.read_text())
        if config != expected:
            raise RuntimeError("Prepared execution differs from the frozen model/source/data/configuration")
    recorder = None
    active_env = env
    try:
        params = HeadlessAgentParams(converter=policy.converter_params)
        with MetricsLogger(args.output, config, resume=args.resume) as logger:
            if args.record:
                from sparx_agency.tasks.planning.objnav_benchmark_runtime.recording import EpisodeRecorder, PolicyProbe, RecordingAgent, RecordingEnv
                from sparx_agency.tasks.planning.objnav_benchmark_runtime.dashboard import write_live_page
                write_live_page(args.output)
                recorder = EpisodeRecorder(args.output, policy, fps=args.video_fps,
                                           selected_episode_ids=config["recording"]["episode_ids"],
                                           display_floor_counts=config["recording"].get("display_floor_counts", {}))
                probe = PolicyProbe(policy)
                agent = RecordingAgent(HeadlessObjNavAgent(probe, gibson_label_mapper(), params, name=policy.name), probe, recorder)
                active_env = RecordingEnv(env, recorder)
            else:
                agent = HeadlessObjNavAgent(policy, gibson_label_mapper(), params, name=policy.name)

            def completed(index, total, row):
                with (args.output / "evaluation_diagnostics.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(env.evaluation_diagnostics(), episode_id=row.episode_id), allow_nan=False) + "\n")
                if recorder is not None:
                    recorder.complete(row)
                video = (recorder.episode_dir / "video.mp4").resolve() if recorder is not None and recorder.episode_dir else None
                report = dict(episode_id=row.episode_id, success=row.success, spl=row.spl,
                              dtg_m=row.distance_to_goal_m, runtime_s=row.wall_s,
                              steps=row.steps, average_fps=row.steps / row.wall_s if row.wall_s > 0 else 0.0,
                              video=str(video) if video is not None and video.is_file() else None)
                with (args.output / "completion.jsonl").open("a") as stream:
                    stream.write(json.dumps(report, allow_nan=False) + "\n")
                print("\n" + "=" * 78 + "\nEPISODE COMPLETE [%d/%d] %s\n" % (index, total, row.episode_id)
                      + "SR=%d SPL=%.4f DTG=%s m Runtime=%.2f s Average FPS=%.3f\n" %
                      (row.success, row.spl, row.distance_to_goal_m, row.wall_s, report["average_fps"])
                      + "VIDEO: %s\n" % (report["video"] or "not recorded") + "=" * 78, flush=True)
                pause = getattr(args, "inspection_pause", 0.0)
                if pause:
                    print("INSPECTION PAUSE: %.0f seconds; next episode will not start before this expires." % pause, flush=True)
                    time.sleep(pause)

            summary = run_benchmark(active_env, agent, logger=logger, episode_ids=config["selected_episode_ids"],
                                    require_stop_for_success=False,
                                    path_length_dimension="3d" if env.protocol.path_length_dimension == "3d" else "planar",
                                    path_length_epsilon_m=env.protocol.path_length_epsilon_m,
                                    kinematics=env.protocol.kinematics(), on_agent_error="record", progress=completed)
        if args.record:
            from sparx_agency.tasks.planning.objnav_benchmark_runtime.dashboard import write_dashboard
            write_dashboard(args.output)
        reports_path = args.output / "completion.jsonl"
        if reports_path.exists():
            reports = [json.loads(line) for line in reports_path.read_text().splitlines()]
            runtime = sum(item["runtime_s"] for item in reports)
            distances = [item["dtg_m"] for item in reports if item["dtg_m"] is not None]
            totals = dict(episodes=len(reports), success_rate=sum(item["success"] for item in reports) / len(reports),
                          spl=sum(item["spl"] for item in reports) / len(reports),
                          mean_dtg_m=sum(distances) / len(distances) if distances else None,
                          total_runtime_s=runtime,
                          average_fps=sum(item["steps"] for item in reports) / runtime if runtime > 0 else 0.0)
            (args.output / "performance_summary.json").write_text(json.dumps(totals, indent=2, allow_nan=False) + "\n")
            print("FINAL SUMMARY (runtime excludes inspection pauses): " + json.dumps(totals), flush=True)
        return summary
    finally:
        if recorder is not None:
            recorder.close()
        active_env.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--explorer", choices=("frontier", "falcon"), required=True)
    add_detector_options(p, required=True)
    p.add_argument("--policy-config", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, help="Explicit smoke-test subset; omitted for the complete building batch")
    p.add_argument("--gpu-device", type=int, default=0)
    p.add_argument("--record", action="store_true")
    p.add_argument("--record-first", action="store_true", help="Record only the first selected episode")
    p.add_argument("--video-fps", type=int, default=6)
    p.add_argument("--inspection-pause", type=float, default=0.0,
                   help="Seconds to inspect each finalized video before continuing (excluded from episode runtime)")
    p.add_argument("--preflight-output", type=Path)
    p.add_argument("--expect-config", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-sim-version-mismatch", action="store_true")
    p.add_argument("--allow-shared-gpu", action="store_true",
                   help="Explicit operator authorization after GPU ownership/memory preflight; frozen in run identity")
    p.set_defaults(agent="rpt")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.record_first:
        args.record = True
    if args.seed < 0 or args.gpu_device < 0 or not 1 <= args.video_fps <= 60:
        raise ValueError("Invalid seed, GPU index or recording rate")
    if not math.isfinite(args.inspection_pause) or args.inspection_pause < 0:
        raise ValueError("Inspection pause must be finite and non-negative")
    if args.record:
        from sparx_agency.tasks.planning.objnav_benchmark_runtime.recording import ffmpeg_executable
        ffmpeg_executable()
    env, policy, config = prepare(args)
    try:
        if args.preflight_output:
            with args.preflight_output.open("x") as stream:
                json.dump(config, stream, indent=2, allow_nan=False)
                stream.write("\n")
            return
        execute(args, env, policy, config)
    finally:
        env.close()


if __name__ == "__main__":
    main()
