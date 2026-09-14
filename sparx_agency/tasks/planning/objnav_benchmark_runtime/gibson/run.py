"""Gibson preflight and evaluation CLI; simulator/model contexts are lazy."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import subprocess

from sparx_agency.core.planning.objnav.agent.headless_agent import HeadlessObjNavAgent
from sparx_agency.core.planning.objnav.labels.datasets.gibson import gibson_label_mapper
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark.results_io import default_run_dir
from sparx_agency.tasks.planning.objnav_benchmark.runner import run_benchmark
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.dataset import GibsonDataset
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.env import GibsonEnv
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.protocol import PROTOCOL, SCENES


class StopDiagnostic:
    """Non-privileged plumbing check, never labelled as the search method."""
    name = "diagnostic-stop"

    def reset(self, episode, target):
        pass

    def plan(self, observation):
        return NavigationCommand.stop_here()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes-dir", default=os.environ.get("GIBSON_EPISODES_DIR"))
    p.add_argument("--scenes-dir", default=os.environ.get("GIBSON_SCENES_DIR"))
    p.add_argument("--output", type=Path)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--print-vocabulary", action="store_true")
    p.add_argument("--agent", choices=("rpt", "stop"), default="rpt")
    p.add_argument("--policy-config", type=Path)
    p.add_argument("--detector-url", default="http://127.0.0.1:8092")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-device", type=int, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--scene", choices=SCENES)
    p.add_argument("--record", action="store_true")
    p.add_argument("--video-fps", type=int, default=6)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-sim-version-mismatch", action="store_true")
    p.add_argument("--allow-shared-gpu", action="store_true")
    return p


def select_episodes(ids, limit=None, shards=1, shard_index=0):
    """Stable sharding before the explicit per-shard subset limit."""
    if shards < 1 or not 0 <= shard_index < shards:
        raise ValueError("Need shards >= 1 and 0 <= shard-index < shards")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive")
    selected = tuple(ids)[shard_index::shards]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("Selection contains no episodes")
    return selected


def source_fingerprint():
    """Hash actual Python sources, including dirty/untracked edits."""
    package = Path(__file__).resolve().parents[4]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _runtime(issues):
    packages = {}
    for module, distribution in (("numpy", "numpy"), ("skfmm", "scikit-fmm"),
                                  ("skimage", "scikit-image"), ("scipy", "scipy"),
                                  ("cv2", "opencv-python"), ("networkx", "networkx"),
                                  ("requests", "requests"), ("quaternion", "numpy-quaternion"),
                                  ("habitat_sim", "habitat-sim")):
        try:
            loaded = importlib.import_module(module)
            try:
                packages[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                packages[distribution] = getattr(loaded, "__version__", "unknown")
        except (ImportError, OSError) as exc:
            issues.append("%s unavailable: %s" % (module, exc))
    return packages


def _method(args):
    if args.agent == "stop":
        return StopDiagnostic(), {"method": "diagnostic-stop", "publishable": False}
    from sparx_agency.core.mapping.topology.llm_client import LLMClient, LLMConfig
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception import HttpDetector
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_policy import RPTSearchPolicy, RPTSettings
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.services import VerifiedLLMClient, public_service_url
    overrides = json.loads(args.policy_config.read_text()) if args.policy_config else {}
    if not isinstance(overrides, dict):
        raise ValueError("--policy-config must hold a JSON object")
    settings = RPTSettings(**dict(overrides, seed=args.seed))
    config = LLMConfig.from_env()
    config.seed = args.seed
    client = VerifiedLLMClient(LLMClient(config))
    detector = HttpDetector(public_service_url(args.detector_url), gibson_label_mapper().vocabulary())
    problems, identities = [], {}
    for name, service in (("LLM", client), ("detector", detector)):
        try:
            identities[name] = service.health()
        except Exception as exc:
            problems.append("%s: %s" % (name, exc))
    if problems:
        raise RuntimeError("; ".join(problems))
    policy = RPTSearchPolicy(detector, client, settings)
    emitted = float(identities["detector"]["metadata"]["detector_config"]["conf_thresh"])
    if emitted > min(settings.detection_confidence, policy.door_settings.confidence):
        raise RuntimeError("Detector emission threshold hides door candidates; restart with --conf 0.05")
    info = policy.configuration()
    llm = asdict(config)
    llm.pop("api_key", None)
    info.update(llm=llm, llm_identity=identities["LLM"], detector=identities["detector"], detector_url=args.detector_url)
    return policy, info


def _gpu_gate(args):
    if args.allow_shared_gpu:
        return
    result = subprocess.run(["nvidia-smi", "--id=%d" % args.gpu_device, "--query-gpu=memory.used",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    if int(result.stdout.strip()) > 512:
        raise RuntimeError("Rendering GPU is occupied; use CPU/remote models or explicitly --allow-shared-gpu")


def prepare(args):
    """Verify the exact requested dataset selection and all runtime services."""
    issues = []
    if args.seed < 0 or args.gpu_device < 0:
        issues.append("--seed and --gpu-device must be non-negative")
    if args.agent == "stop" and args.limit is None:
        issues.append("The stop diagnostic requires an explicit --limit")
    if args.scene and (args.limit is None or not 1 <= args.limit <= 200):
        issues.append("A single-scene run requires --limit between 1 and 200")
    if not 1 <= args.video_fps <= 60:
        issues.append("--video-fps must be between 1 and 60")
    if args.record:
        try:
            from sparx_agency.tasks.planning.objnav_benchmark_runtime.recording import ffmpeg_executable
            ffmpeg_executable()
        except Exception as exc:
            issues.append("Recording: %s" % exc)
    try:
        _gpu_gate(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        issues.append("Rendering GPU: %s" % exc)
    runtime = _runtime(issues)
    version_match = runtime.get("habitat-sim") == PROTOCOL.reference_sim_version
    if not version_match and not args.allow_sim_version_mismatch:
        issues.append("Reference habitat-sim is 0.1.5; installed %r. Use --allow-sim-version-mismatch only for an acknowledged port." % runtime.get("habitat-sim"))
    dataset = env = policy = None
    ids, method = (), {}
    if not args.episodes_dir or not args.scenes_dir:
        issues.append("Set --episodes-dir to v1.1/val and --scenes-dir to Gibson GLB/navmesh files")
    else:
        try:
            dataset = GibsonDataset(args.episodes_dir, args.scenes_dir, full=args.scene is None, scene=args.scene)
            env = GibsonEnv(dataset, args.seed, args.gpu_device)
            ids = select_episodes(env.episode_ids(), args.limit, args.shards, args.shard_index)
            env.validate_starts(ids)
        except (OSError, ValueError, KeyError, ImportError, EOFError, pickle.UnpicklingError) as exc:
            issues.append("Dataset/scorer: %s" % exc)
    try:
        policy, method = _method(args)
    except Exception as exc:
        issues.append("Method services/configuration: %s" % exc)
    config = {"protocol": asdict(PROTOCOL), "runtime": runtime, "source_sha256": source_fingerprint(),
              "method": method, "seed": args.seed, "gpu_device": args.gpu_device,
              "reference_sim_version_match": version_match, "selected_episode_ids": list(ids),
              "shards": args.shards, "shard_index": args.shard_index, "limit": args.limit,
              "on_agent_error": "record", "allow_shared_gpu": args.allow_shared_gpu,
              "scene": args.scene, "recording": {"enabled": args.record, "fps": args.video_fps},
              "kinematics": asdict(PROTOCOL.kinematics()),
              "full_split": args.scene is None and args.limit is None and args.shards == 1,
              "dataset": dataset.manifest() if dataset else None}
    return env, policy, config, issues


def main(argv=None, *, configuration_guard=None):
    args = parser().parse_args(argv)
    if args.print_vocabulary:
        print(",".join(gibson_label_mapper().vocabulary()))
        return 0
    if args.resume and args.output is None:
        parser().error("--resume requires --output")
    env, policy, config, issues = prepare(args)
    try:
        if issues:
            print(json.dumps({"ready": False, "issues": issues}, indent=2))
            return 1
        if configuration_guard is not None:
            # Validate THIS prepared policy, not only an earlier preflight.
            configuration_guard(config)
        if args.preflight:
            print(json.dumps({"ready": True, "configuration": config}, indent=2))
            return 0
        _gpu_gate(args)
        output = args.output or default_run_dir("gibson", "val")
        recorder = None
        if args.record:
            from sparx_agency.tasks.planning.objnav_benchmark_runtime.recording import EpisodeRecorder, PolicyProbe, RecordingAgent, RecordingEnv
            from sparx_agency.tasks.planning.objnav_benchmark_runtime.dashboard import write_live_page
            output.mkdir(parents=True, exist_ok=True)
            write_live_page(output)
            recorder = EpisodeRecorder(output, policy, fps=args.video_fps)
            probe = PolicyProbe(policy)
            agent = RecordingAgent(HeadlessObjNavAgent(probe, gibson_label_mapper(), name=policy.name), probe, recorder)
            env = RecordingEnv(env, recorder)
        else:
            agent = HeadlessObjNavAgent(policy, gibson_label_mapper(), name=policy.name)

        def progress(i, n, row):
            if recorder is not None:
                recorder.complete(row)
            print("%d/%d %s SR=%d SPL=%.3f DTG=%s SoftSPL=%.3f" %
                  (i, n, row.episode_id, row.success, row.spl, row.distance_to_goal_m, row.soft_spl), flush=True)

        with MetricsLogger(output, config, resume=args.resume) as logger:
            summary = run_benchmark(env, agent, logger=logger, episode_ids=config["selected_episode_ids"],
                                    on_agent_error="record", require_stop_for_success=False,
                                    path_length_dimension="planar", path_length_epsilon_m=PROTOCOL.path_length_epsilon_m,
                                    kinematics=PROTOCOL.kinematics(), progress=progress)
        from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson.report import write_report
        write_report(output)
        if args.record:
            from sparx_agency.tasks.planning.objnav_benchmark_runtime.dashboard import write_dashboard
            write_dashboard(output)
        print("Results: %s (%d episodes)" % (output, summary.overall.n_episodes))
        return 0
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())

