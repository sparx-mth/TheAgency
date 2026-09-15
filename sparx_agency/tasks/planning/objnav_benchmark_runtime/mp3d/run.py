"""MP3D ObjectNav preflight and evaluation CLI; simulator/model contexts are lazy.

Everything benchmark-specific lives here and in this package's siblings; the
execution path itself is the shared
:func:`~sparx_agency.tasks.planning.objnav_benchmark_runtime.evaluation.run_evaluation`,
so a Gibson number and an MP3D number come out of the same loop and the same
arithmetic.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

from sparx_agency.core.planning.objnav.labels.datasets.mp3d import mp3d_label_mapper
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.tasks.planning.objnav_benchmark.results_io import default_run_dir
from sparx_agency.tasks.planning.objnav_benchmark_runtime.evaluation import (
    evaluation_configuration, run_evaluation)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.provenance import (
    select_episodes, source_fingerprint)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.dataset import MP3DDataset
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.env import MP3DEnv
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.protocol import (
    PROTOCOL, PUBLISHED_VAL_EPISODES, PUBLISHED_VAL_SCENES)

#: The embodiment is the protocol's; only the standoff is a method choice.
#: MP3D grants success for STOP within 0.1 m of a published view point, and
#: those sit within about a metre of the goal's surface, so the search drives
#: to roughly that distance of the landmark it believes in. This is a method
#: parameter, not a redefinition of the evaluator's success.
METHOD_DEFAULTS = {
    "body_height_m": PROTOCOL.agent_height_m,
    "body_radius_m": PROTOCOL.agent_radius_m,
    "preferred_clearance_m": 0.30,
    "stop_distance_m": 1.0,
    # MP3D buildings are the largest of the five benchmarks; the per-episode
    # observed map is centred on the start and must not run out mid-episode.
    "map_size_m": 100.0,
}


class StopDiagnostic:
    """Non-privileged plumbing check, never labelled as the search method."""

    name = "diagnostic-stop"

    def reset(self, episode, target):
        pass

    def plan(self, observation):
        return NavigationCommand.stop_here()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes-dir", default=os.environ.get("MP3D_EPISODES_DIR"),
                   help="objectnav_mp3d_v1 val directory (holds content/)")
    p.add_argument("--scenes-dir", default=os.environ.get("MP3D_SCENES_DIR"),
                   help="data/scene_datasets, the parent of mp3d/")
    p.add_argument("--output", type=Path)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--print-vocabulary", action="store_true")
    p.add_argument("--agent", choices=("rpt", "stop"), default="rpt")
    p.add_argument("--policy-config", type=Path)
    p.add_argument("--detector-url", default="http://127.0.0.1:8092")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-device", type=int, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--scene", help="one scene id, for a labelled subset run")
    p.add_argument("--record", action="store_true")
    p.add_argument("--video-fps", type=int, default=6)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-start-validation", action="store_true",
                   help="skip the per-start preflight (it loads every selected "
                        "scene, so it needs the rendering GPU)")
    p.add_argument("--exclude-unreachable-starts", action="store_true",
                   help="drop starts from which no view point is reachable, and "
                        "record the exclusion (this forfeits the full-split claim)")
    p.add_argument("--allow-sim-version-mismatch", action="store_true")
    p.add_argument("--allow-shared-gpu", action="store_true")
    return p


def _runtime(issues):
    """Versions of what actually runs, and a named issue for anything missing."""
    packages = {}
    for module, distribution in (("numpy", "numpy"), ("skimage", "scikit-image"),
                                 ("scipy", "scipy"), ("cv2", "opencv-python"),
                                 ("networkx", "networkx"), ("requests", "requests"),
                                 ("quaternion", "numpy-quaternion"),
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
    """Build the exact policy that will run, with model identity pinned."""
    if args.agent == "stop":
        return StopDiagnostic(), {"method": "diagnostic-stop", "publishable": False}
    from sparx_agency.core.mapping.topology.llm_client import LLMClient, LLMConfig
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception import HttpDetector
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_policy import (
        RPTSearchPolicy, RPTSettings)
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.services import (
        VerifiedLLMClient, public_service_url)

    overrides = json.loads(args.policy_config.read_text()) if args.policy_config else {}
    if not isinstance(overrides, dict):
        raise ValueError("--policy-config must hold a JSON object")
    settings = RPTSettings(**dict(METHOD_DEFAULTS, **overrides, seed=args.seed))
    config = LLMConfig.from_env()
    config.seed = args.seed
    client = VerifiedLLMClient(LLMClient(config))
    detector = HttpDetector(public_service_url(args.detector_url),
                            mp3d_label_mapper().vocabulary())
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
        raise RuntimeError("Detector emission threshold hides door candidates; "
                           "restart the service with --conf 0.05")
    info = policy.configuration()
    llm = asdict(config)
    llm.pop("api_key", None)
    info.update(llm=llm, llm_identity=identities["LLM"], detector=identities["detector"],
                detector_url=args.detector_url, method_defaults=dict(METHOD_DEFAULTS),
                method_overrides=dict(overrides))
    return policy, info


def _gpu_gate(args):
    if args.allow_shared_gpu:
        return
    result = subprocess.run(["nvidia-smi", "--id=%d" % args.gpu_device,
                             "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    if int(result.stdout.strip()) > 512:
        raise RuntimeError("Rendering GPU is occupied; use CPU/remote models or "
                           "explicitly --allow-shared-gpu")


def prepare(args):
    """Verify the exact requested selection, assets and services.

    Everything except the start preflight is render-free; that one loads each
    selected scene so its distances come from the navmesh the run scores on
    (``--skip-start-validation`` opts out).
    """
    issues = []
    if args.seed < 0 or args.gpu_device < 0:
        issues.append("--seed and --gpu-device must be non-negative")
    if args.agent == "stop" and args.limit is None:
        issues.append("The stop diagnostic requires an explicit --limit")
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
        issues.append("Reference habitat-sim is %s; installed %r. Use "
                      "--allow-sim-version-mismatch only for an acknowledged port."
                      % (PROTOCOL.reference_sim_version, runtime.get("habitat-sim")))
    dataset = env = policy = None
    ids, method, starts, excluded = (), {}, None, ()
    if not args.episodes_dir or not args.scenes_dir:
        issues.append("Set --episodes-dir to the objectnav_mp3d_v1 val directory and "
                      "--scenes-dir to data/scene_datasets")
    else:
        try:
            full = args.scene is None and args.limit is None and args.shards == 1
            dataset = MP3DDataset(args.episodes_dir, args.scenes_dir,
                                  full=full, scene=args.scene)
            env = MP3DEnv(dataset, args.seed, args.gpu_device)
            ids = select_episodes(env.episode_ids(), args.limit, args.shards, args.shard_index)
            if not args.skip_start_validation:
                starts = env.validate_starts(ids)
                unreachable = starts["unreachable_starts"]
                if unreachable and args.exclude_unreachable_starts:
                    excluded = tuple(unreachable)
                    ids = select_episodes([i for i in ids if i not in set(excluded)])
                elif unreachable:
                    issues.append(
                        "No published view point is reachable from %d selected start(s), "
                        "so their l is infinite and SPL is undefined: %s. This is worth "
                        "diagnosing before it is worked around: re-run the preflight with "
                        "a protocol whose navmesh_source is %r to see whether they are an "
                        "artefact of the agent-recomputed navmesh. To run anyway, pass "
                        "--exclude-unreachable-starts, which drops them, records them, and "
                        "forfeits the full-split claim."
                        % (len(unreachable), ", ".join(unreachable[:5]), "published"))
        except (OSError, ValueError, KeyError, ImportError) as exc:
            issues.append("Dataset/scorer: %s" % exc)
    try:
        policy, method = _method(args)
    except Exception as exc:
        issues.append("Method services/configuration: %s" % exc)
    full_split = (args.scene is None and args.limit is None and args.shards == 1
                  and not excluded and len(ids) == PUBLISHED_VAL_EPISODES
                  and dataset is not None and len(dataset.scene_counts) == PUBLISHED_VAL_SCENES)
    config = {"protocol": asdict(PROTOCOL), "runtime": runtime,
              "source_sha256": source_fingerprint(), "method": method,
              "seed": args.seed, "gpu_device": args.gpu_device,
              "reference_sim_version_match": version_match,
              "shards": args.shards, "shard_index": args.shard_index, "limit": args.limit,
              "scene": args.scene, "allow_shared_gpu": args.allow_shared_gpu,
              "recording": {"enabled": args.record, "fps": args.video_fps},
              "start_validation": starts, "full_split": full_split,
              "excluded_unreachable_starts": list(excluded),
              # Settled once here, so preflight, the lock and the run all
              # score the same episodes in the same order.
              "selected_episode_ids": list(ids),
              "dataset": dataset.manifest() if dataset is not None else None}
    return env, policy, config, issues


def main(argv=None, *, frozen_lock=None, expected_episode_ids=None):
    args = parser().parse_args(argv)
    if args.print_vocabulary:
        print(",".join(mp3d_label_mapper().vocabulary()))
        return 0
    if args.resume and args.output is None:
        parser().error("--resume requires --output")
    env, policy, config, issues = prepare(args)
    try:
        if issues:
            print(json.dumps({"ready": False, "issues": issues}, indent=2))
            return 1
        ids = tuple(config["selected_episode_ids"])
        settings = PROTOCOL.evaluation()
        if args.preflight:
            print(json.dumps({"ready": True, "configuration": evaluation_configuration(
                config, ids, settings, policy=policy)}, indent=2, sort_keys=True))
            return 0
        _gpu_gate(args)
        output = args.output or default_run_dir(PROTOCOL.benchmark, PROTOCOL.split)
        total = len(ids)

        def progress(index, count, row):
            print("%d/%d %s SR=%d SPL=%.3f DTG=%s SoftSPL=%.3f"
                  % (index, count, row.episode_id, row.success, row.spl,
                     row.distance_to_goal_m, row.soft_spl), flush=True)

        summary = run_evaluation(
            env, policy, mp3d_label_mapper(), output=output, config=config,
            settings=settings, episode_ids=ids, resume=args.resume,
            frozen_lock=frozen_lock, expected_episode_ids=expected_episode_ids,
            record=args.record, video_fps=args.video_fps, progress=progress)
        env = None  # run_evaluation closes it
        from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.report import write_report
        write_report(output)
        print("Results: %s (%d of %d episodes)"
              % (output, summary.overall.n_episodes, total))
        return 0
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
