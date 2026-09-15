"""HM3D preflight and evaluation CLI; simulator and model contexts stay lazy.

Everything reproducible about a run is assembled by :func:`prepare` *before*
anything is loaded or rendered, so ``--preflight`` answers "would this run, and
under exactly what configuration" without touching the GPU. Execution itself is
the shared
:func:`~sparx_agency.tasks.planning.objnav_benchmark_runtime.evaluation.run_evaluation`
-- this module adds no second scoring loop, no second logger and no second
definition of the protocol.
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

from sparx_agency.core.planning.objnav.labels.datasets.hm3d import hm3d_label_mapper
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.tasks.planning.objnav_benchmark.results_io import default_run_dir
from sparx_agency.tasks.planning.objnav_benchmark_runtime.evaluation import run_evaluation
from sparx_agency.tasks.planning.objnav_benchmark_runtime.provenance import (
    select_episodes, source_fingerprint)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.dataset import HM3DDataset
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.env import HM3DEnv
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.protocol import (
    PROTOCOLS, protocol_for)

#: How many validated starts must reproduce the publisher's own geodesic
#: distance to within a millimetre before the run is allowed to proceed. Not
#: 100%: a handful of starts sit exactly on a view point, where float32 and
#: navmesh snapping put the two numbers a hair apart.
GEODESIC_AGREEMENT_FRACTION = 0.99

#: Packages the runtime needs, and the distribution each version is read from.
RUNTIME_PACKAGES = (
    ("numpy", "numpy"), ("scipy", "scipy"), ("skimage", "scikit-image"),
    ("cv2", "opencv-python"), ("networkx", "networkx"), ("requests", "requests"),
    ("quaternion", "numpy-quaternion"), ("habitat_sim", "habitat-sim"),
)


class StopDiagnostic:
    """Non-privileged plumbing check. Never labelled as the search method."""

    name = "diagnostic-stop"

    def reset(self, episode, target):
        pass

    def plan(self, observation):
        return NavigationCommand.stop_here()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", choices=sorted(PROTOCOLS), default="v2",
                   help="Which published HM3D ObjectNav dataset to run")
    p.add_argument("--split", default="val",
                   help="val is the published benchmark; train and val_mini are "
                        "for development and are always labelled a subset")
    p.add_argument("--episodes-dir", default=os.environ.get("HM3D_EPISODES_DIR"),
                   help="The split directory holding <split>.json.gz and content/")
    p.add_argument("--scenes-dir", default=os.environ.get("HM3D_SCENES_DIR"),
                   help="What habitat-lab calls data/scene_datasets/")
    p.add_argument("--output", type=Path)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--print-vocabulary", action="store_true")
    p.add_argument("--agent", choices=("rpt", "stop"), default="rpt")
    p.add_argument("--policy-config", type=Path,
                   help="JSON object of RPTSettings overrides")
    p.add_argument("--detector-url", default="http://127.0.0.1:18092")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-device", type=int, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--scenes", nargs="+", help="Scene folder names, for a subset run")
    p.add_argument("--episodes-per-scene", type=int,
                   help="Keep only the first N episodes of each scene. Required "
                        "for a development split: HM3D train is millions of rows")
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--record", action="store_true")
    p.add_argument("--video-fps", type=int, default=6)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--validate-starts", action="store_true",
                   help="Load every selected scene and measure l up front. Slow, "
                        "and the only way to find an unreachable start before "
                        "the run stops at it.")
    p.add_argument("--exclude-unreachable", action="store_true",
                   help="With --validate-starts, drop episodes whose goals are "
                        "unreachable. Recorded explicitly; never silent.")
    p.add_argument("--allow-geodesic-mismatch", action="store_true",
                   help="Continue even when our shortest paths disagree with the "
                        "publisher's own. Recorded in the manifest; normally this "
                        "means the wrong navmesh, and every SPL would be wrong")
    p.add_argument("--skip-scene-hashes", action="store_true",
                   help="Hash only the episode files. Faster preflight, weaker "
                        "provenance: a swapped scene release becomes invisible.")
    p.add_argument("--allow-shared-gpu", action="store_true")
    return p


def geodesic_agreement_issue(agreement, *, allow=False):
    """Why this run's shortest paths should not be trusted, or ``None``.

    ``info.geodesic_distance`` is the publisher's geodesic to the nearest goal
    view point -- the same quantity as ``l``. So the two disagreeing is not a
    difference of definition: it is our navmesh, our frames or our view points
    being wrong, and it would move every SPL without failing anything
    downstream. That makes it a preflight issue rather than a logged number.

    Args:
        agreement: What :meth:`HM3DEnv.geodesic_agreement` returned.
        allow: The operator has said the deviation is deliberate.

    Returns:
        The complaint, or ``None`` when the agreement is good enough or was
        explicitly accepted.
    """
    if agreement is None or allow:
        return None
    if agreement["within_1mm"] >= GEODESIC_AGREEMENT_FRACTION * agreement["episodes"]:
        return None
    return ("Only %d of %d checked starts reproduce the publisher's own geodesic "
            "distance within 1 mm (median off by %.3f m, worst %.3f m). That is "
            "the same quantity as l, so this is most likely the wrong navmesh, "
            "and every SPL would be wrong with it. Investigate, or pass "
            "--allow-geodesic-mismatch to record the deviation and continue."
            % (agreement["within_1mm"], agreement["episodes"],
               agreement["median_abs_m"], agreement["max_abs_m"]))


def _runtime(issues):
    """Version every package whose behaviour could change a number."""
    packages = {}
    for module, distribution in RUNTIME_PACKAGES:
        try:
            loaded = importlib.import_module(module)
            try:
                packages[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                packages[distribution] = getattr(loaded, "__version__", "unknown")
        except (ImportError, OSError) as exc:
            issues.append("%s unavailable: %s" % (module, exc))
    return packages


def _method(args, protocol):
    """Build the search policy and prove its services are the ones recorded."""
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
    # Embodiment comes from the benchmark's own agent, never from a default.
    embodiment = {"body_height_m": protocol.agent_height_m,
                  "body_radius_m": protocol.agent_radius_m,
                  "preferred_clearance_m": 0.30,
                  "stop_distance_m": 1.0}
    settings = RPTSettings(**dict(embodiment, **dict(overrides, seed=args.seed)))
    config = LLMConfig.from_env()
    config.seed = args.seed
    client = VerifiedLLMClient(LLMClient(config))
    detector = HttpDetector(public_service_url(args.detector_url),
                            hm3d_label_mapper().vocabulary())
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
                           "restart the detector with --conf 0.05")
    info = policy.configuration()
    llm = asdict(config)
    llm.pop("api_key", None)
    info.update(llm=llm, llm_identity=identities["LLM"],
                detector=identities["detector"], detector_url=args.detector_url)
    return policy, info


def _gpu_gate(args):
    """Refuse to share the rendering GPU with a resident model by accident."""
    if args.allow_shared_gpu:
        return
    result = subprocess.run(
        ["nvidia-smi", "--id=%d" % args.gpu_device, "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    if int(result.stdout.strip()) > 512:
        raise RuntimeError("Rendering GPU is occupied; run the models on CPU or "
                           "another device, or pass --allow-shared-gpu explicitly")


def prepare(args):
    """Assemble the exact run configuration, and every reason it could not run.

    Returns:
        ``(env, policy, label_mapper, config, issues)``. ``env`` is returned
        even when ``issues`` is non-empty so the caller can close it.
    """
    issues = []
    protocol = protocol_for(args.version).with_split(args.split)
    development = protocol.split != "val"
    subset = bool(args.scenes or args.limit or args.shards > 1 or development
                  or args.episodes_per_scene)
    unbounded_development = development and args.episodes_per_scene is None
    if unbounded_development:
        issues.append("A development split needs --episodes-per-scene: HM3D's "
                      "train split holds millions of episodes, not thousands")
    if args.seed < 0 or args.gpu_device < 0:
        issues.append("--seed and --gpu-device must be non-negative")
    if args.agent == "stop" and args.limit is None:
        issues.append("The stop diagnostic requires an explicit --limit")
    if not 1 <= args.video_fps <= 60:
        issues.append("--video-fps must be between 1 and 60")
    if args.exclude_unreachable and not args.validate_starts:
        issues.append("--exclude-unreachable needs --validate-starts to find them")
    if args.record:
        try:
            from sparx_agency.tasks.planning.objnav_benchmark_runtime.recording import (
                ffmpeg_executable)
            ffmpeg_executable()
        except Exception as exc:
            issues.append("Recording: %s" % exc)
    try:
        _gpu_gate(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        issues.append("Rendering GPU: %s" % exc)
    runtime = _runtime(issues)
    sim_match = runtime.get("habitat-sim") == protocol.reference_habitat_sim_version
    dataset = env = policy = None
    ids, method, manifest = (), {}, None
    excluded, start_distances = [], {}
    if not args.episodes_dir or not args.scenes_dir:
        issues.append("Set --episodes-dir to the split directory and --scenes-dir "
                      "to the directory holding hm3d/<split>/<scene>/")
    elif unbounded_development:
        # Loading it to find out how big it is would BE the problem.
        pass
    else:
        try:
            dataset = HM3DDataset(args.episodes_dir, args.scenes_dir, protocol,
                                  full=not subset, scenes=args.scenes,
                                  episodes_per_scene=args.episodes_per_scene)
            manifest = dataset.manifest(hash_scenes=not args.skip_scene_hashes)
            env = HM3DEnv(dataset, protocol, args.seed, args.gpu_device)
            ids = select_episodes(env.episode_ids(), args.limit, args.shards,
                                  args.shard_index)
            if args.validate_starts:
                start_distances = env.validate_starts(ids)
                excluded = list(env.unreachable_episode_ids)
                if excluded and not args.exclude_unreachable:
                    issues.append(
                        "%d selected episodes have no reachable goal view point "
                        "(%s...). Re-run with --exclude-unreachable to drop them "
                        "explicitly, which is recorded in the run manifest."
                        % (len(excluded), ", ".join(excluded[:3])))
                elif excluded:
                    env.excluded_episode_ids = tuple(excluded)
                    ids = tuple(i for i in ids if i not in set(excluded))
        except (OSError, ValueError, KeyError, ImportError) as exc:
            issues.append("Dataset/scorer: %s" % exc)
    try:
        policy, method = _method(args, protocol)
    except Exception as exc:
        issues.append("Method services/configuration: %s" % exc)
    config = {
        "protocol": asdict(protocol),
        "protocol_id": protocol.protocol_id,
        "runtime": runtime,
        "reference_habitat_sim_version_match": sim_match,
        "method": method,
        "seed": args.seed,
        "gpu_device": args.gpu_device,
        "shards": args.shards,
        "shard_index": args.shard_index,
        "limit": args.limit,
        "scenes": list(args.scenes or ()),
        "episodes_per_scene": args.episodes_per_scene,
        "allow_shared_gpu": args.allow_shared_gpu,
        "recording": {"enabled": args.record, "fps": args.video_fps},
        "split": protocol.split,
        "published_split": not development,
        "starts_validated": bool(args.validate_starts),
        "excluded_unreachable_episode_ids": sorted(excluded),
        "full_split": not subset and not excluded,
        "dataset": manifest,
    }
    if start_distances:
        # Strict JSON cannot hold an infinity, and an unreachable start is
        # exactly the case this summary exists to surface -- so report the
        # reachable range and count the rest.
        finite = [v for v in start_distances.values() if v != float("inf")]
        config["start_distance_summary"] = {
            "episodes": len(start_distances),
            "unreachable": len(start_distances) - len(finite),
            "min_m": min(finite) if finite else None,
            "max_m": max(finite) if finite else None,
        }
        if env is not None:
            agreement = env.geodesic_agreement()
            if agreement is not None:
                config["published_geodesic_agreement"] = agreement
                # The publisher's info.geodesic_distance is the same quantity as
                # l, so a systematic disagreement is not a definitional
                # difference: it is our navmesh, our frames or our view points
                # being wrong, and it would move every SPL without failing
                # anything downstream.
                complaint = geodesic_agreement_issue(
                    agreement, allow=args.allow_geodesic_mismatch)
                if complaint is not None:
                    issues.append(complaint)
                config["published_geodesic_agreement"]["accepted"] = complaint is None
            # Which navmesh each scene was actually measured on. Deterministic,
            # so it is part of the frozen configuration rather than a log line.
            if env.navmesh_by_scene:
                config["navmesh_by_scene"] = dict(sorted(env.navmesh_by_scene.items()))
    config["selected_episode_ids"] = list(ids)
    return env, policy, hm3d_label_mapper(), config, issues


def main(argv=None, *, frozen_lock=None, expected_episode_ids=None):
    args = parser().parse_args(argv)
    if args.print_vocabulary:
        print(",".join(hm3d_label_mapper().vocabulary()))
        return 0
    if args.resume and args.output is None:
        parser().error("--resume requires --output")
    protocol = protocol_for(args.version).with_split(args.split)
    env, policy, label_mapper, config, issues = prepare(args)
    closed = False
    try:
        if issues:
            print(json.dumps({"ready": False, "issues": issues}, indent=2))
            return 1
        if args.preflight:
            print(json.dumps({"ready": True, "configuration": config}, indent=2,
                             default=str))
            return 0
        _gpu_gate(args)
        output = args.output or default_run_dir(protocol.benchmark, protocol.split)
        selected = tuple(config["selected_episode_ids"])
        total = len(selected)

        def progress(index, total_count, row):
            """One line per finished episode; a 1000-episode run runs for hours."""
            print("%d/%d %s %s SR=%d SPL=%.3f DTG=%s SoftSPL=%.3f"
                  % (index, total_count, row.episode_id, row.target_category,
                     row.success, row.spl, row.distance_to_goal_m, row.soft_spl),
                  flush=True)

        summary = run_evaluation(
            env, policy, label_mapper, output=output, config=config,
            settings=protocol.evaluation_settings(), episode_ids=selected,
            resume=args.resume, frozen_lock=frozen_lock,
            expected_episode_ids=expected_episode_ids, record=args.record,
            video_fps=args.video_fps, progress=progress)
        closed = True  # run_evaluation closes the environment itself
        from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.report import write_report
        write_report(output)
        print("Results: %s (%d of %d episodes)"
              % (output, summary.overall.n_episodes, total))
        return 0
    finally:
        if env is not None and not closed:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
