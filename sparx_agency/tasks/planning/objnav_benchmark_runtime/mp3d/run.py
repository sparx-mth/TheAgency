"""MP3D preflight and evaluation CLI; simulator and model contexts stay lazy.

Everything reproducible about a run is assembled by :func:`prepare` *before*
anything is loaded or rendered, so ``--preflight`` answers "would this run, and
under exactly what configuration" without touching the GPU. Execution itself is
the shared
:func:`~sparx_agency.tasks.planning.objnav_benchmark_runtime.evaluation.run_evaluation`
-- this module adds no second scoring loop, no second logger and no second
definition of the protocol. The method is built by the one shared
``policy_factory.build_policy``; the only thing this entrypoint tells it that
is MP3D's is the label table and the protocol's body (height and radius).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess

from sparx_agency.core.planning.objnav.labels.datasets.mp3d import mp3d_label_mapper
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_options import (
    PERCEPTION_GROUND_TRUTH, add_detector_options)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.policy_factory import (  # noqa: F401  (StopDiagnostic re-exported for the sibling CLIs)
    AGENT_CHOICES, AGENT_HELP, StopDiagnostic, build_policy)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.provenance import (
    episodes_of_category, explicit_episodes, gpu_plan_record, select_episodes)
# Everything below the dataset is the Habitat benchmark family's, not MP3D's: the method's defaults, the
# whole ``main``, the preflight's geodesic check (the publisher's geodesic is the same quantity as l on
# every habitat-lab ObjectNav), the oracle liveness gate, the runtime record and the GPU gate.
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.entrypoint import (  # noqa: F401  (re-exported)
    RESUMABLE_CONFIG_CHANGES, RUNTIME_PACKAGES, geodesic_agreement_issue, gpu_gate, method_defaults, oracle_alive,
    run_main, runtime_record)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.dataset import CATEGORIES, MP3DDataset
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.env import MP3DEnv
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.protocol import (
    PROTOCOLS, protocol_for)

#: The MP3D detector's port. Gibson's is 18095 and HM3D's 18092: the three label
#: tables share words but not their order (and MP3D's has 21 rows), and a
#: YOLO-World service is pinned to the vocabulary it was started with.
DEFAULT_DETECTOR_URL = "http://127.0.0.1:18093"



def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", choices=sorted(PROTOCOLS), default="v1",
                   help="Which published MP3D ObjectNav dataset to run (v1 is the only one)")
    p.add_argument("--split", default="val",
                   help="val is the published benchmark; train and val_mini are "
                        "for development and are always labelled a subset")
    p.add_argument("--episodes-dir", default=os.environ.get("MP3D_EPISODES_DIR"),
                   help="The split directory holding <split>.json.gz and content/")
    p.add_argument("--scenes-dir", default=os.environ.get("MP3D_SCENES_DIR"),
                   help="What habitat-lab calls data/scene_datasets/ (the parent of mp3d/)")
    p.add_argument("--output", type=Path)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--print-vocabulary", action="store_true")
    p.add_argument("--agent", choices=AGENT_CHOICES, default="rpt", help=AGENT_HELP)
    p.add_argument("--policy-config", type=Path,
                   help="JSON object of RPTSettings overrides")
    p.add_argument("--explorer", choices=("frontier", "falcon"), default=None,
                   help="Local exploration inside a room: the observed-map frontier "
                        "sweep (default) or the planar FALCON adaptation. Must agree "
                        "with local_exploration in --policy-config when both are given")
    # The detector and the (off by default) segmenter: the same switches as gibson.run
    # and hm3d.run, so a campaign forwards one set of perception flags to any benchmark.
    add_detector_options(p, url_default=DEFAULT_DETECTOR_URL)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu-device", type=int, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--scenes", nargs="+", help="Scene hashes (e.g. 2azQ1b91cZZ), for a subset run")
    p.add_argument("--episodes-per-scene", type=int,
                   help="Keep only the first N episodes of each scene. Required "
                        "for a development split: MP3D train is millions of rows")
    p.add_argument("--category", choices=CATEGORIES,
                   help="Only the published episodes whose goal is this category. Scoring is "
                        "unchanged; the run is a subset, never the full split")
    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle the selection with --seed before --limit applies, so --limit N is a "
                        "seeded random sample instead of the first N in split order; requires --limit")
    p.add_argument("--episode-ids", nargs="+", metavar="ID",
                   help="Run exactly these published episodes (e.g. 2azQ1b91cZZ/000003), in this "
                        "order: the same start and goal for a re-run after a change")
    p.add_argument("--lean", action="store_true",
                   help="Keep only the counters that explain an outcome in each episode row (no "
                        "per-action series or event logs); the scores are untouched")
    p.add_argument("--progress-file", type=Path,
                   help="Where the live progress JSON is rewritten after every episode "
                        "(default: <output>/progress.json)")
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--record", action="store_true")
    p.add_argument("--video-fps", type=int, default=6)
    p.add_argument("--save-frames", action="store_true",
                   help="With --record, also keep every dashboard frame as a JPEG still under "
                        "recordings/<key>/frames/ so single moments can be inspected without "
                        "decoding the video")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume-accepting-changes", nargs="+", choices=RESUMABLE_CONFIG_CHANGES, default=(),
                   metavar="KEY", dest="resume_accepting_changes",
                   help="With --resume: the frozen-config keys that may differ from the recorded run -- "
                        "provenance or the kinematic self-check only (%s); what differed is written into "
                        "run.json's resume entry" % ", ".join(RESUMABLE_CONFIG_CHANGES))
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
    p.add_argument("--software-render", action="store_true",
                   help="Render with Mesa's software EGL device (llvmpipe) instead of a CUDA GPU: habitat gets no "
                        "GPU device, the GPU gate is skipped, and the frozen configuration records the renderer. "
                        "The launcher must leave Mesa no hardware device (run_benchmark.sh hides /dev/dri), or "
                        "device 0 is whatever hardware Mesa finds first")
    return p


def _method(args, protocol, ground_truth=None):
    """Build the search policy and prove its services are the ones recorded.

    One builder for every benchmark (``policy_factory.build_policy``); the
    defaults below ``--policy-config`` are the shared Habitat
    :func:`~...habitat.entrypoint.method_defaults` -- the benchmark's own
    agent (body height and radius) and, under its view-point evaluator, the
    planar STOP overshoot off (since the 2026-11-11 merge; HM3D's finding, the
    same scoring rule here) -- so every other method knob (clearance, stop
    distance, closing, peeks, ...) is ``RPTSettings``' default here exactly as
    it is on Gibson and HM3D. Nothing here came from an MP3D score.
    ``ground_truth`` is the semantic-frame source under ``--perception
    ground-truth``, built by :func:`prepare` beside the environment.
    """
    return build_policy(args, mp3d_label_mapper(), defaults=method_defaults(protocol), ground_truth=ground_truth)


def ground_truth_issues(dataset, episode_ids=None):
    """Why the ground-truth perception could not run here: the annotation files it needs and lacks.

    habitat-sim renders instance 0 everywhere when a scene's ``.house`` or
    ``_semantic.ply`` is absent, and misplaces the semantic mesh without the
    publisher's ``mp3d.scene_dataset_config.json`` -- silently both times -- so
    the config and every scene's files are checked before a renderer exists.

    The release's ``category_to_mp3d_category_id`` must also be mpcat40.tsv's
    indices: the oracle labels goal classes by mpcat40 name.

    Returns:
        A list of complaints, empty when every annotation is present.
    """
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.ground_truth.mp3d_categories import check_release
    try:
        if not dataset.category_to_mp3d_category_id:
            raise ValueError("the release has no category_to_mp3d_category_id to tie its goals to mpcat40 classes")
        check_release(dataset.category_to_mp3d_category_id)
    except ValueError as exc:
        return [str(exc)]
    try:
        dataset.semantic_scene_dataset_config()
    except (OSError, ValueError) as exc:
        return [str(exc)]
    seen, missing = set(), []
    episodes = (dataset.episodes[i] for i in episode_ids) if episode_ids is not None else dataset.episodes.values()
    for episode in episodes:
        if episode.scene_key in seen:
            continue
        seen.add(episode.scene_key)
        try:
            dataset.semantic_assets(episode)
        except (OSError, ValueError) as exc:
            missing.append(str(exc))
    if not missing:
        return []
    return ["%d of %d selected scenes lack their Matterport3D annotations (first: %s)" % (len(missing), len(seen), missing[0])]


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
                  or args.episodes_per_scene or args.category or args.episode_ids)
    unbounded_development = development and args.episodes_per_scene is None
    if unbounded_development:
        issues.append("A development split needs --episodes-per-scene: MP3D's "
                      "train split holds millions of episodes, not thousands")
    if args.seed < 0 or args.gpu_device < 0:
        issues.append("--seed and --gpu-device must be non-negative")
    if args.agent == "stop" and args.limit is None:
        issues.append("The stop diagnostic requires an explicit --limit")
    if args.shuffle and args.limit is None and not args.episode_ids:
        issues.append("--shuffle is a seeded random sample and needs --limit N")
    if not 1 <= args.video_fps <= 60:
        issues.append("--video-fps must be between 1 and 60")
    if args.save_frames and not args.record:
        issues.append("--save-frames keeps the recorder's frames and so needs --record")
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
        gpu_gate(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        issues.append("Rendering GPU: %s" % exc)
    runtime = runtime_record(issues)
    sim_match = runtime.get("habitat-sim") == protocol.reference_habitat_sim_version
    oracle = args.perception == PERCEPTION_GROUND_TRUTH
    ground_truth = None
    dataset = env = policy = None
    ids, method, manifest = (), {}, None
    excluded, start_distances = [], {}
    if not args.episodes_dir or not args.scenes_dir:
        issues.append("Set --episodes-dir to the split directory and --scenes-dir "
                      "to the directory holding mp3d/<scene>/")
    elif unbounded_development:
        pass  # loading it to find out how big it is would BE the problem
    else:
        try:
            dataset = MP3DDataset(args.episodes_dir, args.scenes_dir, protocol,
                                  full=not subset, scenes=args.scenes,
                                  episodes_per_scene=args.episodes_per_scene)
            manifest = dataset.manifest(hash_scenes=not args.skip_scene_hashes)
            if oracle:
                # The annotations are checked before a renderer exists: habitat-sim says nothing when they are absent.
                annotation_issues = ground_truth_issues(dataset)
                issues.extend("Ground-truth perception: %s" % issue for issue in annotation_issues)
                if not annotation_issues:
                    from sparx_agency.tasks.planning.objnav_benchmark_runtime.ground_truth.mp3d_categories import MP3D_TABLE
                    from sparx_agency.tasks.planning.objnav_benchmark_runtime.ground_truth.source import GroundTruthSource
                    ground_truth = GroundTruthSource(mp3d_label_mapper(), MP3D_TABLE)
            if not oracle or ground_truth is not None:
                # habitat-sim matches gpu_device_id against the EGL devices' CUDA ids; -1 skips the match
                # and takes EGL device 0, which the software-render launch makes Mesa's llvmpipe.
                env = MP3DEnv(dataset, protocol, args.seed,
                              -1 if args.software_render else args.gpu_device, ground_truth=ground_truth)
            pool = episodes_of_category(dataset.episode_ids() if env is None else env.episode_ids(),
                                        dataset.episodes, args.category, CATEGORIES)
            if args.episode_ids:
                ids = explicit_episodes(args.episode_ids, pool)
            else:
                ids = select_episodes(pool, args.limit, args.shards, args.shard_index,
                                      shuffle_seed=args.seed if args.shuffle else None)
            if args.validate_starts and env is not None:
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
        if oracle and ground_truth is None:
            raise RuntimeError("ground-truth perception has no annotations to read (see the issues above)")
        policy, method = _method(args, protocol, ground_truth)
        if args.agent in ("rpt", "falcon-yolo"):
            policy.target_projector = getattr(env, "target_projector", None)
            method["target_navmesh_projection"] = policy.target_projector is not None
    except Exception as exc:
        issues.append("Method services/configuration: %s" % exc)
    try:
        gpu_plan = gpu_plan_record()
    except RuntimeError as exc:
        issues.append(str(exc))
        gpu_plan = None
    config = {
        "protocol": asdict(protocol),
        "protocol_id": protocol.protocol_id,
        "runtime": runtime,
        "reference_habitat_sim_version_match": sim_match,
        "method": method,
        "publishable": method.get("publishable", True),
        "seed": args.seed,
        "gpu_device": args.gpu_device,
        "renderer": ("software (Mesa llvmpipe EGL device; --software-render)" if args.software_render
                     else "cuda gpu %d" % args.gpu_device),
        "kinematics": asdict(protocol.kinematics()),
        "gpu_plan": gpu_plan,
        "shards": args.shards,
        "shard_index": args.shard_index,
        "limit": args.limit,
        "scenes": list(args.scenes or ()),
        "episodes_per_scene": args.episodes_per_scene,
        "category": args.category,
        "shuffle": bool(args.shuffle),
        "episode_ids": list(args.episode_ids) if args.episode_ids else None,
        "lean_records": bool(args.lean),
        "allow_shared_gpu": args.allow_shared_gpu,
        "recording": {"enabled": args.record, "fps": args.video_fps, "frames": bool(args.save_frames)},
        "split": protocol.split,
        "published_split": not development,
        "starts_validated": bool(args.validate_starts),
        "excluded_unreachable_episode_ids": sorted(excluded),
        "full_split": not subset and not excluded,
        "dataset": manifest,
    }
    if start_distances:
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
                complaint = geodesic_agreement_issue(
                    agreement, allow=args.allow_geodesic_mismatch)
                if complaint is not None:
                    issues.append(complaint)
                config["published_geodesic_agreement"]["accepted"] = complaint is None
            if env.navmesh_by_scene:
                config["navmesh_by_scene"] = dict(sorted(env.navmesh_by_scene.items()))
    config["selected_episode_ids"] = list(ids)
    if oracle:
        # Only the oracle's record carries the key, as on HM3D: a service run's config stays what it was.
        config["perception"] = PERCEPTION_GROUND_TRUTH
    return env, policy, mp3d_label_mapper(), config, issues


def main(argv=None, *, frozen_lock=None, expected_episode_ids=None):
    """The MP3D entrypoint: the shared Habitat ``run_main`` with this adapter's parser, table, protocols, preflight and report."""
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.report import write_report
    return run_main(argv, benchmark="mp3d", parser=parser, label_mapper=mp3d_label_mapper, protocol_for=protocol_for,
                    prepare=prepare, write_report=write_report, frozen_lock=frozen_lock,
                    expected_episode_ids=expected_episode_ids)


if __name__ == "__main__":
    raise SystemExit(main())
