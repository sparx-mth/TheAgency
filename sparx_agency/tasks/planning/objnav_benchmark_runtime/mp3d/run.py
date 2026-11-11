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
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess

from sparx_agency.core.planning.objnav.agent.headless_agent import HeadlessObjNavAgent
from sparx_agency.core.planning.objnav.labels.datasets.mp3d import mp3d_label_mapper
from sparx_agency.tasks.planning.objnav_benchmark.results_io import default_run_dir
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_options import (
    PERCEPTION_GROUND_TRUTH, add_detector_options)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.evaluation import run_evaluation
from sparx_agency.tasks.planning.objnav_benchmark_runtime.policy_factory import (  # noqa: F401  (StopDiagnostic re-exported for the sibling CLIs)
    AGENT_CHOICES, AGENT_HELP, StopDiagnostic, build_policy)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.progress import (
    LeanHeadlessObjNavAgent, ProgressEnv, ProgressTracker)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.provenance import (
    episodes_of_category, explicit_episodes, gpu_plan_record, select_episodes)
# The two preflight/run-time guards shared with HM3D: both are habitat-lab's
# benchmark family's, not either dataset's (the publisher's geodesic is the same
# quantity as l on both; the node oracle is the method's).
from sparx_agency.tasks.planning.objnav_benchmark_runtime.hm3d.run import (
    geodesic_agreement_issue, oracle_alive)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.dataset import CATEGORIES, MP3DDataset
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.env import MP3DEnv
from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.protocol import (
    PROTOCOLS, protocol_for)

#: The MP3D detector's port. Gibson's is 18095 and HM3D's 18092: the three label
#: tables share words but not their order (and MP3D's has 21 rows), and a
#: YOLO-World service is pinned to the vocabulary it was started with.
DEFAULT_DETECTOR_URL = "http://127.0.0.1:18093"

#: The frozen-config keys a ``--resume`` may differ in when the operator names them
#: (``--resume-accepting-changes``): provenance (``source_sha256``, ``gpu_plan``) and the kinematic
#: self-check (``kinematics``, and ``evaluation``, which nests it) -- none an input to a decision
#: or a score, so the rows before and after are one experiment. As ``hm3d.run``.
RESUMABLE_CONFIG_CHANGES = ("kinematics", "evaluation", "source_sha256", "gpu_plan")

#: Packages the runtime needs, and the distribution each version is read from.
RUNTIME_PACKAGES = (
    ("numpy", "numpy"), ("scipy", "scipy"), ("skimage", "scikit-image"),
    ("cv2", "opencv-python"), ("networkx", "networkx"), ("requests", "requests"),
    ("quaternion", "numpy-quaternion"), ("habitat_sim", "habitat-sim"),
)


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
    p.add_argument("--record-log-only", action="store_true",
                   help="Record every episode's log -- steps.jsonl (pose, action, the method's snapshot and the "
                        "command of every decision), trajectory.csv, the map journal, metrics.json -- under "
                        "recordings/<key>/ and render no dashboard frame: no video, no stills, no FFmpeg. "
                        "What single_storey_failures.py and review_recording.py read; a video of any episode "
                        "can be re-made afterwards by replaying it with its recorded LLM answers. "
                        "Implies --record; exclusive with --save-frames")
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


def _method(args, protocol, ground_truth=None):
    """Build the search policy and prove its services are the ones recorded.

    One builder for every benchmark (``policy_factory.build_policy``). The only
    defaults this entrypoint sets below ``--policy-config`` are the benchmark's
    own agent -- body height and radius -- so every method knob (clearance,
    stop distance, closing, peeks, ...) is ``RPTSettings``' default here
    exactly as it is on Gibson and HM3D. Nothing here came from an MP3D score.
    ``ground_truth`` is the semantic-frame source under ``--perception
    ground-truth``, built by :func:`prepare` beside the environment.
    """
    embodiment = {"body_height_m": protocol.agent_height_m,
                  "body_radius_m": protocol.agent_radius_m}
    return build_policy(args, mp3d_label_mapper(), defaults=embodiment, ground_truth=ground_truth)


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


def _gpu_gate(args):
    """Refuse to share the rendering GPU with a resident model by accident."""
    if args.allow_shared_gpu or getattr(args, "software_render", False):
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
    if args.record_log_only:
        args.record = True          # the log-only recording is a recording without its video
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
    if args.save_frames and args.record_log_only:
        issues.append("--save-frames keeps the dashboard frames; --record-log-only renders none")
    if args.exclude_unreachable and not args.validate_starts:
        issues.append("--exclude-unreachable needs --validate-starts to find them")
    if args.record and not args.record_log_only:
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
        "recording": {"enabled": args.record, "video": bool(args.record and not args.record_log_only),
                      "fps": args.video_fps, "frames": bool(args.save_frames)},
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


def _finite_or_none(value):
    return None if isinstance(value, float) and not math.isfinite(value) else value


def _live_hooks(args, output, scored_env, total, label):
    """The ``wrap_env`` and per-episode callback that keep ``progress.json`` and the diagnostics sidecar."""
    live = {}

    def wrap_env(env, logger):
        live["tracker"] = ProgressTracker(
            args.progress_file or output / "progress.json", total,
            completed=logger.records, label=label)
        return ProgressEnv(env, live["tracker"])

    def progress(index, total_count, row):
        oracle_alive(live, row)
        diagnostics = {key: _finite_or_none(value)
                       for key, value in scored_env.evaluation_diagnostics().items()}
        with (output / "evaluation_diagnostics.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(diagnostics, episode_id=row.episode_id), allow_nan=False) + "\n")
        print(live["tracker"].update(index, total_count, row), flush=True)

    return live, wrap_env, progress


def main(argv=None, *, frozen_lock=None, expected_episode_ids=None):
    args = parser().parse_args(argv)
    if args.print_vocabulary:
        print(",".join(mp3d_label_mapper().vocabulary()))
        return 0
    if args.resume and args.output is None:
        parser().error("--resume requires --output")
    if args.resume_accepting_changes and not args.resume:
        parser().error("--resume-accepting-changes requires --resume")
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

        scope = ", ".join(part for part in (args.category, *(args.scenes or ())) if part)
        live, wrap_env, progress = _live_hooks(
            args, output, env, total, "mp3d %s %s%s" % (
                args.version, protocol.split, " (%s)" % scope if scope else ""))
        summary = run_evaluation(
            env, policy, label_mapper, output=output, config=config,
            settings=protocol.evaluation_settings(), episode_ids=selected,
            resume=args.resume, accept_changes=tuple(args.resume_accepting_changes), frozen_lock=frozen_lock,
            expected_episode_ids=expected_episode_ids, record=args.record,
            record_video=not args.record_log_only,
            video_fps=args.video_fps, save_frames=args.save_frames, progress=progress, wrap_env=wrap_env,
            agent_class=LeanHeadlessObjNavAgent if args.lean else HeadlessObjNavAgent)
        live["tracker"].finish()
        closed = True  # run_evaluation closes the environment itself
        from sparx_agency.tasks.planning.objnav_benchmark_runtime.mp3d.report import write_report
        write_report(output)
        print("Results: %s (%d of %d episodes)"
              % (output, summary.overall.n_episodes, total))
        return 0
    finally:
        if env is not None and not closed:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
