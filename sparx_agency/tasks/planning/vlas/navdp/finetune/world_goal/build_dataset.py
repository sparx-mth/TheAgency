"""Turn flight recordings + a surveyed map into a labelled, three-way-split dataset.

    python -m sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.build_dataset \
        --recordings ~/sim_flight_recordings_v4 ~/sim_flight_recordings_v3 \
        --scene office --splits <configs>/splits_office.yaml \
        --out ~/navdp_world_goal/dataset --workers 8

What it writes is deliberately *not* a copy of the pixels. Each sample is a
pointer -- recording id, frame index -- plus the goal token and the 24x3 action
label, about 350 bytes. The images stay where they were recorded, which means
relabelling with a different horizon or a different goal mixture costs minutes
and no disk, and the same frame can back a dozen different goals.

Output::

    out/
      index.json          scene config, expert config, sampler config, split plan,
                          per-recording provenance, and the full statistics block
      train/samples.npz   the arrays below
      val/samples.npz
      test/samples.npz

    samples.npz keys:
      recording  (N,)     int32  index into index.json["recordings"]
      frame      (N,)     int32  frame index within that recording
      scene      (N,)     int8   index into index.json["scenes"]
      pose       (N, 3)   f32    world (x, y, yaw) -- how the loss reaches the map
      goal_token (N, 2)   f32    (forward, left) after NavDP's clipping: the input
      goal_world (N, 2)   f32    the true world goal, before clipping
      action     (N,24,3) f32    the diffusion target x0
      goal_kind  (N,)     int8   index into goal_sampler.GOAL_KINDS
      goal_dist  (N,)     f32    body-frame range to the goal, metres
      route_len  (N,)     f32    full expert route length to the goal, metres
      horizon_m  (N,)     f32    arc length the 24 steps actually cover
      min_clear  (N,)     f32    label's own worst clearance on the true map
      mean_clear (N,)     f32
      turn_deg   (N,)     f32    heading deviation inside the horizon
      reaches    (N,)     bool   the label ends at the goal (an arrival sample)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal import polyline
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.expert import (
    ExpertConfig, build_label, make_corrector,
)
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.goal_sampler import (
    GOAL_KINDS, GoalSampler, GoalSamplerConfig, route_ahead_world,
)
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.scene import (
    Scene, SceneConfig,
)
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.sources import (
    RecordingSource, SourceConfig, discover, load_source,
)
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.splits import (
    SPLITS, SplitPlan, load_split_plan,
)

SCHEMA_VERSION = 2
_STATE: Dict = {}

ARRAY_SPEC = (
    ("recording", np.int32), ("frame", np.int32), ("scene", np.int8),
    ("pose", np.float32), ("goal_token", np.float32), ("goal_world", np.float32),
    ("action", np.float32), ("goal_kind", np.int8), ("goal_dist", np.float32),
    ("route_len", np.float32), ("horizon_m", np.float32), ("min_clear", np.float32),
    ("mean_clear", np.float32), ("turn_deg", np.float32), ("reaches", np.bool_),
)


def label_recording(source: RecordingSource, scene: Scene, sampler: GoalSampler,
                    expert_config: ExpertConfig, plan: SplitPlan,
                    seed: int, scene_index: int = 0
                    ) -> Tuple[Dict[str, List[dict]], Dict[str, int]]:
    """Label every admissible frame of one recording.

    Args:
        source: The recording to label.
        scene: The surveyed building **this recording was flown in**. Every
            label, clearance and route comes from it, so handing a recording the
            wrong scene silently supervises it against another building's walls.
        sampler: Goal sampler bound to the same scene.
        expert_config: Label geometry.
        plan: The split plan.
        seed: Campaign seed; combined with the recording and frame index.
        scene_index: Row into ``index.json["scenes"]``, stamped on every sample
            so training can look up the right ESDF (``SceneFields.sample``).

    Returns:
        ``(per_split_samples, rejection_counts)``. A candidate goal is discarded
        -- and counted -- when no route exists, the route is degenerate, the
        label would clip geometry, or the label leaves the frame's split region.
    """
    corrector = make_corrector(expert_config)
    poses = source.recording.poses
    per_split: Dict[str, List[dict]] = {split: [] for split in SPLITS}
    rejects: Dict[str, int] = {}

    def note(reason: str) -> None:
        rejects[reason] = rejects.get(reason, 0) + 1

    for frame in source.frames.tolist():
        pose = (float(poses[frame, 1]), float(poses[frame, 2]), float(poses[frame, 3]))
        split = plan.assign(scene.name, pose[0], pose[1])
        if split is None:
            note("frame_in_split_buffer")
            continue

        rng = np.random.default_rng((seed, source.index, frame))
        ahead = route_ahead_world(poses, frame)
        accepted = 0
        for candidate in sampler.candidates(pose, rng, ahead,
                                            limit=sampler.config.goals_per_frame
                                            * sampler.config.max_attempts_per_goal):
            if accepted >= sampler.config.goals_per_frame:
                break
            label, reason = build_label(scene, pose, (candidate.x, candidate.y),
                                        candidate.kind, expert_config, corrector)
            if label is None:
                note(reason)
                continue
            world = polyline.to_world(label.waypoints_body.astype(np.float64), pose)
            if not plan.route_inside(scene.name, split, world):
                note("label_leaves_split")
                continue
            per_split[split].append({
                "recording": source.index, "frame": frame, "scene": scene_index,
                "pose": np.asarray(pose, np.float32),
                "goal_token": label.goal_token, "goal_world": label.goal_world,
                "action": label.action,
                "goal_kind": GOAL_KINDS.index(label.goal_kind),
                "goal_dist": label.goal_distance_m, "route_len": label.route_length_m,
                "horizon_m": label.horizon_used_m, "min_clear": label.min_clearance_m,
                "mean_clear": label.mean_clearance_m, "turn_deg": label.turn_deg,
                "reaches": label.reaches_goal,
            })
            accepted += 1
    return per_split, rejects


def _worker(index: int):
    """Fork-inherited entry point: label one recording by index.

    Each source carries the index of the building it was flown in, so a
    multi-building campaign labels every recording against its own map.
    """
    state = _STATE
    scene_index = state["scene_of_source"][index]
    return index, label_recording(
        state["sources"][index], state["scenes"][scene_index],
        state["samplers"][scene_index], state["expert"], state["plan"],
        state["seed"], scene_index)


def stack(samples: List[dict]) -> Dict[str, np.ndarray]:
    """Column-stack a list of sample dicts into the on-disk arrays."""
    if not samples:
        return {name: np.zeros((0,), dtype=dtype) for name, dtype in ARRAY_SPEC}
    return {name: np.asarray([s[name] for s in samples], dtype=dtype)
            for name, dtype in ARRAY_SPEC}


def describe(arrays: Dict[str, np.ndarray]) -> Dict:
    """Statistics for one split: size, goal mixture, turn and clearance spread."""
    count = int(arrays["frame"].size)
    if count == 0:
        return {"samples": 0}
    kinds = arrays["goal_kind"]
    percentiles = [5, 25, 50, 75, 95]
    return {
        "samples": count,
        "frames": int(np.unique(np.stack([arrays["recording"], arrays["frame"]]),
                                axis=1).shape[1]),
        "recordings": int(np.unique(arrays["recording"]).size),
        "goal_kinds": {k: int((kinds == i).sum()) for i, k in enumerate(GOAL_KINDS)},
        "goal_dist_m": dict(zip(map(str, percentiles),
                                np.percentile(arrays["goal_dist"], percentiles).round(2).tolist())),
        "turn_deg": dict(zip(map(str, percentiles),
                             np.percentile(arrays["turn_deg"], percentiles).round(1).tolist())),
        "turn_buckets": {
            "straight_lt15": int((arrays["turn_deg"] < 15).sum()),
            "gentle_15_40": int(((arrays["turn_deg"] >= 15) & (arrays["turn_deg"] < 40)).sum()),
            "sharp_ge40": int((arrays["turn_deg"] >= 40).sum()),
        },
        "label_min_clear_m": {
            "min": float(arrays["min_clear"].min()),
            "mean": float(arrays["min_clear"].mean()),
        },
        "arrival_samples": int(arrays["reaches"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--recordings", nargs="+", required=True,
                        help="recording dirs, campaign dirs, or globs")
    parser.add_argument("--scene", default="office",
                        help="single building; ignored when --scenes is given")
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="several surveyed buildings. Each recording is "
                             "labelled against the one it was flown in, which "
                             "meta.json records, so the order here is only the "
                             "order of index.json[\"scenes\"]")
    parser.add_argument("--altitude", type=float, default=1.5)
    parser.add_argument("--map-dir", default=None)
    parser.add_argument("--splits", required=True, help="split-plan YAML")
    parser.add_argument("--out", required=True)
    parser.add_argument("--goals-per-frame", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--horizon-m", type=float, default=4.8)
    parser.add_argument("--no-center", action="store_true",
                        help="skip medial-axis centring (ablation)")
    parser.add_argument("--no-exploration", action="store_true",
                        help="drop the FALCON exploration runs (they are used by default)")
    parser.add_argument("--strict-outcomes", action="store_true",
                        help="use only A-to-B episodes that ended in a clean landing")
    parser.add_argument("--min-label-clearance", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit-recordings", type=int, default=0,
                        help="stop after N recordings (smoke test)")
    args = parser.parse_args()

    scene_names = args.scenes or [args.scene]
    scene_configs = [SceneConfig(scene=name, altitude_m=args.altitude,
                                 map_dir=args.map_dir) for name in scene_names]
    scenes: List[Scene] = []
    for config in scene_configs:
        print(f"[scene] loading {config.scene} @ {args.altitude:.2f} m ...", flush=True)
        loaded = Scene.load(config)
        print(f"[scene] {config.scene}: grid {loaded.grid.height}x{loaded.grid.width} @ "
              f"{loaded.resolution:.2f} m  goal cells {int(loaded.goal_region.sum())}",
              flush=True)
        scenes.append(loaded)
    scene_index_of = {scene.name: i for i, scene in enumerate(scenes)}

    plan = load_split_plan(args.splits)
    for line in plan.describe():
        print(f"[split] {line}", flush=True)

    source_config = SourceConfig(frame_stride=args.frame_stride,
                                 include_exploration=not args.no_exploration,
                                 require_clean_outcome=args.strict_outcomes)
    expert_config = ExpertConfig(horizon_m=args.horizon_m, center=not args.no_center,
                                 min_label_clearance_m=args.min_label_clearance)
    sampler_config = GoalSamplerConfig(goals_per_frame=args.goals_per_frame)
    samplers = [GoalSampler(scene, sampler_config) for scene in scenes]

    # A recording names its own building in meta.json, so each is offered to
    # every loaded scene and kept by the one it was flown in. load_source is
    # what rejects a mismatch, which is why it is the test rather than the
    # directory name -- a campaign directory can be renamed, meta.json cannot.
    sources: List[RecordingSource] = []
    scene_of_source: List[int] = []
    skipped: List[str] = []
    for path in discover(args.recordings):
        reasons = []
        for scene_index, scene in enumerate(scenes):
            source, reason = load_source(path, scene, source_config,
                                         index=len(sources))
            if source is not None:
                sources.append(source)
                scene_of_source.append(scene_index)
                print(f"[source] {source.name:<28} {source.frames.size:5d} frames "
                      f"in {scene.name} "
                      f"({source.meta.get('outcome', 'exploration')})", flush=True)
                break
            reasons.append(reason)
        else:
            skipped.append(f"{path}: {'; '.join(dict.fromkeys(reasons))}")
        if args.limit_recordings and len(sources) >= args.limit_recordings:
            break
    for line in skipped:
        print(f"[skip] {line}", flush=True)
    if not sources:
        raise SystemExit("no usable recordings -- see the [skip] lines above")
    per_scene = {scenes[i].name: scene_of_source.count(i) for i in range(len(scenes))}
    print(f"[source] {len(sources)} recordings: "
          + ", ".join(f"{k} {v}" for k, v in per_scene.items()), flush=True)

    _STATE.update({"sources": sources, "scenes": scenes, "samplers": samplers,
                   "scene_of_source": scene_of_source,
                   "expert": expert_config, "plan": plan, "seed": args.seed})

    per_split: Dict[str, List[dict]] = {split: [] for split in SPLITS}
    rejects: Dict[str, int] = {}
    total_frames = sum(int(s.frames.size) for s in sources)
    print(f"[label] {total_frames} frames x up to {args.goals_per_frame} goals", flush=True)

    def absorb(index: int, result) -> None:
        splits, reasons = result
        for split, rows in splits.items():
            per_split[split].extend(rows)
        for reason, n in reasons.items():
            rejects[reason] = rejects.get(reason, 0) + n
        done = sum(len(v) for v in per_split.values())
        print(f"[label] {sources[index].name:<28} -> {done} samples so far", flush=True)

    if args.workers > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context("fork")) as pool:
            for index, result in pool.map(_worker, range(len(sources))):
                absorb(index, result)
    else:
        for index in range(len(sources)):
            absorb(*_worker(index))

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, Dict] = {}
    for split in SPLITS:
        arrays = stack(per_split[split])
        (out / split).mkdir(exist_ok=True)
        np.savez_compressed(out / split / "samples.npz", **arrays)
        stats[split] = describe(arrays)

    index = {
        "version": SCHEMA_VERSION,
        "scenes": [asdict(config) for config in scene_configs],
        "recordings_per_scene": per_scene,
        "expert": asdict(expert_config),
        "sampler": {**asdict(sampler_config)},
        "sources": asdict(source_config),
        "split_plan": plan.describe(),
        "split_plan_path": str(Path(args.splits).resolve()),
        "seed": args.seed,
        "recordings": [s.summary() for s in sources],
        "skipped": skipped,
        "rejected_goals": dict(sorted(rejects.items(), key=lambda kv: -kv[1])),
        "stats": stats,
    }
    (out / "index.json").write_text(json.dumps(index, indent=2))

    print("\n" + "=" * 72)
    for split in SPLITS:
        entry = stats[split]
        if not entry.get("samples"):
            print(f"  {split:<6} EMPTY -- check the split plan against the flight paths")
            continue
        print(f"  {split:<6} {entry['samples']:7d} samples  {entry['frames']:6d} frames  "
              f"turn>=40deg {entry['turn_buckets']['sharp_ge40']:6d}  "
              f"arrivals {entry['arrival_samples']:6d}")
    print("  rejected goals: " + ", ".join(f"{k}={v}" for k, v in
                                           list(index["rejected_goals"].items())[:6]))
    print(f"  wrote {out}/index.json")
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
