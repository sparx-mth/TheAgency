"""Did the fine-tune help? Paired, on held-out geometry, against the true map.

    python -m ...world_goal.evaluate --dataset ~/navdp_world_goal/dataset \
        --run ~/navdp_world_goal/run1 --features ~/navdp_world_goal/features

Three arms answer three different questions, all on the **test** split -- a wing
of the building that neither training nor checkpoint selection ever touched:

``baseline``  the pretrained NavDP, as shipped. The thing to beat.
``trained``   the fine-tuned weights (EMA by default).
``expert``    the label itself. Not a competitor: it is the ceiling imitation
              can reach, and the gap to it says whether the remaining error is
              the student's or the teacher's.

Every arm answers the *same* (frame, goal) pairs with the *same* diffusion noise
seed, so the comparison is paired and the only variable is the weights. Results
are reported per metric with a Wilcoxon signed-rank test and a rank-biserial
effect size, and win/loss counts alongside the mean -- a mean gain assembled
from a few large wins and many small regressions is not a safety improvement.

Then everything is reported again **split by how much the label turns**, because
"safer on straight corridors, worse at corners" and "safer everywhere" produce
the same overall mean and are completely different outcomes.

Writes ``per_sample.csv``, ``evaluation.json``, and BEV route figures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from sparx_agency.tasks.planning.vlas.navdp.finetune.eval import stats
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal import metrics as M
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.dataset import (
    DatasetConfig, WorldGoalDataset,
)
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.polyline import decode_action
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.infer import NavDPRunner
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.model import (
    WorldGoalModelConfig, WorldGoalNavDP,
)
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.scene import (
    Scene, SceneConfig,
)

REPORTED = ("min_clear_m", "p5_clear_m", "frac_below_safe", "centre_offset_m",
            "goal_progress_m", "goal_gap_m", "bending", "mean_clear_m")
TURN_BUCKETS = (("straight", 0.0, 15.0), ("gentle", 15.0, 40.0), ("sharp", 40.0, 181.0))


def encode_batch(model: WorldGoalNavDP, batch: Dict, device: str) -> torch.Tensor:
    """Scene embedding for a batch, from cached tokens or live pixels."""
    if "rgb_tokens" in batch:
        return model.encode_tokens(batch["rgb_tokens"].to(device).float(),
                                   batch["depth_tokens"].to(device).float())
    return model.encode(batch["images"].to(device), batch["depth"].to(device))


@torch.no_grad()
def run_arm(model: WorldGoalNavDP, loader: DataLoader, scenes: Sequence[Scene],
            device: str, sample_num: int, seed: int, max_batches: int,
            d_safe_m: float) -> Tuple[List[Dict], List[np.ndarray]]:
    """Score one set of weights over the loader.

    Returns:
        ``(rows, trajectories)`` -- per-sample metric dicts (plus the sample's
        identifiers) and the body-frame trajectories, kept so the figures can
        draw the same samples for every arm.
    """
    runner = NavDPRunner(model, sample_num=sample_num)
    rows: List[Dict] = []
    trajectories: List[np.ndarray] = []
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        rgbd = encode_batch(model, batch, device)
        result = runner.run(rgbd, batch["goal"].to(device), seed=seed + index)
        paths = result.trajectory.float().cpu().numpy()
        poses = batch["pose"].numpy()
        goals = batch["goal_world"].numpy()
        for item in range(paths.shape[0]):
            scene = scenes[int(batch["scene"][item])]
            scored = M.score(paths[item], poses[item], goals[item], scene, d_safe_m)
            rows.append({**scored.as_dict(),
                         "turn_deg": float(batch["turn_deg"][item]),
                         "goal_kind": int(batch["goal_kind"][item]),
                         "critic_best": float(result.critic[item].max())})
            trajectories.append(paths[item])
    return rows, trajectories


def run_expert(loader: DataLoader, scenes: Sequence[Scene], max_batches: int,
               d_safe_m: float) -> Tuple[List[Dict], List[np.ndarray]]:
    """Score the labels themselves -- the ceiling any imitation can reach."""
    rows: List[Dict] = []
    trajectories: List[np.ndarray] = []
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        actions = batch["action"].numpy()
        poses, goals = batch["pose"].numpy(), batch["goal_world"].numpy()
        for item in range(actions.shape[0]):
            path = decode_action(actions[item])
            scene = scenes[int(batch["scene"][item])]
            scored = M.score(path, poses[item], goals[item], scene, d_safe_m)
            rows.append({**scored.as_dict(),
                         "turn_deg": float(batch["turn_deg"][item]),
                         "goal_kind": int(batch["goal_kind"][item]),
                         "critic_best": float("nan")})
            trajectories.append(path.astype(np.float32))
    return rows, trajectories


def bucketed(baseline: List[Dict], trained: List[Dict]) -> Dict[str, Dict]:
    """Paired results per turn bucket -- corners are where policies actually fail."""
    out: Dict[str, Dict] = {}
    turns = np.array([row["turn_deg"] for row in baseline])
    for name, low, high in TURN_BUCKETS:
        keep = np.flatnonzero((turns >= low) & (turns < high))
        if keep.size < 12:
            out[name] = {"n": int(keep.size), "note": "too few samples to test"}
            continue
        results = stats.compare_all([baseline[i] for i in keep],
                                    [trained[i] for i in keep],
                                    REPORTED, M.HIGHER_IS_BETTER)
        out[name] = {"n": int(keep.size),
                     **{m: {"baseline": r.ref_mean, "trained": r.arm_mean,
                            "delta": r.mean_delta, "p": r.p_value,
                            "effect": r.effect_size, "verdict": r.verdict}
                        for m, r in results.items()}}
    return out


def serialise(result) -> Dict:
    """A :class:`PairedResult` as JSON, *including* its derived verdict.

    ``verdict`` and ``significant`` are properties rather than fields, so a bare
    ``__dict__`` silently drops exactly the two values the report renders.
    """
    return {**result.__dict__, "verdict": result.verdict,
            "significant": bool(result.significant)}


def write_csv(path: Path, arms: Dict[str, List[Dict]]) -> None:
    """One row per (sample, arm), so any of this can be re-analysed elsewhere."""
    import csv

    first = next(iter(arms.values()))
    fields = ["arm", "sample"] + sorted(first[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for arm, rows in arms.items():
            for index, row in enumerate(rows):
                writer.writerow({"arm": arm, "sample": index, **row})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True, help="training output directory")
    parser.add_argument("--checkpoint", default="best.pth")
    parser.add_argument("--weights", default="ema", choices=("ema", "model"))
    parser.add_argument("--features", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--ckpt", default="~/Downloads/navdp-cross-modal.ckpt")
    parser.add_argument("--navdp-repo", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=60)
    parser.add_argument("--sample-num", type=int, default=16)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--d-safe", type=float, default=0.5)
    parser.add_argument("--figures", type=int, default=12,
                        help="how many sample routes to draw (0 disables)")
    args = parser.parse_args()

    run = Path(args.run).expanduser()
    dataset = WorldGoalDataset(args.dataset, args.split,
                               DatasetConfig(cache_dir=args.features))
    if not len(dataset):
        raise SystemExit(f"split {args.split!r} is empty")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, drop_last=False)
    scenes = [Scene.load(SceneConfig(**entry)) for entry in dataset.index["scenes"]]
    print(f"[eval] {dataset.describe()}", flush=True)

    model = WorldGoalNavDP(args.ckpt, args.navdp_repo, device=args.device,
                           config=WorldGoalModelConfig()).to(args.device).eval()

    print("[eval] arm 1/3: pretrained baseline", flush=True)
    base_rows, base_paths = run_arm(model, loader, scenes, args.device,
                                    args.sample_num, args.seed, args.max_batches,
                                    args.d_safe)

    checkpoint = torch.load(run / args.checkpoint, map_location="cpu",
                            weights_only=False)
    changed = model.load_trainable(checkpoint[args.weights], strict=True)
    print(f"[eval] arm 2/3: {args.checkpoint}:{args.weights} "
          f"(step {checkpoint.get('step')}, {changed} tensors changed)", flush=True)
    trained_rows, trained_paths = run_arm(model, loader, scenes, args.device,
                                          args.sample_num, args.seed,
                                          args.max_batches, args.d_safe)

    print("[eval] arm 3/3: expert labels (the imitation ceiling)", flush=True)
    expert_rows, expert_paths = run_expert(loader, scenes, args.max_batches, args.d_safe)

    results = stats.compare_all(base_rows, trained_rows, REPORTED, M.HIGHER_IS_BETTER)
    ceiling = stats.compare_all(base_rows, expert_rows, REPORTED, M.HIGHER_IS_BETTER)

    print("\n" + stats.format_table(results, "baseline", "trained"), flush=True)
    print(f"\ncollision rate   baseline {stats.collision_rate(base_rows):.1%}   "
          f"trained {stats.collision_rate(trained_rows):.1%}   "
          f"expert {stats.collision_rate(expert_rows):.1%}", flush=True)
    print("\nBy how hard the label turns (this is where policies fail):", flush=True)
    buckets = bucketed(base_rows, trained_rows)
    for name, entry in buckets.items():
        if "note" in entry:
            print(f"  {name:<9} n={entry['n']:<5} {entry['note']}", flush=True)
            continue
        clear, gap = entry["min_clear_m"], entry["goal_gap_m"]
        print(f"  {name:<9} n={entry['n']:<5} min_clear {clear['baseline']:+.3f} -> "
              f"{clear['trained']:+.3f} ({clear['delta']:+.3f}, {clear['verdict']})"
              f"   goal_gap {gap['baseline']:.2f} -> {gap['trained']:.2f}", flush=True)

    payload = {
        "split": args.split, "checkpoint": str(run / args.checkpoint),
        "weights": args.weights, "samples": len(base_rows),
        "sample_num": args.sample_num, "seed": args.seed,
        "summary": {"baseline": M.summarise([M.TrajectoryMetrics(
                        **{k: r[k] for k in M.TrajectoryMetrics.__annotations__})
                        for r in base_rows]),
                    "trained": M.summarise([M.TrajectoryMetrics(
                        **{k: r[k] for k in M.TrajectoryMetrics.__annotations__})
                        for r in trained_rows]),
                    "expert": M.summarise([M.TrajectoryMetrics(
                        **{k: r[k] for k in M.TrajectoryMetrics.__annotations__})
                        for r in expert_rows])},
        "paired_vs_baseline": {m: serialise(r) for m, r in results.items()},
        "ceiling_vs_baseline": {m: serialise(r) for m, r in ceiling.items()},
        "by_turn": buckets,
        "collision_rate": {"baseline": stats.collision_rate(base_rows),
                           "trained": stats.collision_rate(trained_rows),
                           "expert": stats.collision_rate(expert_rows)},
    }
    (run / "evaluation.json").write_text(json.dumps(payload, indent=2, default=float))
    write_csv(run / "per_sample.csv", {"baseline": base_rows, "trained": trained_rows,
                                       "expert": expert_rows})
    print(f"\n[eval] wrote {run}/evaluation.json and per_sample.csv", flush=True)

    if args.figures > 0:
        from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal import figures
        path = figures.route_panels(
            run / "routes.png", dataset, scenes,
            {"baseline": base_paths, "trained": trained_paths, "expert": expert_paths},
            count=args.figures)
        print(f"[eval] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
