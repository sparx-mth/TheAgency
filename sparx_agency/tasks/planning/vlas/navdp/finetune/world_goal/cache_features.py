"""Pre-compute the frozen DINOv2 patch tokens, once, so training never runs a ViT.

    python -m ...world_goal.cache_features --dataset ~/navdp_world_goal/dataset \
        --out ~/navdp_world_goal/features --ckpt ~/Downloads/navdp-cross-modal.ckpt

Nine ViT-S passes per sample -- eight RGB memory frames plus one depth frame --
are about 99 % of a NavDP forward, and in the default freeze policy every one of
those weights is fixed. A fixed function of a fixed frame is worth computing
once. Caching turns a ~1.5 s optimiser step into ~0.05 s on an 8 GB laptop GPU,
which is what makes a 100k-step run possible at all.

The cache is keyed on everything that could change the tokens -- the checkpoint,
the colour order, the depth range, the image size -- and the dataset refuses to
read a cache whose key disagrees with the run asking for it. A silent mismatch
here would be a very expensive kind of wrong.

Layout::

    out/
      meta.json                 the key, plus per-recording row counts
      <recording index>/
        rgb.npy                 (F, 256, 384) float16
        depth.npy               (F, 256, 384) float16
        row_of_frame.npy        (num_frames,) int32, -1 where not cached

Only frames some split actually reads are stored, including each sample's
memory window. About 390 kB per frame; a full office corpus is a few GB.

**Not valid once the depth trunk is unfrozen.** Stage 2 trains
``rgbd_encoder.depth_model``, at which point its output stops being a fixed
function; the trainer detects that and refuses the cached path.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.dataset import (
    DatasetConfig, WorldGoalDataset, merged_frames_needed,
)
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.model import (
    WorldGoalModelConfig, WorldGoalNavDP,
)
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.preprocess import (
    DEPTH_MAX_M, DEPTH_MIN_M, IMAGE_SIZE, preprocess_depth, preprocess_rgb,
)
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.splits import SPLITS

PATCH_TOKENS = 256
TOKEN_DIM = 384


def cache_key(ckpt: str, color_order: str, memory_stride: int) -> Dict:
    """Everything that would change the tokens. Mismatches must be loud."""
    return {
        "checkpoint": str(Path(ckpt).expanduser().resolve()),
        "checkpoint_bytes": Path(ckpt).expanduser().stat().st_size,
        "color_order": color_order,
        "image_size": IMAGE_SIZE,
        "depth_min_m": DEPTH_MIN_M,
        "depth_max_m": DEPTH_MAX_M,
        "memory_stride": memory_stride,
        "token_shape": [PATCH_TOKENS, TOKEN_DIM],
    }


@torch.no_grad()
def cache_recording(model: WorldGoalNavDP, recording, frames: np.ndarray,
                    out_dir: Path, color_order: str, batch_size: int,
                    device: str) -> int:
    """Tokenise and store one recording's needed frames.

    Returns:
        Number of frames written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    count = int(frames.size)
    rgb_store = np.lib.format.open_memmap(
        out_dir / "rgb.npy", mode="w+", dtype=np.float16,
        shape=(count, PATCH_TOKENS, TOKEN_DIM))
    depth_store = np.lib.format.open_memmap(
        out_dir / "depth.npy", mode="w+", dtype=np.float16,
        shape=(count, PATCH_TOKENS, TOKEN_DIM))

    for start in range(0, count, batch_size):
        chunk = frames[start:start + batch_size]
        images = np.stack([preprocess_rgb(recording.rgb(int(f)), color_order)
                           for f in chunk])
        depths = np.stack([preprocess_depth(recording.depth(int(f))) for f in chunk])
        rgb_tokens = model.tokenize_rgb(torch.from_numpy(images).to(device))
        depth_tokens = model.tokenize_depth(torch.from_numpy(depths).to(device))
        rgb_store[start:start + len(chunk)] = rgb_tokens.half().cpu().numpy()
        depth_store[start:start + len(chunk)] = depth_tokens.half().cpu().numpy()

    rgb_store.flush()
    depth_store.flush()
    row_of_frame = np.full(int(recording.num_frames), -1, dtype=np.int32)
    row_of_frame[frames] = np.arange(count, dtype=np.int32)
    np.save(out_dir / "row_of_frame.npy", row_of_frame)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, help="output of build_dataset.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ckpt", default="~/Downloads/navdp-cross-modal.ckpt")
    parser.add_argument("--navdp-repo", default=None,
                        help="external NavDP repo (else $NAVDP_REPO)")
    parser.add_argument("--color-order", default="bgr", choices=("bgr", "rgb"))
    parser.add_argument("--memory-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    args = parser.parse_args()

    config = DatasetConfig(memory_stride=args.memory_stride,
                           color_order=args.color_order)
    datasets: List[WorldGoalDataset] = []
    for split in args.splits:
        dataset = WorldGoalDataset(args.dataset, split, config)
        if len(dataset):
            datasets.append(dataset)
            print(f"[cache] {dataset.describe()}", flush=True)
    if not datasets:
        raise SystemExit(f"no non-empty splits among {args.splits}")

    needed = merged_frames_needed(datasets)
    total = sum(int(v.size) for v in needed.values())
    gigabytes = total * 2 * PATCH_TOKENS * TOKEN_DIM * 2 / 1e9
    print(f"[cache] {total} frames across {len(needed)} recordings "
          f"-> about {gigabytes:.1f} GB", flush=True)

    model = WorldGoalNavDP(args.ckpt, args.navdp_repo, device=args.device,
                           config=WorldGoalModelConfig()).to(args.device).eval()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    recordings = datasets[0].index["recordings"]
    written: Dict[str, int] = {}
    started = time.time()
    done = 0
    for recording_index, frames in needed.items():
        entry = recordings[recording_index]
        recording = datasets[0].recording(recording_index)
        rows = cache_recording(model, recording, frames, out / str(recording_index),
                               args.color_order, args.batch_size, args.device)
        written[str(recording_index)] = rows
        done += rows
        rate = done / max(time.time() - started, 1e-6)
        print(f"[cache] {entry['name']:<28} {rows:5d} frames  "
              f"({done}/{total}, {rate:.0f} frames/s)", flush=True)

    meta = dict(cache_key(args.ckpt, args.color_order, args.memory_stride))
    meta.update({"dataset": str(Path(args.dataset).expanduser().resolve()),
                 "recordings": written, "frames": done})
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[cache] wrote {out} ({done} frames, {time.time() - started:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
