"""Short pose-free NavDP fine-tune on precomputed pixel-goal labels.

Reuses the exact model, loss, and training loop as the pose-based trainer
(``navdp.train.build_model_loss`` / ``train_loop``) -- only the dataset differs
(:class:`.pixel_dataset.PixelGoalDataset`). Meant for quick iteration on ONE
recording: generate labels, train a few epochs, evaluate, adjust, repeat.

    # 1) labels (once per recording / config)
    python -m ...finetune.train.pixel_labels --rec walk_into --n-per-frame 25
    # 2) short train
    python -m ...finetune.train.train_pixel --labels ~/flight_dataset/walk_into/labels.npz \
        --epochs 3 --out-dir runs/walk_into
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import ConcatDataset, random_split

from ..train import build_model_loss, train_loop
from .pixel_dataset import PixelGoalDataset

_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/navdp_finetune.yaml"
_DEFAULT_CKPT = Path.home() / "Downloads/navdp-cross-modal.ckpt"
_DEFAULT_REPO = Path.home() / "PycharmProjects/NavDP/baselines/navdp"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, nargs="+", required=True,
                    help="one or more labels.npz (multiple recordings are concatenated)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap total optimizer steps (a step budget instead of full epochs)")
    ap.add_argument("--val-frac", type=float, default=0.1,
                    help="fraction of samples held out for validation (0 disables)")
    ap.add_argument("--val-every", type=int, default=1000,
                    help="run validation + log a row every N steps (also at each epoch end)")
    ap.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    ap.add_argument("--ckpt", type=Path, default=_DEFAULT_CKPT)
    ap.add_argument("--navdp-repo", type=Path, default=_DEFAULT_REPO)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ema-decay", type=float, default=None,
                    help="override EMA decay; use ~0.99 for short runs so the "
                         "checkpoint reflects training (config default 0.999)")
    ap.add_argument("--l2sp", type=float, default=None,
                    help="override L2-SP anti-forgetting weight (config default 1e-3; "
                         "raise to ~1e-2 to keep the model closer to pretrained)")
    ap.add_argument("--lr", type=float, default=None,
                    help="override the head learning rate (config default 1e-4)")
    ap.add_argument("--out-dir", type=Path, default=Path("runs/navdp_pixel"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    cfg["optim"]["epochs"] = args.epochs
    cfg["optim"]["batch_size"] = args.batch_size
    if args.ema_decay is not None:
        cfg["optim"]["ema_decay"] = args.ema_decay
    if args.l2sp is not None:
        cfg["optim"]["l2sp_weight"] = args.l2sp
    if args.lr is not None:
        cfg["finetune"]["lr_head"] = args.lr
    if args.max_steps is not None:
        cfg["optim"]["max_steps"] = args.max_steps
    cfg["optim"]["val_every"] = args.val_every
    cfg["checkpoint"]["out_dir"] = str(args.out_dir)

    model, loss_fn = build_model_loss(cfg, str(args.ckpt.expanduser()),
                                      str(args.navdp_repo.expanduser()), args.device)
    parts = [PixelGoalDataset(p.expanduser(), memory_size=cfg["data"]["memory_size"])
             for p in args.labels]
    full = parts[0] if len(parts) == 1 else ConcatDataset(parts)

    # deterministic train/val split (a held-out slice of the sampled goals)
    val_ds = None
    ds = full
    if args.val_frac > 0:
        n_val = int(len(full) * args.val_frac)
        n_train = len(full) - n_val
        gen = torch.Generator().manual_seed(42)
        ds, val_ds = random_split(full, [n_train, n_val], generator=gen)

    print(f"loaded {len(full)} pixel-goal samples from {len(parts)} recording(s) "
          f"-> train {len(ds)} / val {len(val_ds) if val_ds else 0}", flush=True)
    train_loop(cfg, model, loss_fn, ds, args.device, val_ds=val_ds)


if __name__ == "__main__":
    main()
