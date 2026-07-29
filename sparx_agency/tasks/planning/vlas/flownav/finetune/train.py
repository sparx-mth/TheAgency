"""FlowNav fine-tune training loop (CLI).

Reproduces FlowNav's flow-matching objective (rectified/OT-CFM, sigma=0) on drone
flights, with the ESDF hinge on the decoded trajectory, L2-SP, and EMA. Fixes the
two shipped bugs (distance-mask reduction order, double LR-scheduler step). Keeps
the action normalization and Euler grid consistent with deployment.

Runs in the ``flownav_trt`` conda env with the external FlowNav repo on
``FLOWNAV_REPO`` and a recorded flight (see the README -- no usable recording yet).

    python -m sparx_agency.tasks.planning.vlas.flownav.finetune.train \
        --config .../configs/flownav_finetune.yaml \
        --recording <flight_dir> --ckpt <flownav_weights.pth> --flownav-repo $FLOWNAV_REPO
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from sparx_agency.tasks.planning.vlas.common.finetune.common.augment import ViewpointAugmentConfig
from sparx_agency.tasks.planning.vlas.common.finetune.common.ema import ModelEma
from sparx_agency.tasks.planning.vlas.common.finetune.common.esdf_penalty import EsdfPenaltyConfig
from sparx_agency.tasks.planning.vlas.common.finetune.common.l2sp import L2SP
from sparx_agency.tasks.planning.vlas.common.finetune.datasets.flight_dataset import FlightDataset, FlightDatasetConfig
from sparx_agency.tasks.planning.vlas.common.finetune.datasets.recording import load_recording
from .finetune_model import FlowNavFinetune, FlowNavFinetuneConfig
from .loss import FlowNavLoss, FlowNavLossConfig


def get_delta(actions: torch.Tensor) -> torch.Tensor:
    """Prepend a zero row and first-difference (FlowNav ``get_delta``)."""
    zero = torch.zeros_like(actions[:, :1])
    prep = torch.cat([zero, actions], dim=1)
    return prep[:, 1:] - prep[:, :-1]


def normalize_data(deltas: torch.Tensor, amin: torch.Tensor, amax: torch.Tensor) -> torch.Tensor:
    """Min-max normalize per-step deltas to [-1, 1] (FlowNav ``normalize_data``)."""
    return (deltas - amin) / (amax - amin) * 2.0 - 1.0


def build(cfg: dict, recording, ckpt: str, flownav_repo: str, device: str):
    ft_cfg = FlowNavFinetuneConfig(
        train_compress=cfg["finetune"]["train_compress"],
        use_film=cfg["finetune"]["use_film"],
        metric_waypoint_spacing=float(cfg["finetune"]["metric_waypoint_spacing"]),
        lr=float(cfg["finetune"]["lr"]),
    )
    model = FlowNavFinetune(ckpt, flownav_repo, device=device, config=ft_cfg).to(device)
    lcfg = cfg["loss"]
    loss = FlowNavLoss(FlowNavLossConfig(
        alpha=float(lcfg["alpha"]), fix_dist_mask=lcfg["fix_dist_mask"],
        esdf=EsdfPenaltyConfig(**lcfg["esdf"]),
    )).to(device)

    aug = ViewpointAugmentConfig(
        enabled=cfg["augment"]["enabled"],
        pitch_deg_range=tuple(cfg["augment"]["pitch_deg_range"]),
        brightness_range=tuple(cfg["augment"]["brightness_range"]),
    ) if cfg["augment"]["enabled"] else None
    ds = FlightDataset(recording, FlightDatasetConfig(
        model="flownav", context_size=cfg["data"]["context_size"],
        goal_lookahead=cfg["data"]["goal_lookahead"],
        flownav_horizon=cfg["data"]["flownav_horizon"],
        metric_waypoint_spacing=float(cfg["data"]["metric_waypoint_spacing"]),
        seed_from_flight=cfg["data"]["seed_from_flight"], augment=aug))
    return model, loss, ds


def train(cfg: dict, recording, ckpt: str, flownav_repo: str, device: str = "cuda") -> None:
    model, loss_fn, ds = build(cfg, recording, ckpt, flownav_repo, device)
    o = cfg["optim"]
    loader = DataLoader(ds, batch_size=o["batch_size"], shuffle=True, drop_last=True,
                        num_workers=4)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=float(cfg["finetune"]["lr"]),
                            weight_decay=float(o["weight_decay"]))
    # single step per epoch (fixes FlowNav's double-step bug)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=o["epochs"])
    ema = ModelEma(model.model, decay=float(o["ema_decay"]))
    l2sp = L2SP(model.model, weight=float(o["l2sp_weight"]))
    horizon = cfg["data"]["flownav_horizon"]
    num_steps = o["num_steps"]
    amin, amax = model.action_min.to(device), model.action_max.to(device)

    out = Path(cfg["checkpoint"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(o["epochs"]):
        model.train()
        for batch in loader:
            obs = batch["obs_img"].to(device)
            goal_img = batch["goal_img"].to(device)
            label = batch["label"].to(device)                 # (B,8,2) absolute wp-units
            distance = batch["distance"].to(device)
            action_mask = batch["action_mask"].to(device)
            sdf = batch["sdf_grid"].to(device)
            res = float(batch["resolution"][0]); ox = float(batch["origin_x"][0]); oy = float(batch["origin_y"][0])
            b = obs.shape[0]

            goal_mask = (torch.rand(b, device=device) < 0.5).long()
            cond = model.encode(obs, goal_img, goal_mask)

            naction = normalize_data(get_delta(label), amin, amax)   # (B,8,2), x1
            noise = torch.randn_like(naction)                        # x0
            t = torch.rand(b, device=device)
            tb = t.view(b, 1, 1)
            xt = (1.0 - tb) * noise + tb * naction
            ut = naction - noise
            vt = model.vfield(xt, t, cond)

            dist_pred = model.distance(cond)
            parts = loss_fn.bc_loss(vt, ut, action_mask, dist_pred, distance, goal_mask)

            wp = model.rollout_waypoints(cond, horizon=horizon, num_steps=num_steps)
            esdf = loss_fn.esdf_loss(wp, sdf, res, ox, oy)
            total = parts["bc"] + esdf + l2sp.penalty(model.model)

            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), float(o["grad_clip"]))
            opt.step()
            ema.update(model.model)
        sched.step()
        if (epoch + 1) % cfg["checkpoint"]["save_every"] == 0:
            torch.save(ema.state_dict(), out / f"ema_{epoch:03d}.pth")
            torch.save(ema.state_dict(), out / "ema_latest.pth")
        print(f"epoch {epoch}: flow={float(parts['flow']):.4f} dist={float(parts['dist']):.4f} esdf={float(esdf):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--flownav-repo", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    rec = load_recording(args.recording)
    train(cfg, rec, args.ckpt, args.flownav_repo, args.device)


if __name__ == "__main__":
    main()
