"""NavDP fine-tune training loop (CLI).

Ties together: :class:`NavDPFinetune` (grad-enabled, frozen backbone),
:class:`NavDPLoss` (diffusion eps-MSE + critic + ESDF hinge), the flight dataset,
L2-SP anti-forgetting, and EMA. Keep the DDPM scheduler identical to inference.

Runs in the ``navdp`` conda env with the external NavDP repo on ``NAVDP_REPO`` and a
recorded flight (see the fine-tune README -- no usable recording exists yet).

    python -m sparx_agency.tasks.planning.finetune.navdp.train \
        --config .../configs/navdp_finetune.yaml \
        --recording <flight_dir> --ckpt <navdp-cross-modal.ckpt> --navdp-repo $NAVDP_REPO
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from ..common.augment import ViewpointAugmentConfig
from ..common.ema import ModelEma
from ..common.esdf_penalty import EsdfPenaltyConfig
from ..common.l2sp import L2SP
from ..datasets.flight_dataset import FlightDataset, FlightDatasetConfig
from ..datasets.recording import load_recording
from .finetune_model import NavDPFinetune, NavDPFinetuneConfig
from .loss import NavDPLoss, NavDPLossConfig


def build_model_loss(cfg: dict, ckpt: str, navdp_repo: str, device: str):
    """Construct the frozen-backbone NavDP model + the fine-tune loss (dataset-free).

    Shared by the pose-based trainer here and the pixel-goal trainer in
    ``train/train_pixel.py`` -- both feed the SAME loop with batches carrying
    ``images/depth/goal/label/sdf_grid/resolution/origin_{x,y}``.
    """
    ft_cfg = NavDPFinetuneConfig(
        train_depth_encoder=cfg["finetune"]["train_depth_encoder"],
        lr_head=float(cfg["finetune"]["lr_head"]),
        lr_backbone=float(cfg["finetune"]["lr_backbone"]),
    )
    model = NavDPFinetune(ckpt, navdp_repo, device=device, config=ft_cfg).to(device)
    lcfg = cfg["loss"]
    loss = NavDPLoss(NavDPLossConfig(
        use_critic=lcfg["use_critic"], critic_weight=float(lcfg["critic_weight"]),
        d_safe_m=float(lcfg["d_safe_m"]), progress_alpha=float(lcfg["progress_alpha"]),
        esdf=EsdfPenaltyConfig(**lcfg["esdf"]),
    )).to(device)
    return model, loss


def build(cfg: dict, recording, ckpt: str, navdp_repo: str, device: str):
    model, loss = build_model_loss(cfg, ckpt, navdp_repo, device)
    aug = ViewpointAugmentConfig(**cfg["augment"]) if cfg["augment"]["enabled"] else None
    ds = FlightDataset(recording, FlightDatasetConfig(
        model="navdp", memory_size=cfg["data"]["memory_size"],
        goal_lookahead=cfg["data"]["goal_lookahead"],
        navdp_horizon=cfg["data"]["navdp_horizon"],
        seed_from_flight=cfg["data"]["seed_from_flight"], augment=aug))
    return model, loss, ds


def train_loop(cfg: dict, model, loss_fn, ds, device: str = "cuda") -> None:
    """Run the DDPM eps-MSE + critic + ESDF + L2-SP loop over any NavDP dataset."""
    o = cfg["optim"]
    loader = DataLoader(ds, batch_size=o["batch_size"], shuffle=True, drop_last=True,
                        num_workers=4)
    opt = torch.optim.AdamW(model.param_groups(), weight_decay=float(o["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=o["epochs"])
    ema = ModelEma(model.policy, decay=float(o["ema_decay"]))
    l2sp = L2SP(model.policy, weight=float(o["l2sp_weight"]))
    scheduler = model.scheduler
    n_steps = scheduler.config.num_train_timesteps

    out = Path(cfg["checkpoint"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    max_steps = o.get("max_steps")            # None -> full epochs
    log_every = int(o.get("log_every", 200))
    save_every_steps = int(o.get("save_every_steps", 2000))
    step = 0
    run = {}                                  # running mean of loss parts (for logging)
    done = False
    for epoch in range(o["epochs"]):
        model.train()
        for batch in loader:
            b = batch["images"].shape[0]
            images = batch["images"].to(device)
            depth = batch["depth"].to(device)
            goal = batch["goal"].to(device)
            x0 = batch["label"].to(device)                    # (B,24,3)
            sdf = batch["sdf_grid"].to(device)
            res = float(batch["resolution"][0]); ox = float(batch["origin_x"][0]); oy = float(batch["origin_y"][0])

            rgbd = model.encode(images, depth)
            goal_embed = model.goal_embed(goal)
            noise = torch.randn_like(x0)
            k = torch.randint(0, n_steps, (b,), device=device)
            x_k = scheduler.add_noise(x0, noise, k)
            pred = model.predict_noise(x_k, k, goal_embed, rgbd)

            parts = {"act": loss_fn.diffusion_loss(pred, noise)}
            if loss_fn.config.use_critic:
                v_tgt = loss_fn.critic_target_from_sdf(x0, sdf, res, ox, oy)
                v_pred = model.predict_critic(x0, rgbd)
                parts["critic"] = loss_fn.critic_loss(v_pred, v_tgt)
            x0_hat = model.x0_from_eps(x_k, k, pred)
            parts["esdf"] = loss_fn.esdf_loss(x0_hat, sdf, res, ox, oy)
            total = loss_fn.total(parts) + l2sp.penalty(model.policy)

            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], float(o["grad_clip"]))
            opt.step()
            ema.update(model.policy)

            step += 1
            for kk, vv in parts.items():
                run[kk] = 0.98 * run.get(kk, float(vv)) + 0.02 * float(vv)
            if step % log_every == 0:
                print("step %d/%s: " % (step, max_steps or "epoch")
                      + " ".join(f"{kk}={vv:.4f}" for kk, vv in run.items()), flush=True)
            if step % save_every_steps == 0:
                torch.save(ema.state_dict(), out / "ema_latest.pth")
            if max_steps and step >= max_steps:
                done = True
                break
        sched.step()
        torch.save(ema.state_dict(), out / f"ema_ep{epoch:03d}.pth")
        torch.save(ema.state_dict(), out / "ema_latest.pth")
        print(f"epoch {epoch} done ({step} steps): "
              + " ".join(f"{kk}={vv:.4f}" for kk, vv in run.items()), flush=True)
        if done:
            break
    torch.save(ema.state_dict(), out / "ema_latest.pth")


def train(cfg: dict, recording, ckpt: str, navdp_repo: str, device: str = "cuda") -> None:
    """Pose-based training entry: build the flight dataset, then run the loop."""
    model, loss_fn, ds = build(cfg, recording, ckpt, navdp_repo, device)
    train_loop(cfg, model, loss_fn, ds, device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--recording", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--navdp-repo", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    rec = load_recording(args.recording)
    train(cfg, rec, args.ckpt, args.navdp_repo, args.device)


if __name__ == "__main__":
    main()
