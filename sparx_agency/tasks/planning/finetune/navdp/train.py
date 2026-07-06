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


def _forward_loss(model, loss_fn, batch, device, n_steps, scheduler, l2sp, gen=None):
    """One forward pass -> (total tensor, parts dict of float act/critic/esdf/total).

    ``gen=None`` draws fresh noise (training); a seeded generator makes the
    validation loss reproducible so it is comparable across intervals.
    """
    b = batch["images"].shape[0]
    images = batch["images"].to(device)
    depth = batch["depth"].to(device)
    goal = batch["goal"].to(device)
    x0 = batch["label"].to(device)                        # (B,24,3)
    sdf = batch["sdf_grid"].to(device)
    res = float(batch["resolution"][0]); ox = float(batch["origin_x"][0]); oy = float(batch["origin_y"][0])

    rgbd = model.encode(images, depth)
    goal_embed = model.goal_embed(goal)
    if gen is None:
        noise = torch.randn_like(x0)
        k = torch.randint(0, n_steps, (b,), device=device)
    else:
        noise = torch.randn(x0.shape, generator=gen, device=device, dtype=x0.dtype)
        k = torch.randint(0, n_steps, (b,), generator=gen, device=device)
    x_k = scheduler.add_noise(x0, noise, k)
    pred = model.predict_noise(x_k, k, goal_embed, rgbd)

    parts = {"act": loss_fn.diffusion_loss(pred, noise)}
    if loss_fn.config.use_critic:
        v_tgt = loss_fn.critic_target_from_sdf(x0, sdf, res, ox, oy)
        parts["critic"] = loss_fn.critic_loss(model.predict_critic(x0, rgbd), v_tgt)
    x0_hat = model.x0_from_eps(x_k, k, pred)
    parts["esdf"] = loss_fn.esdf_loss(x0_hat, sdf, res, ox, oy)
    total = loss_fn.total(parts) + l2sp.penalty(model.policy)
    # detach before float() -- these tensors carry grad; float() alone warns
    flat = {kk: float(vv.detach()) for kk, vv in parts.items()}
    flat["total"] = float(total.detach())
    return total, flat


@torch.no_grad()
def _validate(model, loss_fn, loader, device, n_steps, scheduler, l2sp, max_batches):
    """Mean loss over up to ``max_batches`` of the val set (fixed noise seed)."""
    training = model.training
    model.eval()
    gen = torch.Generator(device=device).manual_seed(20260101)
    acc, n = {}, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        _, parts = _forward_loss(model, loss_fn, batch, device, n_steps, scheduler, l2sp, gen=gen)
        for kk, vv in parts.items():
            acc[kk] = acc.get(kk, 0.0) + vv
        n += 1
    if training:
        model.train()
    return {kk: vv / max(n, 1) for kk, vv in acc.items()}


_HDR = "  step | epoch |    lr    | train_tot | val_act val_crit val_esdf | val_tot | best_val | status"


def train_loop(cfg: dict, model, loss_fn, ds, device: str = "cuda", val_ds=None) -> None:
    """DDPM eps-MSE + critic + ESDF + L2-SP loop with validation-tracked logging.

    Every ``optim.val_every`` steps and at each epoch end it logs one clean row:
    running train loss, held-out validation loss (fixed-noise, comparable across
    intervals), and whether it improved -- saving ``best.pth`` on every new-best
    validation and ``ema_latest.pth`` each time. ``val_ds=None`` -> train-only log.
    """
    o = cfg["optim"]
    bs = int(o["batch_size"])
    train_loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=4)
    val_loader = (DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=2)
                  if val_ds is not None and len(val_ds) >= bs else None)
    opt = torch.optim.AdamW(model.param_groups(), weight_decay=float(o["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(o["epochs"])))
    ema = ModelEma(model.policy, decay=float(o["ema_decay"]))
    l2sp = L2SP(model.policy, weight=float(o["l2sp_weight"]))
    scheduler = model.scheduler
    n_steps = scheduler.config.num_train_timesteps

    out = Path(cfg["checkpoint"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    max_steps = o.get("max_steps")
    val_every = int(o.get("val_every", 1000))
    val_batches = int(o.get("val_batches", 80))
    spe = max(1, len(train_loader))
    total_steps = max_steps or spe * int(o["epochs"])

    print("train=%d  val=%d  steps/epoch=%d  target_steps=%s  batch=%d  lr=%.1e"
          % (len(ds), len(val_ds) if val_ds is not None else 0, spe, total_steps, bs,
             opt.param_groups[0]["lr"]), flush=True)
    print(_HDR, flush=True)
    print("-" * len(_HDR), flush=True)

    state = {"best": float("inf"), "since": 0, "last": -1}
    run = {}
    step = 0
    done = False

    def checkpoint(epoch_f):
        if step == state["last"]:
            return                                    # already logged this step
        state["last"] = step
        torch.save(ema.state_dict(), out / "ema_latest.pth")
        lr = opt.param_groups[0]["lr"]
        if val_loader is None:
            print("%6d | %5.2f | %.2e | %9.3f |  (no validation set)"
                  % (step, epoch_f, lr, run.get("total", float("nan"))), flush=True)
            return
        val = _validate(model, loss_fn, val_loader, device, n_steps, scheduler, l2sp, val_batches)
        vt = val["total"]
        if vt < state["best"]:
            state["best"], state["since"] = vt, 0
            torch.save(ema.state_dict(), out / "best.pth")
            status = "*** NEW BEST -> best.pth"
        else:
            state["since"] += 1
            status = "no improve x%d" % state["since"]
        print("%6d | %5.2f | %.2e | %9.3f | %7.3f %8.3f %8.3f | %7.3f | %8.3f | %s"
              % (step, epoch_f, lr, run.get("total", float("nan")),
                 val.get("act", 0.0), val.get("critic", 0.0), val.get("esdf", 0.0),
                 vt, state["best"], status), flush=True)

    for epoch in range(int(o["epochs"])):
        model.train()
        for batch in train_loader:
            total, parts = _forward_loss(model, loss_fn, batch, device, n_steps, scheduler, l2sp)
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], float(o["grad_clip"]))
            opt.step()
            ema.update(model.policy)

            step += 1
            for kk, vv in parts.items():
                run[kk] = 0.98 * run.get(kk, vv) + 0.02 * vv
            if step % val_every == 0:
                checkpoint(step / spe)
            if max_steps and step >= max_steps:
                done = True
                break
        sched.step()
        checkpoint(step / spe)                        # always at epoch end
        if done:
            break
    print("-" * len(_HDR), flush=True)
    print("done. best val=%.4f  ->  %s/best.pth   (latest: ema_latest.pth)"
          % (state["best"], out), flush=True)


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
