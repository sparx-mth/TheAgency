"""The fine-tune itself.

    python -m ...world_goal.train --dataset ~/navdp_world_goal/dataset \
        --features ~/navdp_world_goal/features --out ~/navdp_world_goal/run1 \
        --ckpt ~/Downloads/navdp-cross-modal.ckpt

Standard diffusion-policy fine-tuning, with four things chosen for this problem
rather than inherited:

* **Validation is a different part of the building**, so "best" means the
  checkpoint that generalises to unseen geometry, not the one that memorised the
  training corridors. The test wing is never opened here at all.
* **Warmup then cosine.** A cold Q-Former at full learning rate destroys the
  pretrained representation in the first few hundred steps; the earlier config
  had a ``warmup_epochs`` key that nothing read.
* **EMA over the trainable weights only**, and evaluated as the shipped model.
  Shadowing NavDP's 91 M frozen parameters would cost 360 MB of an 8 GB card to
  compute the identity function.
* **Five checkpoints, not fifty**: ``best`` (lowest validation loss), ``last``,
  and three evenly spaced milestones, each holding only the ~44.5 M trainable
  tensors. That is enough to see the trajectory of the run and to roll back.

Every term of the objective, both learning rates, the gradient norm, throughput
and the navigation metrics land in ``metrics.jsonl`` at every validation, and
``plots.py`` runs automatically at the end. Runs in the ``navdp`` conda env.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from sparx_agency.tasks.planning.vlas.common.finetune.common.ema import ModelEma
from sparx_agency.tasks.planning.vlas.common.finetune.common.l2sp import L2SP
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.dataset import (
    DatasetConfig, WorldGoalDataset,
)
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.logger import RunLogger
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.loss import (
    SceneField, SceneFields, WorldGoalLoss, WorldGoalLossConfig,
)
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.model import (
    WorldGoalModelConfig, WorldGoalNavDP,
)
from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.scene import (
    Scene, SceneConfig,
)

VAL_SEED = 20260728
AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}


def autocast(dtype):
    """Autocast when a dtype is configured, a no-op context otherwise.

    ``torch.autocast(dtype=None)`` raises rather than disabling itself, so an
    ``amp: fp32`` run needs a real null context, not ``enabled=False``.
    """
    from contextlib import nullcontext
    return nullcontext() if dtype is None else torch.autocast("cuda", dtype=dtype)


def build_fields(index: Dict, device: str) -> SceneFields:
    """Put every scene's signed ESDF on the device, in dataset scene-id order."""
    fields = []
    for entry in index["scenes"]:
        scene = Scene.load(SceneConfig(**entry))
        fields.append(SceneField(scene.sdf, scene.resolution,
                                 scene.grid.origin_x, scene.grid.origin_y, device))
    return SceneFields(fields)


def forward_step(model: WorldGoalNavDP, loss_fn: WorldGoalLoss, batch: Dict,
                 fields: SceneFields, device: str,
                 generator: Optional[torch.Generator] = None):
    """Encode, noise, denoise, and evaluate the objective for one batch."""
    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    if "rgb_tokens" in batch:
        rgbd = model.encode_tokens(batch["rgb_tokens"].float(),
                                   batch["depth_tokens"].float())
    else:
        rgbd = model.encode(batch["images"], batch["depth"])

    x0 = batch["action"]
    steps = model.scheduler.config.num_train_timesteps
    if generator is None:
        noise = torch.randn_like(x0)
        k = torch.randint(0, steps, (x0.shape[0],), device=device)
    else:
        noise = torch.randn(x0.shape, generator=generator, device=device, dtype=x0.dtype)
        k = torch.randint(0, steps, (x0.shape[0],), generator=generator, device=device)

    x_k = model.scheduler.add_noise(x0, noise, k)
    eps = model.predict_noise(x_k, k, model.goal_embed(batch["goal"]), rgbd)
    return loss_fn.compute(model, batch, rgbd, x0, x_k, k, noise, eps, fields, generator)


@torch.no_grad()
def validate(model, loss_fn, loader, fields, device, max_batches: int,
             l2sp: L2SP) -> Dict[str, float]:
    """Mean over a fixed slice of the validation split, with fixed noise.

    The generator is re-seeded every pass so two validations differ only by the
    weights -- otherwise the curve measures the noise draw as much as the model.
    """
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(VAL_SEED)
    accumulated: Dict[str, float] = {}
    count = 0
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        total, parts = forward_step(model, loss_fn, batch, fields, device, generator)
        parts["total"] = float(total.detach()) + float(l2sp.penalty(model.policy).detach())
        for key, value in parts.items():
            accumulated[key] = accumulated.get(key, 0.0) + float(value)
        count += 1
    if was_training:
        model.train()
    return {key: value / max(count, 1) for key, value in accumulated.items()}


def lr_schedule(step: int, warmup: int, total: int, final_fraction: float) -> float:
    """Linear warmup then cosine decay, as a multiplier on each group's base LR."""
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return final_fraction + (1.0 - final_fraction) * cosine


def save_checkpoint(path: Path, model: WorldGoalNavDP, ema: ModelEma, step: int,
                    epoch: float, val: Optional[Dict], config: Dict) -> None:
    """Write trainable weights (raw and EMA) plus enough context to explain them.

    Deliberately *without* optimiser state: these are the checkpoints that get
    shipped and compared, and doubling 356 MB five times over to carry Adam
    moments nobody will ever use is not a good trade. Crash recovery is
    :func:`save_resume` instead.
    """
    torch.save({"model": model.trainable_state_dict(), "ema": ema.state_dict(),
                "step": step, "epoch": epoch, "val": val, "config": config,
                "param_counts": model.param_counts()}, path)


def save_resume(path: Path, model: WorldGoalNavDP, ema: ModelEma,
                optimizer, scheduler, state: Dict, config: Dict) -> None:
    """Write everything needed to carry on exactly where the run left off.

    Written at every validation and overwritten in place, so a run killed by a
    power cut, an OOM or a reboot costs at most ``val_every`` steps instead of
    everything. That is not hypothetical: this run lost three and a half hours
    at 57 % to a power outage, and the shipped checkpoints could not resume it
    because they carry no optimiser state.

    Written to a temporary file and renamed, because the process dying *during*
    the write is exactly the case this exists for and a half-written resume file
    is worse than none.
    """
    payload = {
        "model": model.trainable_state_dict(), "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "state": dict(state), "config": config,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_resume(path: Path, model: WorldGoalNavDP, ema: ModelEma, optimizer,
                scheduler, state: Dict, logger) -> bool:
    """Restore a run from ``path``. Returns whether anything was restored.

    Accepts a full resume file *or* one of the shipped checkpoints. The shipped
    ones carry no optimiser state, so restarting from one rebuilds Adam's
    moments from scratch -- a transient of a few hundred steps -- but it still
    recovers the weights and, importantly, the position in the cosine schedule,
    which is the part that cannot be guessed.
    """
    if not path.is_file():
        return False
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model.load_trainable(blob["model"])
    ema.load_state_dict(blob["ema"])
    restored = dict(blob.get("state") or {})
    if not restored:                       # a shipped checkpoint, not a resume file
        # Carry the validation loss forward as the bar to beat. Starting from
        # infinity would make the very first validation "the best" and overwrite
        # best.pth -- and after a resume from a shipped checkpoint the optimiser
        # restarts, so that first score is usually a transient *worse* than the
        # weights being resumed from. The run would quietly replace a good
        # checkpoint with a worse one on its first validation.
        previous = (blob.get("val") or {}).get("total")
        restored = {"step": int(blob.get("step", 0)), "stale": 0,
                    "best": float(previous) if previous is not None else float("inf")}
    state.update(restored)
    if "optimizer" in blob:
        optimizer.load_state_dict(blob["optimizer"])
        scheduler.load_state_dict(blob["scheduler"])
        logger.note(f"resumed from {path.name} at step {state['step']} "
                    f"with optimiser state")
    else:
        # Fast-forward the schedule by hand so the learning rate carries on
        # decaying from where it was rather than restarting at the peak.
        for _ in range(int(state["step"])):
            scheduler.step()
        logger.note(f"resumed from {path.name} at step {state['step']}; it carries "
                    f"no optimiser state, so Adam's moments restart and the "
                    f"learning-rate schedule was fast-forwarded instead")
    return True


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default=str(here / "configs" / "navdp_world_goal.yaml"))
    parser.add_argument("--ckpt", default="~/Downloads/navdp-cross-modal.ckpt")
    parser.add_argument("--navdp-repo", default=None)
    parser.add_argument("--features", default=None,
                        help="feature cache from cache_features.py (much faster)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=None,
                        help="steps between validation passes; a short run wants "
                             "this small enough to give a readable curve")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="carry on from a previous run. Bare --resume takes "
                             "<out>/resume.pth; give a path to start from a "
                             "specific checkpoint (a shipped best.pth works, but "
                             "carries no optimiser state)")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    for key, value in (("epochs", args.epochs), ("batch_size", args.batch_size)):
        if value is not None:
            config["optim"][key] = value
    if args.val_every is not None:
        config["eval"]["val_every"] = args.val_every
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_config = DatasetConfig(cache_dir=args.features, **config["data"])
    train_set = WorldGoalDataset(args.dataset, "train", data_config)
    val_set = WorldGoalDataset(args.dataset, "val", data_config)
    if not len(train_set) or not len(val_set):
        raise SystemExit(f"empty split: train={len(train_set)} val={len(val_set)}")

    model_config = WorldGoalModelConfig(**config["model"])
    model = WorldGoalNavDP(args.ckpt, args.navdp_repo, device=args.device,
                           config=model_config).to(args.device)
    if args.features and model.depth_encoder_trainable:
        raise SystemExit(
            "the feature cache is only valid while the depth trunk is frozen; "
            "drop --features for a stage-2 run, or set train_depth_encoder: false")
    loss_fn = WorldGoalLoss(WorldGoalLossConfig(**config["loss"])).to(args.device)
    fields = build_fields(train_set.index, args.device)

    optim = config["optim"]
    batch_size = int(optim["batch_size"])
    workers = int(optim["num_workers"])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=workers, pin_memory=True,
                              persistent_workers=workers > 0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            drop_last=False, num_workers=max(1, workers // 2),
                            pin_memory=True, persistent_workers=workers > 0)

    steps_per_epoch = max(1, len(train_loader) // max(1, int(optim["accum_steps"])))
    total_steps = args.max_steps or steps_per_epoch * int(optim["epochs"])
    optimizer = torch.optim.AdamW(model.param_groups(),
                                  weight_decay=float(optim["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_schedule(s, int(optim["warmup_steps"]), total_steps,
                                         float(optim["lr_final_fraction"])))
    trainable = {name for name, p in model.policy.named_parameters() if p.requires_grad}
    ema = ModelEma(model.policy, decay=float(optim["ema_decay"]), keys=trainable)
    l2sp = L2SP(model.policy, weight=float(optim["l2sp_weight"]))
    amp_dtype = AMP_DTYPES[str(optim.get("amp", "fp32"))]

    out = Path(args.out).expanduser()
    logger = RunLogger(out, config, extra={
        "dataset": str(Path(args.dataset).expanduser().resolve()),
        "features": args.features, "checkpoint": args.ckpt,
        "dataset_index": train_set.index.get("stats"),
        "split_plan": train_set.index.get("split_plan"),
        "param_counts": model.param_counts(),
        "total_steps": total_steps, "steps_per_epoch": steps_per_epoch,
    }, tensorboard=args.tensorboard)
    counts = model.param_counts()
    logger.note(f"{train_set.describe()} | {val_set.describe()}")
    logger.note(f"trainable {counts['trainable'] / 1e6:.1f} M of "
                f"{counts['total'] / 1e6:.1f} M  |  {total_steps} steps "
                f"({steps_per_epoch}/epoch, batch {batch_size}"
                f"x{optim['accum_steps']})  |  vision="
                f"{'cached tokens' if args.features else 'live pixels'}")
    logger.header()

    milestones = {int(fraction * total_steps): f"milestone_{int(fraction * 100)}"
                  for fraction in config["eval"]["milestones"]}
    state = {"best": float("inf"), "stale": 0, "step": 0}
    if args.resume:
        resume_path = (out / "resume.pth" if args.resume == "auto"
                       else Path(args.resume).expanduser())
        if not load_resume(resume_path, model, ema, optimizer, scheduler, state, logger):
            logger.note(f"--resume given but {resume_path} does not exist; "
                        f"starting from the pretrained checkpoint")
    running: Dict[str, float] = {}
    started, seen, stop = time.time(), 0, False
    resumed_at = int(state["step"])

    for epoch in range(10_000):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for micro, batch in enumerate(train_loader):
            with autocast(amp_dtype):
                total, parts = forward_step(model, loss_fn, batch, fields, args.device)
            objective = (total + l2sp.penalty(model.policy)) / int(optim["accum_steps"])
            objective.backward()
            seen += batch["action"].shape[0]
            for key, value in parts.items():
                running[key] = 0.98 * running.get(key, value) + 0.02 * value

            if (micro + 1) % int(optim["accum_steps"]):
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                float(optim["grad_clip"]))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model.policy)
            state["step"] += 1
            step = state["step"]

            if step % int(config["eval"]["val_every"]) and step != total_steps:
                continue
            val = validate(model, loss_fn, val_loader, fields, args.device,
                           int(config["eval"]["val_batches"]), l2sp)
            epoch_f = step / steps_per_epoch
            note = ""
            if val["total"] < state["best"]:
                state["best"], state["stale"] = val["total"], 0
                save_checkpoint(out / "best.pth", model, ema, step, epoch_f, val, config)
                note = "*** best"
            else:
                state["stale"] += 1
                note = f"no gain x{state['stale']}"
            logger.log("train", step, epoch_f, {
                **{f"train/{k}": v for k, v in running.items()},
                "lr_head": optimizer.param_groups[0]["lr"],
                "grad_norm": float(grad_norm), "samples_per_s": seen / max(time.time() - started, 1e-6),
                "gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1e9
                               if torch.cuda.is_available() else 0.0)})
            logger.log("val", step, epoch_f, val)
            logger.row(step, epoch_f, optimizer.param_groups[0]["lr"],
                       running.get("total", float("nan")), val, note)
            # Overwritten every validation, so an outage costs val_every steps
            # rather than the whole run.
            save_resume(out / "resume.pth", model, ema, optimizer, scheduler,
                        state, config)

            for milestone_step, name in milestones.items():
                if abs(step - milestone_step) < int(config["eval"]["val_every"]) / 2:
                    save_checkpoint(out / f"{name}.pth", model, ema, step, epoch_f,
                                    val, config)
            if state["stale"] >= int(config["eval"]["patience"]) or step >= total_steps:
                stop = True
                break
        if stop:
            break

    save_checkpoint(out / "last.pth", model, ema, state["step"],
                    state["step"] / steps_per_epoch, None, config)
    summary = {"steps": state["step"], "best_val_total": state["best"],
               "resumed_from_step": resumed_at or None,
               "checkpoints": sorted(p.name for p in out.glob("*.pth")),
               "samples_per_s": seen / max(time.time() - started, 1e-6)}
    logger.finish(summary)
    print(json.dumps(summary, indent=2), flush=True)

    if not args.no_plots:
        from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal import plots
        plots.render(out)
        print(f"[train] curves -> {out}/training_curves.png", flush=True)


if __name__ == "__main__":
    main()
