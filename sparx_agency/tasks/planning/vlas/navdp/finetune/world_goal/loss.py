"""The objective: reach the destination, by the safest route, without forgetting.

Six terms. Two of them are the job the user asked for -- *arrive* and *stay
safe* -- and the split between them is deliberate, because a single term cannot
express both without one silently dominating.

``act``       Diffusion behaviour cloning: predict the noise that was added to
              the expert action. This is NavDP's own pretraining objective and
              the only term that speaks the network's native language, so it
              carries the bulk of the learning. On its own it is scale-free
              noise regression and a poor geometric teacher, which is why the
              next two exist.
``waypoint``  The decoded trajectory, in **metres**, against the expert's. This
              is where corridor-centring is actually taught: the expert route
              was already re-centred on the corridor's medial axis, so matching
              it geometrically *is* learning to fly down the middle. Smooth-L1,
              so one bad label cannot dominate a batch.
``clearance`` A hinge on the **global** signed ESDF, sampled at the predicted
              waypoints transformed into world coordinates. This is the hard
              safety floor, and it is the term that could not exist before:
              the previous fine-tune measured clearance on a single depth frame,
              which cannot see round a corner, so it could not penalise a
              trajectory heading into a wall the camera had not met yet. Here
              the map knows.
``goal``      Match the expert's remaining distance to the true world goal. Route
              shape is already covered by ``waypoint``; this term is about
              *progress*, and it survives when two routes round an obstacle are
              equally good and the waypoint term is therefore ambiguous.
``critic``    NavDP picks one of 16 diffusion samples using its critic, so the
              critic decides what actually gets flown. Training it on expert
              trajectories alone teaches it nothing -- it never sees a bad one.
              Here each sample is scored against several deliberately wrong
              trajectories (rotated, straight-lined, pushed sideways), every one
              valued by the true map. That contrast is what makes the ranking
              mean something at inference.
``l2sp``     Applied by the trainer: a pull back toward the pretrained weights.
              A few thousand frames of one office against NavDP's pretraining
              corpus is a recipe for forgetting how to navigate anywhere else.

**Noise-level weighting.** The three geometric terms are evaluated on the
one-step estimate ``x0_hat``, which at high diffusion noise is barely more than
a guess. Penalising the geometry of a guess teaches nothing and adds variance,
so each is weighted by ``alpha_bar_k`` -- full weight where the estimate is
sharp, vanishing where it is not. ``act`` is unweighted; it is well-posed at
every noise level.

Torch only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.scene_field import (
    SceneField, SceneFields,
)


def body_to_world(waypoints: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """``(B, N, 2)`` body FLU waypoints at ``(B, 3)`` world poses -> world ``(B, N, 2)``."""
    cos = torch.cos(pose[:, 2]).unsqueeze(1)
    sin = torch.sin(pose[:, 2]).unsqueeze(1)
    x = pose[:, 0:1] + waypoints[..., 0] * cos - waypoints[..., 1] * sin
    y = pose[:, 1:2] + waypoints[..., 0] * sin + waypoints[..., 1] * cos
    return torch.stack([x, y], dim=-1)


def decode_action(action: torch.Tensor, scale: float = 4.0) -> torch.Tensor:
    """NavDP action ``(B, T, 3)`` -> body-frame waypoints ``(B, T, 2)``."""
    return torch.cumsum(action[..., :2] / scale, dim=1)


def encode_waypoints(waypoints: torch.Tensor, scale: float = 4.0,
                     clamp: float = 1.0) -> torch.Tensor:
    """Body-frame waypoints ``(B, T, 2)`` -> NavDP action ``(B, T, 3)``.

    The inverse of :func:`decode_action`, with the heading channel filled the
    same way the offline label encoder fills it -- a critic candidate whose third
    channel were always zero would be distinguishable from an expert trajectory
    by that alone, and the critic would learn the tell instead of the geometry.
    """
    zero = waypoints.new_zeros((waypoints.shape[0], 1, 2))
    steps = torch.diff(torch.cat([zero, waypoints], dim=1), dim=1)
    headings = torch.atan2(steps[..., 1], steps[..., 0])
    previous = torch.cat([headings.new_zeros((headings.shape[0], 1)),
                          headings[:, :-1]], dim=1)
    delta = torch.atan2(torch.sin(headings - previous), torch.cos(headings - previous))
    action = torch.cat([steps * scale, (delta * scale).unsqueeze(-1)], dim=-1)
    return action.clamp(-clamp, clamp)


@dataclass(frozen=True)
class WorldGoalLossConfig:
    """Term weights and constants.

    Attributes:
        w_act / w_waypoint / w_clearance / w_goal / w_critic: Term weights. The
            defaults put behaviour cloning and geometry on equal footing and
            keep the two shaping terms an order of magnitude below, so they
            refine the imitation instead of replacing it.
        action_scale: NavDP's x4 encoding constant.
        clearance_margin_m: Clearance the hinge drives toward. The airframe is
            0.35 m in radius, so this is the radius plus a real safety margin.
        clearance_worst_weight: Weight on the **worst** waypoint's hinge, added
            to the mean over waypoints. Averaging alone was measured to make this
            whole term 0.7 % of the objective: typical clearance is around 1 m,
            so 23 of 24 waypoints contribute exactly zero and dilute the one that
            matters by 24x. A trajectory with a single waypoint 20 cm from a wall
            is unsafe whatever the other 23 do, and the minimum is what the
            evaluation reports, so the minimum is what the loss should feel.
        clearance_clamp_m: Signed clearance is clamped to +-this before the
            hinge, so one waypoint deep inside a wall cannot dominate a batch.
        waypoint_beta_m / goal_beta_m: Smooth-L1 transition points, metres.
        critic_negatives: Wrong trajectories scored per sample.
        critic_d_safe_m / critic_progress_alpha: The value function's collision
            threshold and its clearance-gain bonus.
        snr_weighting: Weight the geometric terms by ``alpha_bar_k``. Off makes
            them uniform over noise levels, which is noisier and usually worse.
    """

    w_act: float = 1.0
    w_waypoint: float = 1.0
    w_clearance: float = 2.0
    w_goal: float = 0.1
    w_critic: float = 0.1
    action_scale: float = 4.0
    clearance_margin_m: float = 0.55
    clearance_worst_weight: float = 3.0
    clearance_clamp_m: float = 2.0
    waypoint_beta_m: float = 0.10
    goal_beta_m: float = 0.50
    critic_negatives: int = 3
    critic_d_safe_m: float = 0.5
    critic_progress_alpha: float = 0.1
    snr_weighting: bool = True


class WorldGoalLoss(nn.Module):
    """The six-term objective. See the module docstring for what each term is for."""

    def __init__(self, config: Optional[WorldGoalLossConfig] = None) -> None:
        super().__init__()
        self.config = config or WorldGoalLossConfig()

    # ------------------------------------------------------------ value target
    @torch.no_grad()
    def value(self, waypoints_body: torch.Tensor, pose: torch.Tensor,
              scene_ids: torch.Tensor, fields: SceneFields) -> torch.Tensor:
        """Privileged value of a trajectory under the true map, ``(B,)``.

        ``V = (-#{waypoints closer than d_safe} + alpha * total clearance gain) / T``.
        Divided by the horizon so it lands in roughly ``[-1, 0.1]``: the earlier
        fine-tune left this unnormalised, where it reached 30-plus and drowned
        every other term in the sum.
        """
        cfg = self.config
        clearance = fields.sample(body_to_world(waypoints_body, pose), scene_ids)
        collisions = (clearance < cfg.critic_d_safe_m).float().sum(dim=1)
        progress = (clearance[:, 1:] - clearance[:, :-1]).sum(dim=1)
        return (-collisions + cfg.critic_progress_alpha * progress) / waypoints_body.shape[1]

    @torch.no_grad()
    def negatives(self, waypoints_body: torch.Tensor,
                  generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """``(B, n, T, 2)`` deliberately wrong trajectories built from the expert's.

        Three failure modes, cycled: flying the right shape in the wrong
        direction, ignoring the corridor and going straight, and hugging one
        wall. All three are things a diffusion sample actually does, which is
        what makes them useful contrast rather than noise.
        """
        batch, horizon, _ = waypoints_body.shape
        device = waypoints_body.device

        def uniform(*shape) -> torch.Tensor:
            return torch.rand(*shape, device=device, generator=generator)

        out: List[torch.Tensor] = []
        ramp = torch.linspace(0.0, 1.0, horizon, device=device).view(1, horizon, 1)
        lengths = torch.linalg.norm(waypoints_body[:, -1], dim=-1, keepdim=True)
        for index in range(self.config.critic_negatives):
            mode = index % 3
            if mode == 0:                                    # rotated by 20..100 deg
                angle = (0.35 + 1.40 * uniform(batch, 1)) * torch.sign(uniform(batch, 1) - 0.5)
                cos, sin = torch.cos(angle), torch.sin(angle)
                x = waypoints_body[..., 0] * cos - waypoints_body[..., 1] * sin
                y = waypoints_body[..., 0] * sin + waypoints_body[..., 1] * cos
                out.append(torch.stack([x, y], dim=-1))
            elif mode == 1:                                  # straight, arbitrary bearing
                bearing = (uniform(batch, 1) - 0.5) * 2.0
                reach = (lengths * ramp.squeeze(-1))
                out.append(torch.stack([reach * torch.cos(bearing),
                                        reach * torch.sin(bearing)], dim=-1))
            else:                                            # pushed sideways into a wall
                offset = (0.5 + 1.5 * uniform(batch, 1)) * torch.sign(uniform(batch, 1) - 0.5)
                steps = torch.diff(torch.cat(
                    [waypoints_body.new_zeros((batch, 1, 2)), waypoints_body], dim=1), dim=1)
                normal = torch.stack([-steps[..., 1], steps[..., 0]], dim=-1)
                normal = normal / normal.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                out.append(waypoints_body + normal * (offset.unsqueeze(-1) * ramp))
        return torch.stack(out, dim=1)

    # ------------------------------------------------------------------ terms
    def compute(self, model, batch: Dict[str, torch.Tensor], rgbd: torch.Tensor,
                x0: torch.Tensor, x_k: torch.Tensor, k: torch.Tensor,
                noise: torch.Tensor, eps_pred: torch.Tensor, fields: SceneFields,
                generator: Optional[torch.Generator] = None):
        """One full objective evaluation.

        Args:
            model: The :class:`~.model.WorldGoalNavDP` being trained.
            batch: Needs ``pose (B,3)``, ``goal_world (B,2)``, ``scene (B,)``.
            rgbd: ``(B, 128, 384)`` scene embedding, already computed.
            x0: ``(B, T, 3)`` expert action label.
            x_k / k / noise: The noised action, its timestep, and the noise added.
            eps_pred: The network's noise prediction for ``x_k``.
            fields: The scenes' ESDFs on the device.
            generator: Seeded generator, for a reproducible validation pass.

        Returns:
            ``(total, parts)`` -- the scalar to backward through, and a dict of
            detached floats holding every term both raw and weighted, plus two
            diagnostics (``min_clear_m``, ``collide_frac``) that are metrics
            rather than losses and are what the training curves are read from.
        """
        cfg = self.config
        pose, goal_world = batch["pose"], batch["goal_world"]
        scene_ids = batch["scene"]

        # Float32 from here on, whatever autocast is doing. These tensors are
        # tiny (B x 24 x 3) and they get transformed into world coordinates that
        # reach 40 m from the origin -- bfloat16 quantises that to ~0.25 m steps,
        # which is half the clearance margin the hinge is trying to enforce.
        x0_hat = model.x0_from_eps(x_k, k, eps_pred).clamp(-1.0, 1.0).float()
        pred = decode_action(x0_hat, cfg.action_scale)
        target = decode_action(x0.float(), cfg.action_scale)

        weight = torch.ones_like(k, dtype=pred.dtype)
        if cfg.snr_weighting:
            alpha_bar = model.scheduler.alphas_cumprod.to(pred.device)[k].to(pred.dtype)
            weight = alpha_bar / alpha_bar.mean().clamp(min=1e-6)

        act = F.mse_loss(eps_pred.float(), noise.float())

        waypoint = (weight * F.smooth_l1_loss(
            pred, target, beta=cfg.waypoint_beta_m, reduction="none").mean(dim=(1, 2))).mean()

        world = body_to_world(pred, pose)
        clearance = fields.sample(world, scene_ids).clamp(
            -cfg.clearance_clamp_m, cfg.clearance_clamp_m)
        hinge = F.relu(cfg.clearance_margin_m - clearance) ** 2
        per_sample = hinge.mean(dim=1) + cfg.clearance_worst_weight * hinge.max(dim=1).values
        clear = (weight * per_sample).mean()

        gap_pred = torch.linalg.norm(world[:, -1] - goal_world, dim=-1)
        with torch.no_grad():
            gap_target = torch.linalg.norm(
                body_to_world(target, pose)[:, -1] - goal_world, dim=-1)
        goal = (weight * F.smooth_l1_loss(
            gap_pred, gap_target, beta=cfg.goal_beta_m, reduction="none")).mean()

        critic = pred.new_zeros(())
        if cfg.w_critic > 0.0 and cfg.critic_negatives > 0:
            critic = self._critic_term(model, rgbd, x0, target, pose, scene_ids,
                                       fields, generator)

        parts = {"act": act, "waypoint": waypoint, "clearance": clear,
                 "goal": goal, "critic": critic}
        total = (cfg.w_act * act + cfg.w_waypoint * waypoint
                 + cfg.w_clearance * clear + cfg.w_goal * goal + cfg.w_critic * critic)

        with torch.no_grad():
            raw = fields.sample(world, scene_ids)
            flat = {f"raw/{name}": float(value.detach()) for name, value in parts.items()}
            flat.update({
                "total": float(total.detach()),
                "loss/act": cfg.w_act * flat["raw/act"],
                "loss/waypoint": cfg.w_waypoint * flat["raw/waypoint"],
                "loss/clearance": cfg.w_clearance * flat["raw/clearance"],
                "loss/goal": cfg.w_goal * flat["raw/goal"],
                "loss/critic": cfg.w_critic * flat["raw/critic"],
                "metric/min_clear_m": float(raw.min(dim=1).values.mean()),
                "metric/collide_frac": float((raw.min(dim=1).values < 0.0).float().mean()),
                "metric/goal_gap_m": float(gap_pred.mean()),
                "metric/goal_gap_expert_m": float(gap_target.mean()),
            })
        return total, flat

    def _critic_term(self, model, rgbd: torch.Tensor, x0: torch.Tensor,
                     target: torch.Tensor, pose: torch.Tensor, scene_ids: torch.Tensor,
                     fields: SceneFields, generator) -> torch.Tensor:
        """Regress the critic onto map-computed values of expert + negatives."""
        cfg = self.config
        batch, horizon = x0.shape[0], x0.shape[1]
        wrong = self.negatives(target, generator)                     # (B, n, T, 2)
        count = wrong.shape[1] + 1

        actions = torch.cat([x0.unsqueeze(1),
                             encode_waypoints(wrong.reshape(-1, horizon, 2),
                                              cfg.action_scale).reshape(
                                                  batch, -1, horizon, 3)], dim=1)
        flat_actions = actions.reshape(batch * count, horizon, 3)
        flat_pose = pose.repeat_interleave(count, dim=0)
        flat_scene = scene_ids.repeat_interleave(count, dim=0)
        with torch.no_grad():
            values = self.value(decode_action(flat_actions, cfg.action_scale),
                                flat_pose, flat_scene, fields)
        predicted = model.predict_critic(flat_actions,
                                         rgbd.repeat_interleave(count, dim=0))
        return F.mse_loss(predicted.reshape(-1).float(), values.reshape(-1).float())
