"""Torch dataset: flight recording -> per-model fine-tune samples.

For each frame it (optionally) applies viewpoint augmentation, generates the
PF/ESDF target for the *augmented* depth (so the label stays consistent with the
warped input), encodes the model's action label, and assembles the model inputs and
the SDF grid the differentiable penalty samples.

The input preprocessing here uses the standard resize/normalize; for bit-exact
parity with deployment, swap in the server transforms
(``tasks/planning/navdp/server`` and ``tasks/planning/flownav/server/preprocess``)
-- validate in the model conda env.

Torch + cv2 -- runs in the model conda env.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

try:
    import cv2
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - torch/cv2 absent in the plain .venv
    Dataset = object  # type: ignore

from ..common.augment import ViewpointAugmentConfig, apply_viewpoint_augment
from ..common.esdf_target import EsdfTargetConfig, generate_target
from ..common.frames import LocalMapConfig
from ..common.label_format import to_flownav_label, to_navdp_label
from .recording import FlightRecording

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class FlightDatasetConfig:
    """Sample assembly configuration.

    Attributes:
        model: ``"navdp"`` or ``"flownav"``.
        memory_size: NavDP RGB memory length (8).
        context_size: FlowNav context frames (3 -> 4 stacked).
        goal_lookahead: Frames ahead used as the auto goal.
        navdp_horizon / flownav_horizon: Label horizons.
        metric_waypoint_spacing: FlowNav waypoint-unit scale.
        augment: Viewpoint augmentation (None disables; recommended ON for the
            height gap).
        seed_from_flight: Seed the corrector with the flown future.
    """

    model: str = "navdp"
    memory_size: int = 8
    context_size: int = 3
    goal_lookahead: int = 24
    navdp_horizon: int = 24
    flownav_horizon: int = 8
    metric_waypoint_spacing: float = 0.25
    augment: Optional[ViewpointAugmentConfig] = field(default_factory=ViewpointAugmentConfig)
    seed_from_flight: bool = True


def _letterbox(img: np.ndarray, size: int) -> np.ndarray:
    """Resize keeping aspect, pad to ``size x size`` (NavDP-style)."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, img.shape[2] if img.ndim == 3 else 1), img.dtype).squeeze()
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    return out


def _norm_rgb_96(rgb: np.ndarray) -> np.ndarray:
    """FlowNav transform: center-crop 4:3, resize 96, ImageNet-normalize -> (3,96,96)."""
    img = cv2.resize(rgb, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.transpose(img, (2, 0, 1))


class FlightDataset(Dataset):
    """Per-frame fine-tune samples for NavDP or FlowNav."""

    def __init__(self, recording: FlightRecording, config: FlightDatasetConfig,
                 target_config: Optional[EsdfTargetConfig] = None,
                 seed: int = 0) -> None:
        self.rec = recording
        self.cfg = config
        self.tcfg = target_config or EsdfTargetConfig(
            local_map=LocalMapConfig(camera_height_m=recording.camera_height_m,
                                     pitch_deg=recording.pitch_deg))
        self._base_seed = seed
        # valid frame range needs memory/context history and a future for the label
        lo = max(config.memory_size, config.context_size)
        self._frames = list(range(lo, recording.num_frames - 1))

    def __len__(self) -> int:
        return len(self._frames)

    def _sdf_fields(self, target) -> Dict[str, "torch.Tensor"]:
        return {
            "sdf_grid": torch.from_numpy(target.sdf_m)[None],            # (1,H,W)
            "resolution": torch.tensor(target.occupancy.resolution),
            "origin_x": torch.tensor(target.occupancy.origin_x),
            "origin_y": torch.tensor(target.occupancy.origin_y),
        }

    def __getitem__(self, idx: int) -> Dict[str, "torch.Tensor"]:
        i = self._frames[idx]
        rng = np.random.default_rng(self._base_seed + i)
        depth = self.rec.depth(i)
        rgb = self.rec.rgb(i)

        # viewpoint augmentation (keeps input + label geometry consistent via pitch)
        pitch = self.tcfg.local_map.pitch_deg
        if self.cfg.augment is not None and self.cfg.augment.enabled:
            aug = apply_viewpoint_augment(self.rec.intrinsics, self.cfg.augment, rng,
                                          rgb=rgb, depth_m=depth)
            depth, rgb, pitch = aug.depth_m, aug.rgb, aug.pitch_deg

        goal = self.rec.goal_body(i, self.cfg.goal_lookahead)
        seed = self.rec.future_path_body(i, self.cfg.navdp_horizon) if self.cfg.seed_from_flight else None
        tcfg = EsdfTargetConfig(
            local_map=LocalMapConfig(**{**self.tcfg.local_map.__dict__, "pitch_deg": pitch}),
            corrector=self.tcfg.corrector, target_clearance_m=self.tcfg.target_clearance_m,
            max_total_shift_m=self.tcfg.max_total_shift_m, n_seed_points=self.cfg.navdp_horizon,
            sdf_clamp_m=self.tcfg.sdf_clamp_m)
        target = generate_target(depth, self.rec.intrinsics, goal, tcfg, seed_path=seed)

        sample = {"goal_fwd": torch.tensor(goal[0]), "goal_left": torch.tensor(goal[1])}
        sample.update(self._sdf_fields(target))

        if self.cfg.model == "navdp":
            label = to_navdp_label(target.corrected_path, horizon=self.cfg.navdp_horizon)
            imgs = np.stack([_letterbox(self.rec.rgb(j) if self.rec.rgb(j) is not None else np.zeros((*depth.shape, 3), np.uint8), 224)
                             for j in range(i - self.cfg.memory_size + 1, i + 1)], axis=0)
            imgs = np.transpose(imgs.astype(np.float32) / 255.0, (0, 3, 1, 2))   # (8,3,224,224)
            depth224 = _letterbox(depth[..., None], 224).astype(np.float32)
            depth224 = np.clip(depth224, 0.1, 5.0)[None]                          # (1,224,224)
            sample.update({
                "images": torch.from_numpy(imgs),
                "depth": torch.from_numpy(depth224),
                "goal": torch.tensor([goal[0], goal[1], 0.0], dtype=torch.float32),
                "label": torch.from_numpy(label),                                # (24,3)
            })
        elif self.cfg.model == "flownav":
            label = to_flownav_label(target.corrected_path, horizon=self.cfg.flownav_horizon,
                                     metric_waypoint_spacing=self.cfg.metric_waypoint_spacing)
            ctx = [self.rec.rgb(j) for j in range(i - self.cfg.context_size, i + 1)]
            obs = np.concatenate([_norm_rgb_96(f if f is not None else np.zeros((*depth.shape, 3), np.uint8)) for f in ctx], axis=0)  # (12,96,96)
            gj = min(i + self.cfg.goal_lookahead, self.rec.num_frames - 1)
            goal_img = self.rec.rgb(gj)
            goal_img = _norm_rgb_96(goal_img if goal_img is not None else np.zeros((*depth.shape, 3), np.uint8))
            sample.update({
                "obs_img": torch.from_numpy(obs.astype(np.float32)),
                "goal_img": torch.from_numpy(goal_img.astype(np.float32)),
                "label": torch.from_numpy(label),                                # (8,2)
                "distance": torch.tensor(float(self.cfg.goal_lookahead)),
                "action_mask": torch.tensor(1.0),
            })
        else:
            raise ValueError(f"unknown model {self.cfg.model!r}")
        return sample
