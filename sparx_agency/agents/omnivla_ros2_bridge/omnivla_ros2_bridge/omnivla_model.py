#!/usr/bin/env python3
"""
OmniVLA Model Wrapper — Pure Python, no ROS dependency.

Loads the OmniVLA model and exposes a single `predict()` method that takes
a camera image (+ optional goals) and returns (linear_vel, angular_vel).

Supported modality combinations (non-satellite):
  ID 4 — pose only
  ID 5 — pose + image
  ID 6 — image only
  ID 7 — language only
  ID 8 — language + pose

Note: language + image is NOT a trained modality in OmniVLA.
      If both are provided without pose, only language is used.
"""

import os
import math
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.nn.utils.rnn import pad_sequence

# ── OmniVLA imports (require OmniVLA repo in PYTHONPATH) ──────────────
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.models.projectors import ProprioProjector
from prismatic.models.action_heads import L1RegressionActionHead_idcat
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction_MMNv1
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, POSE_DIM
from transformers import AutoConfig, AutoProcessor, AutoModelForVision2Seq, AutoImageProcessor


# ── Helpers ───────────────────────────────────────────────────────────
def _clip_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _load_ckpt(name: str, path: str, step: int, device="cpu") -> dict:
    ckpt = os.path.join(path, f"{name}--{step}_checkpoint.pt")
    if not os.path.exists(ckpt) and name == "pose_projector":
        ckpt = os.path.join(path, f"proprio_projector--{step}_checkpoint.pt")
    sd = torch.load(ckpt, map_location=device)
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


# ── Modality ID lookup ────────────────────────────────────────────────
_MODALITY_TABLE = {
    # (satellite, language, pose, image) → (id, name)
    (True,  False, False, False): (0, "satellite"),
    (True,  False, True,  False): (1, "satellite+pose"),
    (True,  False, False, True):  (2, "satellite+image"),
    (True,  False, True,  True):  (3, "satellite+pose+image"),
    (False, False, True,  False): (4, "pose"),
    (False, False, True,  True):  (5, "pose+image"),
    (False, False, False, True):  (6, "image"),
    (False, True,  False, False): (7, "language"),
    (False, True,  True,  False): (8, "language+pose"),
}

# Combinations that fall back because they're not in the table
_FALLBACK_MSG = (
    "language+image is NOT a trained OmniVLA modality. "
    "Falling back to language-only. Add a goal_pose to use language+pose."
)


class OmniVLAModel:
    """Load OmniVLA once, call predict() every tick."""

    def __init__(
        self,
        vla_path: str = "./omnivla-original",
        resume_step: int = 120_000,
        device: str = "cuda:0",
        max_linear: float = 0.3,
        max_angular: float = 0.3,
        waypoint_index: int = 4,
    ):
        self.device = torch.device(device)
        self.max_v = max_linear
        self.max_w = max_angular
        self.wp_idx = waypoint_index
        self.wp_spacing = 0.1

        # ── Register custom HF classes ────────────────────────────────
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction_MMNv1)

        # ── Load processor & VLA backbone ─────────────────────────────
        self.processor = AutoProcessor.from_pretrained(vla_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.img_transform = self.processor.image_processor.apply_transform
        self.action_tokenizer = ActionTokenizer(self.tokenizer)

        self.vla = AutoModelForVision2Seq.from_pretrained(
            vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).to(self.device)
        self.vla.vision_backbone.set_num_images_in_input(2)
        self.vla.to(dtype=torch.bfloat16, device=self.device)
        self.vla.eval()

        # ── Pose projector ────────────────────────────────────────────
        self.pose_proj = ProprioProjector(
            llm_dim=self.vla.llm_dim, proprio_dim=POSE_DIM
        )
        self.pose_proj.load_state_dict(_load_ckpt("pose_projector", vla_path, resume_step))
        self.pose_proj = self.pose_proj.to(self.device).eval()

        # ── Action head ───────────────────────────────────────────────
        self.action_head = L1RegressionActionHead_idcat(
            input_dim=self.vla.llm_dim,
            hidden_dim=self.vla.llm_dim,
            action_dim=ACTION_DIM,
        )
        self.action_head.load_state_dict(_load_ckpt("action_head", vla_path, resume_step))
        self.action_head = self.action_head.to(torch.bfloat16).to(self.device).eval()

        # ── Patch count (current_img + goal_img + goal_pose token) ────
        npp = self.vla.vision_backbone.get_num_patches()
        self.num_patches = npp * 2 + 1

        # ── Default black placeholder for goal image ──────────────────
        self._black_goal = Image.new("RGB", (224, 224), (0, 0, 0))

        print(f"[OmniVLA] Loaded from {vla_path}  step={resume_step}  device={self.device}")

    # ══════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════════
    def predict(
        self,
        current_image: Image.Image,
        instruction: Optional[str] = None,
        goal_image: Optional[Image.Image] = None,
        goal_pose: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, np.ndarray, str]:
        """
        Parameters
        ----------
        current_image : PIL.Image  — current RGB camera view
        instruction   : str | None — e.g. "move toward the blue door"
        goal_image    : PIL.Image | None — egocentric goal image
        goal_pose     : np.ndarray shape (4,) [rel_y, -rel_x, cos_h, sin_h] | None

        Returns
        -------
        (linear_vel, angular_vel, raw_waypoints, modality_name)
        """
        has_lang  = instruction is not None and instruction.strip() != ""
        has_img   = goal_image is not None
        has_pose  = goal_pose is not None

        lang_text   = instruction if has_lang else "xxxx"
        g_img       = goal_image if has_img else self._black_goal
        g_pose      = goal_pose if has_pose else np.zeros(4, dtype=np.float32)
        mod_id, mod_name = self._modality_id(False, has_lang, has_pose, has_img)

        batch  = self._prepare_batch(current_image, lang_text, g_img, g_pose)
        wps    = self._forward(batch, mod_id)                # (1, N, 4)
        lin, ang = self._pd_controller(wps[0])
        return lin, ang, wps[0], mod_name

    # ══════════════════════════════════════════════════════════════════
    #  INTERNALS
    # ══════════════════════════════════════════════════════════════════
    @staticmethod
    def _modality_id(sat, lang, pose, img):
        key = (sat, lang, pose, img)
        entry = _MODALITY_TABLE.get(key)
        if entry:
            idx, name = entry
            return torch.tensor([idx], dtype=torch.float32), name

        # Fallback: unsupported combination → degrade gracefully
        # language+image (no pose) → language only
        if lang and img and not pose:
            print(f"[OmniVLA] WARNING: {_FALLBACK_MSG}")
            return torch.tensor([7], dtype=torch.float32), "language (fallback)"
        # everything else → language only as safe default
        return torch.tensor([7], dtype=torch.float32), "language (fallback)"

    # ── batch construction (mirrors run_omnivla.py logic) ─────────────
    def _prepare_batch(self, cur_img, lang, goal_img, goal_pose):
        IGNORE = -100
        dummy_actions = np.random.rand(8, 4).astype(np.float32)

        cur_act_str = self.action_tokenizer(dummy_actions[0])
        fut_act_str = "".join(self.action_tokenizer(a) for a in dummy_actions[1:])
        chunk_str = cur_act_str + fut_act_str
        chunk_len = len(chunk_str)

        human_msg = ("No language instruction" if lang == "xxxx"
                     else f"What action should the robot take to {lang}?")
        conv = [{"from": "human", "value": human_msg},
                {"from": "gpt",   "value": chunk_str}]

        pb = PurePromptBuilder("openvla")
        for t in conv:
            pb.add_turn(t["from"], t["value"])

        ids = torch.tensor(self.tokenizer(pb.get_prompt(), add_special_tokens=True).input_ids)
        labels = ids.clone()
        labels[:-(chunk_len + 1)] = IGNORE

        pv_cur  = self.img_transform(cur_img)
        pv_goal = self.img_transform(goal_img)

        return {
            "input_ids":      ids.unsqueeze(0),
            "labels":         labels.unsqueeze(0),
            "attention_mask": ids.unsqueeze(0).ne(self.tokenizer.pad_token_id),
            "pixel_values":   torch.cat([pv_cur.unsqueeze(0), pv_goal.unsqueeze(0)], dim=1),
            "goal_pose":      torch.from_numpy(goal_pose.astype(np.float32)).unsqueeze(0),
        }

    # ── model forward pass ────────────────────────────────────────────
    def _forward(self, batch, mod_id) -> np.ndarray:
        dev = self.device
        bf  = torch.bfloat16

        with torch.no_grad(), torch.autocast("cuda", dtype=bf):
            out = self.vla(
                input_ids=batch["input_ids"].to(dev),
                attention_mask=batch["attention_mask"].to(dev),
                pixel_values=batch["pixel_values"].to(bf).to(dev),
                modality_id=mod_id.to(bf).to(dev),
                labels=batch["labels"].to(dev),
                output_hidden_states=True,
                proprio=batch["goal_pose"].to(bf).to(dev),
                proprio_projector=self.pose_proj,
            )

        gt_ids = batch["labels"][:, 1:].to(dev)
        mask_c = get_current_action_mask(gt_ids)
        mask_n = get_next_actions_mask(gt_ids)

        hs = out.hidden_states[-1][:, self.num_patches:-1]
        B  = batch["input_ids"].shape[0]
        act_hs = (
            hs[mask_c | mask_n]
            .reshape(B, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(bf)
        )
        with torch.no_grad():
            pred = self.action_head.predict_action(act_hs, mod_id.to(bf).to(dev))

        return pred.float().cpu().numpy()

    # ── PD controller: waypoints → (linear, angular) ─────────────────
    def _pd_controller(self, waypoints: np.ndarray) -> Tuple[float, float]:
        wp = waypoints[self.wp_idx].copy()
        wp[:2] *= self.wp_spacing
        dx, dy, hx, hy = wp

        EPS, DT = 1e-8, 1.0 / 3.0
        if abs(dx) < EPS and abs(dy) < EPS:
            v = 0.0
            w = _clip_angle(math.atan2(hy, hx)) / DT
        elif abs(dx) < EPS:
            v = 0.0
            w = float(np.sign(dy)) * math.pi / (2 * DT)
        else:
            v = dx / DT
            w = math.atan(dy / dx) / DT

        v = float(np.clip(v, 0, 0.5))
        w = float(np.clip(w, -1.0, 1.0))
        return self._limit(v, w)

    def _limit(self, v: float, w: float) -> Tuple[float, float]:
        mv, mw = self.max_v, self.max_w
        if abs(v) <= mv:
            if abs(w) <= mw:
                return v, w
            rd = v / w
            return mw * np.sign(v) * abs(rd), mw * np.sign(w)
        if abs(w) <= 0.001:
            return mv * np.sign(v), 0.0
        rd = v / w
        if abs(rd) >= mv / mw:
            return mv * np.sign(v), mv * np.sign(w) / abs(rd)
        return mw * np.sign(v) * abs(rd), mw * np.sign(w)