"""Construct the trained NavDP point-goal policy and dump its numpy head params.

Loads the external ``NavDP_Policy`` with the *verified* checkpoint config
(``temporal_depth=16`` -- the model default is 8, the checkpoint has 16 decoder
layers), loads ``navdp-cross-modal.ckpt`` with ``strict=False`` (the checkpoint
carries unused image/pixel-goal encoders), and -- crucially -- asserts that
every weight feeding the three exported graphs is actually present, so a silent
``strict=False`` drop can never ship a partly random-initialised engine.

``dump_head_params`` writes the small tensors the numpy runtime needs and that
are NOT in any engine: the ``point_encoder`` linear, the 10-row sinusoidal time
table (one per diffusion timestep), and the scheduler's ``alphas_cumprod``.

Runs in the navdp conda env (torch + the external NavDP repo). The repo location
comes from ``--navdp-repo`` / ``NAVDP_REPO`` -- never hardcoded.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Verified architecture of navdp-cross-modal.ckpt (see checkpoint introspection).
IMAGE_SIZE = 224
MEMORY_SIZE = 8
PREDICT_SIZE = 24
TEMPORAL_DEPTH = 16
HEADS = 8
TOKEN_DIM = 384

# Every weight prefix that feeds one of the three exported graphs. If any of
# these is reported "missing" by load_state_dict, the export must abort.
REQUIRED_PREFIXES = (
    "rgbd_encoder.rgb_model", "rgbd_encoder.depth_model", "rgbd_encoder.former_net",
    "rgbd_encoder.former_pe", "rgbd_encoder.former_query", "rgbd_encoder.project_layer",
    "point_encoder", "decoder.layers", "input_embed", "cond_pos_embed",
    "out_pos_embed", "action_head", "critic_head", "layernorm",
)


def resolve_navdp_repo(navdp_repo=None):
    """Resolve and validate the external NavDP repo path (arg > env > fail)."""
    repo = navdp_repo or os.environ.get("NAVDP_REPO")
    if not repo:
        raise ValueError(
            "NavDP repo path unknown: pass --navdp-repo or set NAVDP_REPO "
            "(the directory containing policy_network.py).")
    repo = Path(repo).expanduser().resolve()
    if not (repo / "policy_network.py").exists():
        raise FileNotFoundError("policy_network.py not found under %s" % repo)
    return repo


def build_navdp_policy(ckpt_path, navdp_repo=None, device="cpu"):
    """Build ``NavDP_Policy`` and load the checkpoint, asserting weight presence.

    Args:
        ckpt_path: path to ``navdp-cross-modal.ckpt`` (a raw state_dict).
        navdp_repo: external NavDP repo path (else ``NAVDP_REPO``).
        device: torch device for the constructed model.

    Returns:
        The eval-mode ``NavDP_Policy`` with weights loaded.

    Raises:
        RuntimeError: if any weight feeding an exported graph is missing.
    """
    repo = resolve_navdp_repo(navdp_repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from policy_network import NavDP_Policy

    policy = NavDP_Policy(IMAGE_SIZE, MEMORY_SIZE, PREDICT_SIZE, TEMPORAL_DEPTH,
                          HEADS, TOKEN_DIM, device=device)
    state = torch.load(Path(ckpt_path).expanduser().resolve(), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = policy.load_state_dict(state, strict=False)

    dropped = [k for k in missing if k.startswith(REQUIRED_PREFIXES)]
    if dropped:
        raise RuntimeError(
            "Checkpoint is missing %d weights feeding the exported graphs, e.g. "
            "%s. The engine would contain random weights -- aborting."
            % (len(dropped), dropped[:8]))
    return policy.to(device).eval()


def dump_head_params(policy, out_npz):
    """Write point-encoder weights, the sinusoidal time table, and alphas_cumprod.

    Args:
        policy: a loaded ``NavDP_Policy``.
        out_npz: output ``.npz`` path consumed by ``NavDPTRTPolicy``.
    """
    sd = policy.state_dict()
    if "point_encoder.weight" not in sd:
        raise RuntimeError("point_encoder.weight absent -- not loaded from checkpoint")
    w = sd["point_encoder.weight"].detach().cpu().numpy().astype(np.float32)
    b = sd["point_encoder.bias"].detach().cpu().numpy().astype(np.float32)

    n_steps = int(policy.noise_scheduler.config.num_train_timesteps)
    with torch.no_grad():
        rows = [policy.time_emb(torch.tensor([t], dtype=torch.float32)).cpu().numpy()[0]
                for t in range(n_steps)]
    time_table = np.asarray(rows, dtype=np.float32)                  # (n_steps, 384)
    alphas_cumprod = policy.noise_scheduler.alphas_cumprod.detach().cpu().numpy().astype(np.float32)

    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, point_encoder_weight=w, point_encoder_bias=b,
             time_table=time_table, alphas_cumprod=alphas_cumprod)
    return out_npz
