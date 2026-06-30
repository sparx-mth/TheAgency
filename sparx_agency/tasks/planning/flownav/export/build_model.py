"""Construct the trained FlowNav model and dump its numpy head params.

Rebuilds FlowNav's ``NoMaD`` exactly as the reference ``deployment/src/utils.py``
``load_model`` does (``NoMaD_ViNT`` vision encoder + ``ConditionalUnet1D`` velocity
field + ``DenseNetwork`` distance head), loads the checkpoint with
``strict=False`` (the released ``flownav_weights.pth`` is a raw ``state_dict``),
and -- crucially -- asserts that every weight feeding the three exported graphs is
present, so a silent ``strict=False`` drop can never ship a partly random engine.

``dump_head_params`` writes the only numbers the numpy runtime needs that are NOT
inside an engine: the per-dim action ``min``/``max`` (from FlowNav's
``data_config.yaml``) used to de-normalize the action deltas.

Runs in the FlowNav build env (torch + the FlowNav repo + the two debOliveira
forks: ``depth_anything_v2``, ``diffusion_policy``). The repo location comes from
``--flownav-repo`` / ``FLOWNAV_REPO`` -- never hardcoded.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Every weight prefix that feeds one of the three exported graphs. If any is
# reported "missing" by load_state_dict, the export must abort.
REQUIRED_PREFIXES = (
    "vision_encoder.obs_encoder", "vision_encoder.goal_encoder",
    "vision_encoder.depth_encoder", "vision_encoder.sa_encoder",
    "noise_pred_net", "dist_pred_net",
)


def resolve_flownav_repo(flownav_repo=None):
    """Resolve and validate the external FlowNav repo path (arg > env > fail)."""
    repo = flownav_repo or os.environ.get("FLOWNAV_REPO")
    if not repo:
        raise ValueError(
            "FlowNav repo path unknown: pass --flownav-repo or set FLOWNAV_REPO "
            "(the directory containing the 'flownav' package).")
    repo = Path(repo).expanduser().resolve()
    if not (repo / "flownav" / "models" / "nomad.py").exists():
        raise FileNotFoundError("flownav/models/nomad.py not found under %s" % repo)
    return repo


def load_config(flownav_repo, config_path=None):
    """Load the FlowNav model config YAML (default: flownav/config/flownav.yaml)."""
    cfg_path = Path(config_path) if config_path else \
        Path(flownav_repo) / "flownav" / "config" / "flownav.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def build_flownav_model(ckpt_path, flownav_repo=None, config_path=None, device="cpu"):
    """Build FlowNav's ``NoMaD`` and load the checkpoint, asserting weight presence.

    Args:
        ckpt_path: path to ``flownav_weights.pth`` (a raw state_dict).
        flownav_repo: external FlowNav repo path (else ``FLOWNAV_REPO``).
        config_path: model config YAML (else ``<repo>/flownav/config/flownav.yaml``).
        device: torch device for the constructed model.

    Returns:
        The eval-mode ``NoMaD`` model with weights loaded.

    Raises:
        RuntimeError: if any weight feeding an exported graph is missing.
    """
    repo = resolve_flownav_repo(flownav_repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from flownav.models.nomad import DenseNetwork, NoMaD
    from flownav.models.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

    cfg = load_config(repo, config_path)
    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=cfg["encoding_size"],
        context_size=cfg["context_size"],
        mha_num_attention_heads=cfg["mha_num_attention_heads"],
        mha_num_attention_layers=cfg["mha_num_attention_layers"],
        mha_ff_dim_factor=cfg["mha_ff_dim_factor"],
        depth_cfg=cfg["depth"],
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=cfg["encoding_size"],
        down_dims=cfg["down_dims"],
        cond_predict_scale=cfg["cond_predict_scale"],
    )
    dist_pred_net = DenseNetwork(embedding_dim=cfg["encoding_size"])
    model = NoMaD(vision_encoder=vision_encoder, noise_pred_net=noise_pred_net,
                  dist_pred_net=dist_pred_net)

    state = torch.load(Path(ckpt_path).expanduser().resolve(), map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)

    dropped = [k for k in missing if k.startswith(REQUIRED_PREFIXES)]
    if dropped:
        raise RuntimeError(
            "Checkpoint is missing %d weights feeding the exported graphs, e.g. "
            "%s. The engine would contain random weights -- aborting. (If the "
            "frozen DepthAnythingV2 depth encoder is absent from the checkpoint, "
            "load depth_anything_v2_vits.pth into vision_encoder.depth_encoder "
            "first.)" % (len(dropped), dropped[:8]))
    return model.to(device).eval()


def load_action_stats(flownav_repo=None):
    """Read the per-dim action ``(min, max)`` from FlowNav's ``data_config.yaml``.

    Args:
        flownav_repo: FlowNav repo path (else ``FLOWNAV_REPO``).

    Returns:
        ``(action_min, action_max)`` as float32 arrays, used to de-normalize the
        velocity-field output (shared by the TRT and the eager-torch runtimes).
    """
    repo = resolve_flownav_repo(flownav_repo)
    with open(repo / "flownav" / "data" / "data_config.yaml", "r") as f:
        stats = yaml.safe_load(f)["action_stats"]
    return (np.asarray(stats["min"], dtype=np.float32),
            np.asarray(stats["max"], dtype=np.float32))


def dump_head_params(flownav_repo, out_npz):
    """Write the action min/max used to de-normalize the velocity-field output.

    Args:
        flownav_repo: FlowNav repo path (reads ``flownav/data/data_config.yaml``).
        out_npz: output ``.npz`` path consumed by ``FlowNavTRTPolicy``.

    Returns:
        ``out_npz`` as a :class:`pathlib.Path`.
    """
    action_min, action_max = load_action_stats(flownav_repo)
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, action_min=action_min, action_max=action_max)
    return out_npz
