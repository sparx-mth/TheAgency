"""Run the PUBLISHED FlowNav model on two images -- the authors' exact path.

No TensorRT, no wrappers, no bakes, no low-K. This reproduces the action
inference of FlowNav's own ``deployment/src/navigation/navigate.py``
(``run_navigation_loop``) verbatim: the authors' model construction (the same as
``deployment/src/utils.py::load_model``), the RAW submodule forwards, and
``torchdiffeq.odeint(..., method="euler")`` at ``--k-steps`` (their default 10).
It exists so you can run "the published model, no games" against our integration
and confirm they agree.

Run (flownav_trt env -- has torch + the two forks; PYTHONPATH = repo root):
    python -m sparx_agency.tasks.planning.vlas.flownav.run_reference \
        --ckpt ~/PycharmProjects/flownav/flownav/checkpoints/flownav_weights.pth \
        --flownav-repo ~/PycharmProjects/flownav \
        --obs /tmp/xtend_frames/<a frame>.jpg --goal ~/Downloads/goal_image.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchdiffeq
import yaml
from PIL import Image as PILImage
from torchvision import transforms

# ImageNet normalization, exactly as transform_images (center_crop=False path).
_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _transform(pil_imgs, image_size):
    """Authors' transform_images: resize -> ToTensor -> Normalize -> cat on channels."""
    if not isinstance(pil_imgs, list):
        pil_imgs = [pil_imgs]
    out = [torch.unsqueeze(_TF(im.resize(tuple(image_size))), 0) for im in pil_imgs]
    return torch.cat(out, dim=1)


def _get_action(ndeltas, stats):
    """Authors' get_action: unnormalize [-1,1] deltas -> cumsum -> absolute waypoints."""
    nd = ndeltas.reshape(ndeltas.shape[0], -1, 2).detach().cpu().numpy()
    nd = (nd + 1.0) / 2.0 * (stats["max"] - stats["min"]) + stats["min"]
    return np.cumsum(nd, axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--flownav-repo", required=True)
    ap.add_argument("--obs", required=True, help="current-view RGB image")
    ap.add_argument("--goal", required=True, help="target-view RGB image")
    ap.add_argument("--k-steps", type=int, default=10, help="Euler steps (authors' default 10)")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--waypoint", type=int, default=2)
    args = ap.parse_args()

    repo = Path(args.flownav_repo).expanduser().resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    # The authors' classes (forks must be importable: depth_anything_v2, diffusion_policy).
    from flownav.models.nomad import DenseNetwork, NoMaD
    from flownav.models.nomad_vint import NoMaD_ViNT, replace_bn_with_gn
    from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

    cfg = yaml.safe_load((repo / "flownav" / "config" / "flownav.yaml").read_text())
    data_cfg = yaml.safe_load((repo / "flownav" / "data" / "data_config.yaml").read_text())
    stats = {"min": np.array(data_cfg["action_stats"]["min"], np.float32),
             "max": np.array(data_cfg["action_stats"]["max"], np.float32)}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build EXACTLY like deployment/src/utils.py::load_model.
    vision_encoder = replace_bn_with_gn(NoMaD_ViNT(
        obs_encoding_size=cfg["encoding_size"], context_size=cfg["context_size"],
        mha_num_attention_heads=cfg["mha_num_attention_heads"],
        mha_num_attention_layers=cfg["mha_num_attention_layers"],
        mha_ff_dim_factor=cfg["mha_ff_dim_factor"], depth_cfg=cfg["depth"]))
    noise_pred_net = ConditionalUnet1D(
        input_dim=2, global_cond_dim=cfg["encoding_size"],
        down_dims=cfg["down_dims"], cond_predict_scale=cfg["cond_predict_scale"])
    model = NoMaD(vision_encoder=vision_encoder, noise_pred_net=noise_pred_net,
                  dist_pred_net=DenseNetwork(embedding_dim=cfg["encoding_size"]))
    model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    model = model.to(device).eval()

    image_size = cfg["image_size"]
    ctx = cfg["context_size"]
    obs_pil = PILImage.open(args.obs).convert("RGB")
    goal_pil = PILImage.open(args.goal).convert("RGB")
    # navigate.py keeps a context_size+1 frame buffer; with one frame we repeat it.
    obs_images = _transform([obs_pil] * (ctx + 1), image_size).to(device)
    goal_image = _transform(goal_pil, image_size).to(device)
    mask = torch.zeros(1).long().to(device)            # navigation: goal used

    with torch.no_grad():
        obsgoal_cond = model("vision_encoder", obs_img=obs_images, goal_img=goal_image,
                             input_goal_mask=mask)
        dist = float(model("dist_pred_net", obsgoal_cond=obsgoal_cond).cpu().numpy().reshape(-1)[0])
        obs_cond = obsgoal_cond.repeat(args.num_samples, 1)
        x0 = torch.randn((args.num_samples, cfg["len_traj_pred"], 2), device=device)
        traj = torchdiffeq.odeint(
            lambda t, x: model.forward("noise_pred_net", sample=x, timestep=t, global_cond=obs_cond),
            x0, torch.linspace(0, 1, args.k_steps, device=device),
            atol=1e-4, rtol=1e-4, method="euler")
        actions = _get_action(traj[-1], stats)         # (N, len_traj_pred, 2)

    print("[reference] PUBLISHED FlowNav, K=%d, N=%d (no TRT, authors' odeint)"
          % (args.k_steps, args.num_samples))
    print("[reference] goal-distance: %.3f" % dist)
    print("[reference] chosen waypoint (sample 0, idx %d): %s"
          % (args.waypoint, np.round(actions[0, args.waypoint], 3).tolist()))
    print("[reference] sample 0 trajectory (forward,left):",
          np.round(actions[0], 3).tolist())


if __name__ == "__main__":
    main()
