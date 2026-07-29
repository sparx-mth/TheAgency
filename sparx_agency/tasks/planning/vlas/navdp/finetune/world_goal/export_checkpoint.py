"""Merge a fine-tune back into a full NavDP checkpoint, so everything else works.

    python -m ...world_goal.export_checkpoint --run ~/navdp_world_goal/run1 \
        --base ~/Downloads/navdp-cross-modal.ckpt \
        --out ~/navdp_world_goal/navdp-world-goal.ckpt

Training saves only the ~44.5 M tensors it can change, which is the right thing
for keeping five checkpoints on a laptop but is not something any other tool in
this repo can load. This writes the merge: the pretrained checkpoint with the
fine-tuned tensors written over it, in exactly the format
``navdp-cross-modal.ckpt`` uses.

The result drops straight into everything already built around NavDP:

* ``serve/navdp_trt_server.py --backend torch --ckpt <this>`` serves it over the
  HTTP contract the FALCON nodes already speak -- so ``navdp_click_node``,
  ``hybrid_planner_node`` and the rest fly the fine-tune with no code change;
* ``trt/export/export_onnx.py`` then ``trt/engine/build_engine.py`` build
  TensorRT engines from it for the real aircraft;
* ``fly_navdp.py`` points the closed-loop comparison at it.

Three things are verified before anything is written, because each has a silent
failure mode that would otherwise surface as "the fine-tune did nothing":
every fine-tuned key must exist in the base checkpoint (a renamed module would
otherwise be dropped by ``strict=False``), something must actually differ, and
the frozen RGB trunk must be untouched.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

RGB_TRUNK_PREFIX = "rgbd_encoder.rgb_model"


def merge(base: Dict[str, torch.Tensor], tuned: Dict[str, torch.Tensor]) -> Dict:
    """Overlay fine-tuned tensors onto a full checkpoint, with the safety checks.

    Args:
        base: The pretrained state dict.
        tuned: The trainable subset saved by training.

    Returns:
        ``(merged_state, report)``.

    Raises:
        KeyError: If a fine-tuned key is absent from the base checkpoint.
        ValueError: If a shape disagrees, if nothing changed, or if the frozen
            RGB trunk moved.
    """
    missing = [key for key in tuned if key not in base]
    if missing:
        raise KeyError(
            f"{len(missing)} fine-tuned tensors are not in the base checkpoint, "
            f"e.g. {missing[:5]} -- the two do not describe the same model")

    merged = {key: value.clone() for key, value in base.items()}
    changed, total_delta, max_delta = 0, 0.0, 0.0
    for key, value in tuned.items():
        if tuple(value.shape) != tuple(base[key].shape):
            raise ValueError(f"{key}: fine-tuned shape {tuple(value.shape)} != "
                             f"base {tuple(base[key].shape)}")
        if key.startswith(RGB_TRUNK_PREFIX) and not torch.equal(
                value.to(base[key].dtype), base[key]):
            raise ValueError(
                f"{key} is in the frozen RGB trunk but differs from the base "
                f"checkpoint -- the freeze policy was not applied")
        delta = (value.to(base[key].dtype).float() - base[key].float())
        norm = float(delta.abs().max()) if delta.numel() else 0.0
        if norm > 0:
            changed += 1
            total_delta += float(delta.pow(2).sum())
            max_delta = max(max_delta, norm)
        merged[key] = value.to(base[key].dtype)

    if changed == 0:
        raise ValueError("no tensor differs from the pretrained checkpoint -- "
                         "the exported model would be the baseline")
    return merged, {"tensors_in_base": len(base), "tensors_overlaid": len(tuned),
                    "tensors_changed": changed, "l2_delta": total_delta ** 0.5,
                    "max_abs_delta": max_delta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True, help="training output directory")
    parser.add_argument("--checkpoint", default="best.pth")
    parser.add_argument("--weights", default="ema", choices=("ema", "model"))
    parser.add_argument("--base", default="~/Downloads/navdp-cross-modal.ckpt")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run = Path(args.run).expanduser()
    payload = torch.load(run / args.checkpoint, map_location="cpu", weights_only=False)
    base = torch.load(Path(args.base).expanduser(), map_location="cpu",
                      weights_only=False)
    if isinstance(base, dict) and "state_dict" in base:
        base = base["state_dict"]

    merged, report = merge(base, payload[args.weights])
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, out)

    print(f"[export] {args.checkpoint}:{args.weights} from step {payload.get('step')}")
    for key, value in report.items():
        print(f"[export]   {key}: {value:.6g}" if isinstance(value, float)
              else f"[export]   {key}: {value}")
    print(f"[export] wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")
    print("[export] serve it with:\n"
          f"  python -m sparx_agency.tasks.planning.vlas.navdp.serve.navdp_trt_server "
          f"--backend torch --ckpt {out} --port 8888")


if __name__ == "__main__":
    main()
