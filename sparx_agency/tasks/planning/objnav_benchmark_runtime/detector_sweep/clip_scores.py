"""Score every detection of a sweep with CLIP: the probability that the crop is the kind its label names (``clip_<name>.json`` per scene).

Standalone (needs the ``clip`` package, torch, Pillow, OpenCV): it runs in the detector environment. One rule for every
label -- a softmax over one prompt per kind and a few look-alike distractors -- so there is no per-label setting.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

#: the detector's label -> the kind it answers (the same table as kinds.DETECTOR_KIND)
KIND = {"bed": "bed", "sofa": "sofa", "couch": "sofa", "armchair": "armchair", "chair": "chair", "tv": "tv", "television": "tv", "monitor": "tv",
        "toilet": "toilet", "potted plant": "plant", "sink": "sink", "oven": "oven", "refrigerator": "refrigerator", "bathtub": "bathtub",
        "desk": "desk", "dining table": "dining table"}
WORDS = {"bed": "bed", "sofa": "sofa", "armchair": "armchair", "chair": "chair", "tv": "television", "toilet": "toilet", "plant": "potted plant",
         "sink": "sink", "oven": "oven", "refrigerator": "refrigerator", "bathtub": "bathtub", "desk": "desk", "dining table": "dining table"}
DISTRACTORS = ("cabinet", "wall", "door", "window", "table", "shelf", "kitchen counter", "picture frame", "lamp", "curtain", "floor", "pillow")
MARGIN = 0.1
BATCH = 64


def crops_of(image, detections):
    h, w = image.shape[:2]
    out = []
    for d in detections:
        x0, y0, x1, y1 = d["xyxy"]
        mx, my = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
        box = (int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(w, x1 + mx)), int(min(h, y1 + my)))
        out.append(Image.fromarray(cv2.cvtColor(image[box[1]:box[3], box[0]:box[2]], cv2.COLOR_BGR2RGB)))
    return out


class Scorer:
    def __init__(self, weights, device="cpu"):
        import clip

        self.model, self.preprocess = clip.load(str(weights), device=device)
        self.kinds = list(WORDS)
        prompts = ["a photo of a %s" % w for w in list(WORDS.values()) + list(DISTRACTORS)]
        with torch.no_grad():
            text = self.model.encode_text(clip.tokenize(prompts).to(device))
        self.text = text / text.norm(dim=-1, keepdim=True)
        self.device = device

    def kind_probabilities(self, images):
        with torch.no_grad():
            batch = torch.stack([self.preprocess(i) for i in images]).to(self.device)
            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return (100.0 * feats @ self.text.T).softmax(dim=-1).cpu().numpy()


def score_scene(scorer, folder, name, min_conf):
    found = json.loads((folder / ("det_%s.json" % name)).read_text())
    out = {}
    for frame, detections in found.items():
        keep = [d for d in detections if d["cls"] in KIND and d["conf"] >= min_conf]
        if keep:
            probs = scorer.kind_probabilities(crops_of(cv2.imread(str(folder / "frames" / frame)), keep))
            for d, p in zip(keep, probs):
                d["clip"] = round(float(p[scorer.kinds.index(KIND[d["cls"]])]), 4)
        out[frame] = keep
    (folder / ("clip_%s.json" % name)).write_text(json.dumps(out))
    return sum(len(v) for v in out.values())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--min-conf", type=float, default=0.10)
    args = p.parse_args(argv)
    torch.set_num_threads(4)
    scorer = Scorer(args.weights)
    for name in args.names:
        for folder in sorted(d for d in args.sweep.iterdir() if (d / ("det_%s.json" % name)).exists()):
            if not (folder / ("clip_%s.json" % name)).exists():
                print("%s %s: %d detections scored" % (name, folder.name, score_scene(scorer, folder, name, args.min_conf)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
