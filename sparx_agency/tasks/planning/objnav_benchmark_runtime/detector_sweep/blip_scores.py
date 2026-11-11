"""Score every detection of a sweep with BLIP-2's image-text matching head: the probability that the crop matches "a photo of a <kind>".

Standalone (torch with CUDA, transformers, Pillow, OpenCV). One rule for every label, one prompt per detection, no per-label setting.
"""
import argparse
import json
from pathlib import Path

import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForImageTextRetrieval

KIND = {"bed": "bed", "sofa": "sofa", "couch": "sofa", "armchair": "armchair", "chair": "chair", "tv": "tv", "television": "tv", "monitor": "tv",
        "toilet": "toilet", "potted plant": "plant", "sink": "sink", "oven": "oven", "refrigerator": "refrigerator", "bathtub": "bathtub",
        "desk": "desk", "dining table": "dining table"}
WORDS = {"bed": "bed", "sofa": "sofa", "armchair": "armchair", "chair": "chair", "tv": "television", "toilet": "toilet", "plant": "potted plant",
         "sink": "sink", "oven": "oven", "refrigerator": "refrigerator", "bathtub": "bathtub", "desk": "desk", "dining table": "dining table"}
MARGIN = 0.1
BATCH = 16


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
    def __init__(self, path):
        self.processor = AutoProcessor.from_pretrained(str(path))
        self.model = Blip2ForImageTextRetrieval.from_pretrained(str(path), torch_dtype=torch.float16).to("cuda").eval()

    def match_probabilities(self, images, kinds):
        out = []
        for i in range(0, len(images), BATCH):
            texts = ["a photo of a %s" % WORDS[k] for k in kinds[i:i + BATCH]]
            inputs = self.processor(images=images[i:i + BATCH], text=texts, return_tensors="pt", padding=True).to("cuda", torch.float16)
            with torch.no_grad():
                logits = self.model(**inputs, use_image_text_matching_head=True).logits_per_image
            out += torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist()
        return out


def score_scene(scorer, folder, name, min_conf):
    found = json.loads((folder / ("det_%s.json" % name)).read_text())
    out = {}
    for frame, detections in found.items():
        keep = [d for d in detections if d["cls"] in KIND and d["conf"] >= min_conf]
        if keep:
            probs = scorer.match_probabilities(crops_of(cv2.imread(str(folder / "frames" / frame)), keep), [KIND[d["cls"]] for d in keep])
            for d, p in zip(keep, probs):
                d["blip2"] = round(p, 4)
        out[frame] = keep
    (folder / ("blip2_%s.json" % name)).write_text(json.dumps(out))
    return sum(len(v) for v in out.values())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--runs", nargs="+", required=True, help="name:min_conf, e.g. yolo_world:0.10 closed_open:0.30")
    args = p.parse_args(argv)
    scorer = Scorer(args.model)
    for run in args.runs:
        name, min_conf = run.split(":")
        for folder in sorted(d for d in args.sweep.iterdir() if (d / ("det_%s.json" % name)).exists()):
            if not (folder / ("blip2_%s.json" % name)).exists():
                print("%s %s: %d detections scored" % (name, folder.name, score_scene(scorer, folder, name, float(min_conf))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
