#!/usr/bin/env python3
"""
build_scans.py
---------------
Reads a folder of images and pose JSONs, runs BBOX detection, and produces
a consolidated "scans" structure grouped by actual yaw angle in radians.

Input layout:
  /path/to/images/
    x0000y0200z1500yaw4712389.jpg
    x0000y0200z1500yaw4712389.json  # {"pose": {"x": ..., "y": ..., "z": ..., "yaw": 4.712389}, "image": "x0000y0200z1500yaw4712389.jpg"}
    ...

Output:
  scans.json   # JSON with [{"yaw": 4.712389, "bboxes":[...]}, ...]
  scans.py     # Python file with variable `scans = [...]`
  images_annotated/  # (optional) annotated copies

Dependencies:
  For YOLO (fast, fixed labels):   pip install ultralytics opencv-python
  For OWLv2 (open-vocab labels):   pip install transformers pillow torch torchvision opencv-python

"""

from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ======================= CONFIG =======================
IMAGES_DIR = Path(r"/home/nadavc/PycharmProjects/TheAgency_workspace/src/room_mapping/images")
ENGINE = "owlv2"  # "yolo" or "owlv2"
THRESH = 0.45
# When using OWLv2 (open-vocab), set your desired labels here:
OPEN_VOCAB_LABELS = ["desk", "cabinet", "tv", "table", "couch", "plant", "bed",
                     "suitcase", "table", "socket", "refrigerator", "bottle",
                     "mouse", "weapon", "chair", "keyboard", "computer", "box", "Cardboard box", "gun", "Plastic chair"]
# Save annotated images with rectangles/labels
SAVE_ANNOTATED = True
ANNOTATED_DIR = IMAGES_DIR / "images_annotated"
# ======================================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Detection:
    klass: str
    bbox: List[int]  # [x1,y1,x2,y2]
    confidence: float


def list_pose_jsons(images_dir: Path) -> List[Path]:
    return sorted([p for p in images_dir.iterdir() if p.suffix.lower() == ".json"])


# ---------- Detection backend (prefers bboxify.py if available) ----------
def detect_with_bboxify(image_paths: List[Path], engine: str, labels: Optional[List[str]], threshold: float) -> Dict[
    Path, List[Detection]]:
    """Try to import bboxify.detect_bboxes; if present, use it. Otherwise fall back to local routines."""
    try:
        from bboxify import detect_bboxes  # type: ignore
        results = detect_bboxes(
            source=str(image_paths[0].parent) if len(image_paths) > 1 else str(image_paths[0]),
            labels=labels,
            engine=engine,
            threshold=threshold,
            save_dir=str(ANNOTATED_DIR) if SAVE_ANNOTATED else None,
        )
        # results is list of dicts {"image": "...", "detections":[...]}
        # Build map for the requested images only
        res_map: Dict[Path, List[Detection]] = {}
        wanted = {p.resolve() for p in image_paths}
        for item in results:
            ip = Path(item["image"]).resolve()
            if ip in wanted:
                dets = [Detection(klass=d["class"], bbox=[int(x) for x in d["bbox"]], confidence=float(d["confidence"]))
                        for d in item["detections"]]
                res_map[ip] = dets
        # ensure every requested image has an entry
        for p in image_paths:
            res_map.setdefault(p.resolve(), [])
        return res_map
    except Exception:
        # fall back
        return detect_fallback(image_paths, engine, labels, threshold)


def detect_fallback(image_paths: List[Path], engine: str, labels: Optional[List[str]], threshold: float) -> Dict[
    Path, List[Detection]]:
    if engine == "yolo":
        try:
            from ultralytics import YOLO
            import cv2
        except Exception as e:
            raise RuntimeError("YOLO fallback requires: pip install ultralytics opencv-python") from e
        model = YOLO("yolov8n.pt")
        out: Dict[Path, List[Detection]] = {}
        for ip in image_paths:
            results = model(str(ip), conf=threshold)
            r = results[0]
            names = r.names
            dets: List[Detection] = []
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    cls_id = int(b.cls.item())
                    conf = float(b.conf.item())
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                    dets.append(
                        Detection(klass=names.get(cls_id, str(cls_id)), bbox=[int(x1), int(y1), int(x2), int(y2)],
                                  confidence=conf))
            out[ip.resolve()] = dets
            if SAVE_ANNOTATED:
                img = cv2.imread(str(ip))
                if img is not None:
                    for d in dets:
                        x1, y1, x2, y2 = d.bbox
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{d.klass} {d.confidence:.2f}"
                        (w, h), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        y1_disp = max(0, y1 - h - base)
                        cv2.rectangle(img, (x1, y1_disp), (x1 + w, y1_disp + h + base), (0, 255, 0), -1)
                        cv2.putText(img, label, (x1, y1_disp + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                                    cv2.LINE_AA)
                    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(ANNOTATED_DIR / ip.name), img)
        return out
    else:
        # OWLv2
        try:
            from PIL import Image
            from transformers import pipeline
            import cv2
            import numpy as np
        except Exception as e:
            raise RuntimeError(
                "OWLv2 fallback requires: pip install transformers pillow torch torchvision opencv-python") from e
        if not labels:
            raise ValueError("OWLv2 requires OPEN_VOCAB_LABELS to be set.")
        det = pipeline("zero-shot-object-detection", model="google/owlv2-base-patch16-finetuned")
        out: Dict[Path, List[Detection]] = {}
        for ip in image_paths:
            im = Image.open(ip).convert("RGB")
            preds = det(im, candidate_labels=labels)
            dets: List[Detection] = []
            for p in preds:
                score = float(p.get("score", 0.0))
                if score < threshold:
                    continue
                box = p["box"]
                x1, y1, x2, y2 = int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"])
                dets.append(Detection(klass=str(p["label"]), bbox=[x1, y1, x2, y2], confidence=score))
            out[ip.resolve()] = dets
            if SAVE_ANNOTATED:
                arr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
                for d in dets:
                    x1, y1, x2, y2 = d.bbox
                    cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{d.klass} {d.confidence:.2f}"
                    (w, h), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    y1_disp = max(0, y1 - h - base)
                    cv2.rectangle(arr, (x1, y1_disp), (x1 + w, y1_disp + h + base), (0, 255, 0), -1)
                    cv2.putText(arr, label, (x1, y1_disp + h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(ANNOTATED_DIR / ip.name), arr)
        return out


# ------------------------- Main logic -------------------------
def main():
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images dir not found: {IMAGES_DIR}")

    pose_files = list_pose_jsons(IMAGES_DIR)
    if not pose_files:
        raise RuntimeError(f"No pose JSON files found in {IMAGES_DIR}")

    # Map: image_path -> (yaw in radians, pose data)
    image_paths: List[Path] = []
    image_data_map: Dict[Path, Tuple[float, Dict]] = {}

    for pf in pose_files:
        try:
            data = json.loads(pf.read_text())
        except Exception as e:
            print(f"WARNING: cannot read {pf}: {e}")
            continue

        pose = data.get("pose", {})
        yaw = float(pose.get("yaw", 0.0))

        image_name = data.get("image") or (pf.stem + ".jpg")
        ip = (IMAGES_DIR / image_name).resolve()
        if not ip.exists():
            # try other image suffixes
            found = None
            for ext in IMG_EXTS:
                cand = IMAGES_DIR / (pf.stem + ext)
                if cand.exists():
                    found = cand.resolve()
                    break
            if found is None:
                print(f"WARNING: image for {pf.name} not found: {image_name}")
                continue
            ip = found

        image_data_map[ip] = (yaw, pose)
        image_paths.append(ip)

    # Run detection (batch-friendly; implementation decides)
    det_map = detect_with_bboxify(image_paths, ENGINE, OPEN_VOCAB_LABELS if ENGINE == "owlv2" else None, THRESH)

    # Create scans with actual yaw values
    scans = []
    for ip in sorted(image_paths):
        if ip in det_map and ip in image_data_map:
            yaw, pose = image_data_map[ip]
            detections = det_map[ip]

            if detections:  # Only include if there are detections
                bboxes = []
                for d in detections:
                    bboxes.append({
                        "class": d.klass,
                        "bbox": [int(v) for v in d.bbox],
                        "confidence": round(float(d.confidence), 4)
                    })

                scan_entry = {
                    "yaw": round(yaw, 6),  # Keep precision but round to avoid floating point issues
                    "pose": {
                        "x": round(pose.get("x", 0.0), 3),
                        "y": round(pose.get("y", 0.0), 3),
                        "z": round(pose.get("z", 0.0), 3),
                        "yaw": round(yaw, 6)
                    },
                    "image": ip.name,
                    "bboxes": bboxes
                }
                scans.append(scan_entry)

    # Sort by yaw for easier navigation
    scans.sort(key=lambda s: s["yaw"])

    # Save outputs
    out_json = IMAGES_DIR / "scans.json"
    out_py = IMAGES_DIR / "scans.py"
    out_json.write_text(json.dumps(scans, ensure_ascii=False, indent=2))
    out_py.write_text("scans = " + json.dumps(scans, ensure_ascii=False, indent=2) + "\n")

    # Print summary
    print("Done.")
    print(f"Total scans with detections: {len(scans)}")
    if scans:
        print(f"Yaw range: {scans[0]['yaw']:.3f} to {scans[-1]['yaw']:.3f} radians")
        print(f"  ({math.degrees(scans[0]['yaw']):.1f}° to {math.degrees(scans[-1]['yaw']):.1f}°)")

    total_detections = sum(len(s["bboxes"]) for s in scans)
    print(f"Total detections: {total_detections}")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_py}")
    if SAVE_ANNOTATED:
        print(f"Annotated images: {ANNOTATED_DIR}")


if __name__ == "__main__":
    main()