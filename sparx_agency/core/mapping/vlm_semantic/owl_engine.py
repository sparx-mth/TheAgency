#!/usr/bin/env python3
import argparse, io, json, os, time, hashlib, base64
from typing import List, Optional, Dict
import cv2
import numpy as np
from PIL import Image

from flask import Flask, request, jsonify
from flask_cors import CORS

from nanoowl.tree import Tree
from nanoowl.tree_predictor import TreePredictor
from nanoowl.owl_predictor import OwlPredictor

# ---------------- Config (can be set via CLI) ----------------
MIN_SCORE = 0.35
NMS_IOU   = 0.5
JPEG_QUALITY = 90

# ---------------- Helpers (adapted from your script) ---------
def _iter_detections(dets):
    if hasattr(dets, "detections"):
        dets = dets.detections
    if isinstance(dets, dict):
        return dets.values()
    return dets or []

def _clean_label(name: str) -> str:
    return name.strip().strip('"').strip("'").lower()

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, x2-x1), max(0, y2-y1)
    inter = iw * ih
    ua = (ax2-ax1)*(ay2-ay1)
    ub = (bx2-bx1)*(by2-by1)
    return inter / max(1e-6, ua + ub - inter)

def _nms_per_class(dets, iou_thresh=NMS_IOU):
    out, by = [], {}
    for d in dets:
        by.setdefault(d["label"], []).append(d)
    for cls, items in by.items():
        items = sorted(items, key=lambda x: x["score"], reverse=True)
        keep = []
        for d in items:
            if all(_iou(d["bbox"], k["bbox"]) < iou_thresh for k in keep):
                keep.append(d)
        out.extend(keep)
    return out

def _pack_detections(predictions, tree: Tree, min_score=MIN_SCORE):
    out = []
    for td in _iter_detections(predictions):
        box    = getattr(td, "box", None)
        labels = getattr(td, "labels", None)
        scores = getattr(td, "scores", None)
        parent = getattr(td, "parent_id", None)
        if box is None or labels is None or scores is None:
            continue
        lbl_idx = int(labels[0]) if isinstance(labels, (list, tuple)) else int(labels)
        score   = float(scores[0]) if isinstance(scores, (list, tuple)) else float(scores)
        name = _clean_label(str(tree.labels[lbl_idx]))
        # skip root node
        if name == "image" or parent == -1:
            continue
        if score < min_score:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        out.append({"label": name, "bbox": [x1, y1, x2, y2], "score": score})
    return out

def _draw_from_dets(img_bgr, dets):
    out = img_bgr.copy()
    H, W = out.shape[:2]
    pal = [(0,255,0), (255,0,0), (0,0,255), (0,255,255),
           (255,255,0), (255,0,255), (128,128,0), (0,128,128)]
    def clip(v, lo, hi): return max(lo, min(hi, int(v)))
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = d["bbox"]
        x1, y1 = clip(x1, 0, W-1), clip(y1, 0, H-1)
        x2, y2 = clip(x2, 0, W-1), clip(y2, 0, H-1)
        if x2 <= x1 or y2 <= y1: continue
        color = pal[i % len(pal)]
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        cv2.putText(out, f"{d['label']}:{d['score']:.2f}",
                    (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return out

def _prompts_hash(prompts: List[str]) -> str:
    return hashlib.sha1(("\n".join(prompts)).encode("utf-8")).hexdigest()

def _encode_prompts(predictor: TreePredictor, prompts: List[str]):
    pstr = "[" + ", ".join([json.dumps(p) for p in prompts]) + "]"
    tree = Tree.from_prompt(pstr)
    clip = predictor.encode_clip_text(tree)
    owl  = predictor.encode_owl_text(tree)
    return tree, clip, owl

def _as_pil_from_bgr(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

# ---------------- App / Model init ----------------
app = Flask(__name__)
CORS(app)

predictor: Optional[TreePredictor] = None
enc_cache: Dict[str, tuple[Tree, object, object]] = {}

@app.route("/health", methods=["GET"])
def health():
    return {"ok": True}, 200

@app.route("/infer", methods=["POST"])
def infer():
    """
    Accept either:
      - multipart/form-data with:
          files['image'] = JPEG/PNG
          form['prompts'] = JSON-encoded list of strings
          form['annotate'] = "1" (optional)
      - application/json with:
          {"image_b64": "...", "prompts": ["a ...", ...], "annotate": true}

    Returns JSON:
      { "image": {"width": W, "height": H},
        "prompts": [...],
        "detections": [...],
        "latency_sec": 0.123,
        "annotated_image_b64": "...optional..." }
    """
    try:
        annotate = False
        img_bgr = None
        prompts = None

        if request.content_type and request.content_type.startswith("multipart/"):
            f = request.files.get("image")
            if not f:
                return jsonify({"error": "missing file 'image'"}), 400
            buf = np.frombuffer(f.read(), dtype=np.uint8)
            img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return jsonify({"error": "bad image data"}), 400
            p_raw = request.form.get("prompts", "")
            try:
                prompts = json.loads(p_raw) if p_raw else []
            except Exception:
                return jsonify({"error": "form 'prompts' must be JSON list"}), 400
            annotate = request.form.get("annotate", "") in ("1","true","yes")
        else:
            data = request.get_json(silent=True) or {}
            b64 = (data.get("image_b64") or "").strip()
            if not b64:
                return jsonify({"error": "missing 'image_b64'"}), 400
            try:
                img_bytes = base64.b64decode(b64, validate=True)
                img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                return jsonify({"error": "invalid 'image_b64'"}), 400
            prompts = data.get("prompts") or []
            annotate = bool(data.get("annotate", False))

        # validate prompts
        if not isinstance(prompts, list) or not all(isinstance(s, str) for s in prompts):
            return jsonify({"error": "'prompts' must be a list of strings"}), 400
        if len(prompts) == 0:
            return jsonify({"error": "empty prompts"}), 400

        H, W = img_bgr.shape[:2]
        image_pil = _as_pil_from_bgr(img_bgr)

        ph = _prompts_hash(prompts)
        trip = enc_cache.get(ph)
        if trip is None:
            tree, clip, owl = _encode_prompts(predictor, prompts)
            enc_cache[ph] = (tree, clip, owl)
        else:
            tree, clip, owl = trip

        t0 = time.perf_counter()
        preds = predictor.predict(image_pil, tree=tree, clip_text_encodings=clip, owl_text_encodings=owl)
        latency = time.perf_counter() - t0

        dets = _pack_detections(preds, tree, MIN_SCORE)
        dets = _nms_per_class(dets, iou_thresh=NMS_IOU)

        out = {
            "image": {"width": int(W), "height": int(H)},
            "prompts": prompts,
            "detections": dets,
            "latency_sec": round(latency, 4)
        }

        if annotate:
            drawn = _draw_from_dets(img_bgr, dets)
            ok, enc = cv2.imencode(".jpg", drawn, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok and enc is not None:
                out["annotated_image_b64"] = base64.b64encode(enc.tobytes()).decode("ascii")

        return jsonify(out), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def main():
    global predictor, MIN_SCORE, NMS_IOU, JPEG_QUALITY

    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="/opt/nanoowl/data/owl_image_encoder_patch32.engine")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5060)
    ap.add_argument("--min-score", type=float, default=MIN_SCORE)
    ap.add_argument("--nms-iou", type=float, default=NMS_IOU)
    ap.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    args = ap.parse_args()

    MIN_SCORE = args.min_score
    NMS_IOU   = args.nms_iou
    JPEG_QUALITY = args.jpeg_quality

    print("Loading NanoOWL engines…")
    t0 = time.perf_counter()
    predictor = TreePredictor(owl_predictor=OwlPredictor(image_encoder_engine=args.engine))
    print(f"NanoOWL ready in {time.perf_counter()-t0:.2f}s")

    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()
