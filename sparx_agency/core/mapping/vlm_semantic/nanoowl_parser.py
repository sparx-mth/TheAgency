import json
from typing import Any

def parse_nanoowl_json_detections(s: str) -> tuple[list[dict[str, Any]], int, int]:
    """
    Returns (detections, W, H)
    detections item: {"bbox":(x1,y1,x2,y2), "label":str, "score":float}
    """
    data = json.loads(s)
    nano = (data.get("nanoowl") or {}).get("result") or {}
    img = nano.get("image") or {}
    W = int(img.get("width", 0))
    H = int(img.get("height", 0))

    dets = nano.get("detections") or []
    out = []
    for d in dets:
        bbox = d.get("bbox", None)
        if not bbox or len(bbox) != 4:
            continue
        out.append({
            "bbox": (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
            "label": str(d.get("label", "")),
            "score": float(d.get("score", 0.0)),
        })
    return out, W, H
