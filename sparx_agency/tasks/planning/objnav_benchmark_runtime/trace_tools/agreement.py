"""Keep a target-kind box only when a second detector puts a box of the same kind on it; other labels pass through unchanged."""
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.evaluate import iou
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_sweep.kinds import DETECTOR_KIND

PARTNER_MIN_CONF = 0.30
AGREE_IOU = 0.5


def agreeing(first, second):
    """``first`` without the target-kind boxes that ``second`` does not confirm (same kind, IoU >= AGREE_IOU, confidence >= PARTNER_MIN_CONF)."""
    partners = [d for d in second if d.cls in DETECTOR_KIND and d.conf >= PARTNER_MIN_CONF]
    kept = []
    for d in first:
        if d.cls not in DETECTOR_KIND or any(DETECTOR_KIND[p.cls] == DETECTOR_KIND[d.cls] and iou(d.xyxy, p.xyxy) >= AGREE_IOU for p in partners):
            kept.append(d)
    return kept
