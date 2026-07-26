"""Tests for the AprilTag per-tag quality diagnostic + CSV log."""
import csv
import os
import tempfile

import numpy as np

from sparx_agency.tasks.localization.apriltag_quality_log import (
    AprilTagQualityLog,
    default_apriltag_log_path,
)
from sparx_agency.tasks.localization.common.apriltag_frame_diag import (
    build_frame_diag,
)
from sparx_agency.tasks.localization.common.apriltag_pnp import (
    CameraPoseResult,
    PerTagStat,
)


class _Det:
    """Minimal stand-in for a pupil_apriltags detection / provider RawDet."""

    def __init__(self, tag_id, cx, cy, size_px, margin):
        self.tag_id = tag_id
        self.decision_margin = margin
        half = size_px / 2.0
        self.center = (cx, cy)
        self.corners = np.array([[cx - half, cy - half], [cx + half, cy - half],
                                 [cx + half, cy + half], [cx - half, cy + half]],
                                dtype=float)


def _result(used_ids, per_tag):
    return CameraPoseResult(
        world_T_cam=np.eye(4), used_tag_ids=list(used_ids), n_tags=len(used_ids),
        reproj_rms_px=1.2, ambiguity=0.1, geometry=0.7, per_tag=tuple(per_tag))


def test_frame_diag_flags_detected_but_unused_and_off_map_tags():
    dets = [_Det(3, 250, 150, 80, 40.0),   # good, used
            _Det(8, 60, 150, 30, 12.0),    # in map, detected, NOT used (outlier)
            _Det(99, 400, 150, 40, 20.0)]  # not in the map at all
    result = _result([3], [PerTagStat(3, 0.9, 80.0, 1.4)])
    diag = build_frame_diag(dets, result, mapped_ids={3, 8},
                            confidence=0.7, pos_std_m=0.03,
                            source="apriltag", stamp_sec=123.0)
    assert diag.n_detected == 3 and diag.n_used == 1
    by_id = {t.tag_id: t for t in diag.tags}
    assert by_id[3].used and by_id[3].in_map and by_id[3].reproj_rms_px == 0.9
    assert by_id[8].in_map and not by_id[8].used and by_id[8].reproj_rms_px is None
    assert not by_id[99].in_map            # a tag on the wall the map does not know
    assert by_id[3].apparent_px == 80.0    # size carried through from corners


def test_blind_frame_has_no_tags_but_still_a_diag():
    diag = build_frame_diag([], None, mapped_ids={3}, confidence=0.0,
                            pos_std_m=1.0, source="blind", stamp_sec=1.0)
    assert diag.n_detected == 0 and diag.n_used == 0 and diag.tags == ()


def test_log_writes_one_row_per_tag_and_a_blank_row_when_blind():
    path = os.path.join(tempfile.mkdtemp(), "apriltag.csv")
    log = AprilTagQualityLog(path)
    dets = [_Det(3, 250, 150, 80, 40.0), _Det(8, 60, 150, 30, 12.0)]
    result = _result([3, 8], [PerTagStat(3, 0.9, 80.0, 1.4),
                              PerTagStat(8, 7.5, 30.0, 3.1)])
    n1 = log.write(build_frame_diag(dets, result, {3, 8}, 0.7, 0.03,
                                    "apriltag", 1.0))
    n2 = log.write(build_frame_diag([], None, {3, 8}, 0.0, 1.0, "blind", 2.0))
    log.close()
    assert n1 == 2 and n2 == 1             # two tags, then one blank-tag row
    rows = list(csv.DictReader(open(path)))
    assert len(rows) == 3
    tag8 = [r for r in rows if r["tag_id"] == "8"][0]
    assert tag8["tag_reproj_rms_px"] == "7.500" and tag8["used"] == "1"
    blind = rows[-1]
    assert blind["source"] == "blind" and blind["tag_id"] == ""


def test_default_path_honours_the_log_dir_env(monkeypatch=None):
    os.environ["FALCON_LOG_DIR"] = "/tmp/falcon"
    try:
        p = default_apriltag_log_path()
        assert p.startswith("/tmp/falcon/apriltag_") and p.endswith(".csv")
    finally:
        del os.environ["FALCON_LOG_DIR"]


def test_report_separates_the_four_failure_modes():
    from sparx_agency.tasks.localization.apriltag_quality_report import summarize
    rows = []
    for k in range(100):
        # tag 1: good everywhere
        rows.append(dict(stamp=str(k), tag_id="1", in_map="1", used="1",
                         decision_margin="55", apparent_px="85",
                         tag_reproj_rms_px="0.8", dist_m="1.3"))
        # tag 2: mis-mapped (used, but reprojects badly)
        rows.append(dict(stamp=str(k), tag_id="2", in_map="1", used="1",
                         decision_margin="45", apparent_px="70",
                         tag_reproj_rms_px="7.9", dist_m="1.7"))
        # tag 5: hard to read (small + low margin, dropped)
        if k % 3 == 0:
            rows.append(dict(stamp=str(k), tag_id="5", in_map="1", used="0",
                             decision_margin="11", apparent_px="16",
                             tag_reproj_rms_px="", dist_m=""))
    rows.append(dict(stamp="10", tag_id="7", in_map="1", used="1",
                     decision_margin="40", apparent_px="60",
                     tag_reproj_rms_px="1.1", dist_m="2.0"))  # rarely seen
    verdicts = {s["tag_id"]: s["verdict"] for s in summarize(rows)}
    assert verdicts[1] == "GOOD"
    assert verdicts[2] == "MIS-MAPPED"
    assert verdicts[5] == "HARD TO READ"
    assert verdicts[7] == "RARELY SEEN"
