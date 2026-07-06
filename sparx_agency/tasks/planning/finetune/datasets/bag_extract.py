"""Export synchronized RGB+depth frame pairs from rosbag2 (.db3) flight
recordings into the on-disk layout ``datasets/recording.py`` reads.

Depth (``/xtend/depth_m``) is the **sparse** stream — the drone publishes it far
less often than RGB (``/xtend/rgb``; e.g. 81 depth vs 801 RGB in ``walk_into``).
So each depth frame is the anchor: we find the RGB frame whose ``header.stamp``
(sensor capture time) is closest and keep the pair only when the two fall within
``--max-dt-ms``. The two files of a pair share a six-digit index, so
``rgb/000123.png`` and ``depth/000123.npy`` are the colour+depth of one instant.

The SQLite bag is read and its CDR messages deserialized **directly on the host**
(see :mod:`cdr_image`) — no ROS, no ``ros2 bag play``, no best-effort frame
drops. Foxy- and Humble-written bags decode identically.

Output per recording (a subset of the ``recording.py`` schema; poses/ESDF are a
later step)::

    <out>/<rec>/
      rgb/000000.png        colour frame (lossless)
      depth/000000.npy      (H, W) float32 metres
      intrinsics.json       {width,height,fx,fy,cx,cy}
      meta.json             counts, timing, sync stats, source bag
      pairs.csv             per-pair matched stamps + dt_ms + depth stats

Run::

    .venv/bin/python -m sparx_agency.tasks.planning.finetune.datasets.bag_extract \
        --bags-root ~/Downloads/OneDrive_1_6-1-2026 --out-root ~/flight_dataset
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np

from .cdr_image import ImageMsg, read_camera_info, read_image, read_stamp, to_ndarray

# The eight vetted recordings (each dir holds exactly one *.db3).
GOOD_RECORDINGS = [
    "rec2", "rec3", "rec4", "rec5", "rec6",
    "rosbag2_2026_06_02-16_38_54", "rosbag2_2026_06_09-17_38_17", "walk_into",
]

# XTEND intrinsics, used only when a bag carries no /xtend/camera_info topic
# (identical across every bag that does carry it).
XTEND_INTRINSICS = {
    "width": 504, "height": 294,
    "fx": 322.635, "fy": 323.389, "cx": 242.065, "cy": 90.030,
    "distortion_model": "plumb_bob",
    "D": [-0.2972, 0.0801, -0.0037, -0.0006, 0.0],
}

RGB_TOPIC = "/xtend/rgb"
DEPTH_TOPIC = "/xtend/depth_m"
CAMERA_INFO_TOPIC = "/xtend/camera_info"


def _topic_ids(con: sqlite3.Connection) -> dict:
    return {name: tid for tid, name in con.execute("SELECT id, name FROM topics")}


def _blob(con: sqlite3.Connection, rowid: int) -> bytes:
    with con.blobopen("messages", "data", int(rowid), readonly=True) as b:
        return b.read()


def _collect_stamps(con: sqlite3.Connection, topic_id: int):
    """Return ``(stamps_ns, rowids, bag_ts)`` for a topic, sorted by header stamp.

    Header stamps are read from a cheap 64-byte prefix of each message. Falls
    back to the bag receive-time clock if the header stamps look degenerate
    (all zero / constant), returning the clock name used.
    """
    rows = con.execute(
        "SELECT rowid, timestamp FROM messages WHERE topic_id=? ORDER BY timestamp",
        (topic_id,),
    ).fetchall()
    rowids = np.array([r[0] for r in rows], dtype=np.int64)
    bag_ts = np.array([r[1] for r in rows], dtype=np.int64)
    stamps = np.empty(len(rows), dtype=np.int64)
    for i, rid in enumerate(rowids):
        with con.blobopen("messages", "data", int(rid), readonly=True) as b:
            stamps[i] = read_stamp(b.read(64))
    clock = "header"
    if stamps.size == 0 or int(stamps.min()) <= 0 or int(stamps.max() - stamps.min()) == 0:
        stamps, clock = bag_ts, "bag"
    order = np.argsort(stamps, kind="stable")
    return stamps[order], rowids[order], bag_ts[order], clock


def _match_nearest(anchor: np.ndarray, other: np.ndarray):
    """Nearest ``other`` index for each ``anchor`` stamp (+ signed dt = other-anchor)."""
    if other.size == 1:
        nn = np.zeros(anchor.size, dtype=np.int64)
        return nn, other[nn] - anchor
    idx = np.clip(np.searchsorted(other, anchor), 1, other.size - 1)
    take_left = (anchor - other[idx - 1]) <= (other[idx] - anchor)
    nn = np.where(take_left, idx - 1, idx)
    return nn, other[nn] - anchor


def _depth_to_metres(arr: np.ndarray, encoding: str) -> np.ndarray:
    if encoding in ("16UC1", "mono16"):        # uint16 millimetres (ROS convention)
        return arr.astype(np.float32) / 1000.0
    if encoding == "32FC1":                     # already metres
        return arr.astype(np.float32)
    raise ValueError(f"unsupported depth encoding {encoding!r}")


def _to_bgr(arr: np.ndarray, encoding: str) -> np.ndarray:
    if encoding == "bgr8":
        return arr
    if encoding == "rgb8":
        return arr[:, :, ::-1].copy()           # cv2.imwrite expects BGR
    if encoding in ("mono8", "8UC1"):
        return arr
    raise ValueError(f"unsupported colour encoding {encoding!r}")


def _intrinsics(con: sqlite3.Connection, tids: dict) -> tuple[dict, str]:
    if CAMERA_INFO_TOPIC in tids:
        row = con.execute(
            "SELECT rowid FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1",
            (tids[CAMERA_INFO_TOPIC],),
        ).fetchone()
        if row is not None:
            return read_camera_info(_blob(con, row[0])), "camera_info"
    return dict(XTEND_INTRINSICS), "fallback"


def export_bag(db3: Path, out_dir: Path, max_dt_ms: float, rgb_ext: str = "png") -> dict:
    """Export one bag; returns a stats dict (also written to ``meta.json``)."""
    con = sqlite3.connect(f"file:{db3}?mode=ro", uri=True)
    try:
        tids = _topic_ids(con)
        for t in (RGB_TOPIC, DEPTH_TOPIC):
            if t not in tids:
                raise ValueError(f"{db3.name}: missing topic {t}")

        d_stamp, d_row, d_bag, d_clock = _collect_stamps(con, tids[DEPTH_TOPIC])
        r_stamp, r_row, r_bag, r_clock = _collect_stamps(con, tids[RGB_TOPIC])
        nn, dt = _match_nearest(d_stamp, r_stamp)
        keep = np.abs(dt) <= max_dt_ms * 1e6

        rgb_dir, depth_dir = out_dir / "rgb", out_dir / "depth"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        manifest, saved = [], 0
        for k in np.nonzero(keep)[0]:
            dmsg: ImageMsg = read_image(_blob(con, d_row[k]))
            depth = _depth_to_metres(to_ndarray(dmsg), dmsg.encoding)
            rmsg: ImageMsg = read_image(_blob(con, r_row[nn[k]]))
            bgr = _to_bgr(to_ndarray(rmsg), rmsg.encoding)

            np.save(depth_dir / f"{saved:06d}.npy", depth)
            cv2.imwrite(str(rgb_dir / f"{saved:06d}.{rgb_ext}"), bgr)

            valid = depth > 0
            nz = depth[valid]
            manifest.append({
                "idx": saved,
                "depth_stamp_ns": int(d_stamp[k]), "rgb_stamp_ns": int(r_stamp[nn[k]]),
                "dt_ms": round(float(dt[k]) / 1e6, 3),
                "depth_bag_ns": int(d_bag[k]), "rgb_bag_ns": int(r_bag[nn[k]]),
                "depth_min_m": round(float(nz.min()), 3) if nz.size else 0.0,
                "depth_max_m": round(float(nz.max()), 3) if nz.size else 0.0,
                "valid_frac": round(float(valid.mean()), 4),
            })
            saved += 1

        intr, intr_src = _intrinsics(con, tids)
        (out_dir / "intrinsics.json").write_text(json.dumps(
            {k: intr[k] for k in ("width", "height", "fx", "fy", "cx", "cy")}, indent=2))

        dt_kept_ms = np.abs(dt[keep]) / 1e6
        span_s = float(d_stamp[keep][-1] - d_stamp[keep][0]) / 1e9 if saved > 1 else 0.0
        stats = {
            "source_bag": str(db3),
            "num_depth_msgs": int(d_stamp.size), "num_rgb_msgs": int(r_stamp.size),
            "num_pairs": saved, "num_rejected_dt": int((~keep).sum()),
            "max_dt_ms": max_dt_ms, "depth_clock": d_clock, "rgb_clock": r_clock,
            "dt_ms_median": round(float(np.median(dt_kept_ms)), 3) if saved else None,
            "dt_ms_p95": round(float(np.percentile(dt_kept_ms, 95)), 3) if saved else None,
            "dt_ms_max": round(float(dt_kept_ms.max()), 3) if saved else None,
            "width": intr["width"], "height": intr["height"],
            "rate_hz": round((saved - 1) / span_s, 3) if span_s > 0 else None,
            "frames": saved, "intrinsics_source": intr_src,
            "rgb_ext": rgb_ext,
            "_todo_hardware": ["camera_height_m", "pitch_deg"],
        }
        (out_dir / "meta.json").write_text(json.dumps(stats, indent=2))
        with (out_dir / "pairs.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()) if manifest else ["idx"])
            w.writeheader()
            w.writerows(manifest)
        return stats
    finally:
        con.close()


def _find_db3(rec_dir: Path) -> Path:
    dbs = sorted(rec_dir.glob("*.db3"))
    if not dbs:
        raise FileNotFoundError(f"no .db3 in {rec_dir}")
    return dbs[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bags-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--recordings", nargs="*", default=GOOD_RECORDINGS)
    ap.add_argument("--max-dt-ms", type=float, default=80.0)
    ap.add_argument("--rgb-ext", default="png", choices=["png", "jpg"])
    args = ap.parse_args()

    bags_root, out_root = args.bags_root.expanduser(), args.out_root.expanduser()
    print(f"{'recording':<32} {'pairs':>6} {'rej':>5} {'depth':>6} {'rgb':>6} "
          f"{'clock':>7} {'dt_med':>7} {'dt_p95':>7} {'dt_max':>7}")
    for rec in args.recordings:
        rec_dir = bags_root / rec
        if not rec_dir.is_dir():
            print(f"{rec:<32}  SKIP (missing)")
            continue
        try:
            s = export_bag(_find_db3(rec_dir), out_root / rec, args.max_dt_ms, args.rgb_ext)
        except Exception as exc:  # noqa: BLE001 - report per-bag, keep going
            print(f"{rec:<32}  ERROR: {exc}")
            continue
        print(f"{rec:<32} {s['num_pairs']:>6} {s['num_rejected_dt']:>5} "
              f"{s['num_depth_msgs']:>6} {s['num_rgb_msgs']:>6} "
              f"{s['depth_clock']+'/'+s['rgb_clock'][:1]:>7} "
              f"{s['dt_ms_median'] or 0:>7} {s['dt_ms_p95'] or 0:>7} {s['dt_ms_max'] or 0:>7}")


if __name__ == "__main__":
    main()
