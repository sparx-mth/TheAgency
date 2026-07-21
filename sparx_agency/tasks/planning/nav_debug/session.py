"""Load a recorded run into a timeline of :class:`NavFrame` for replay.

A run folder (written by ``nav_debug_recorder_node``) holds the BEV maps, the
three route layers and the replan events; the per-tick **certainty CSV** (written
by the flight node, unchanged) holds pose, both command sets, drift and
localization quality. This module joins them: the CSV rows (or, if it is absent,
the recorder's own ``telemetry.jsonl``) define the frame timeline, and each
frame's map / routes / active replan event are resolved by an *as-of* join (the
most recent one at or before the frame's timestamp) -- exactly how the drone saw
them at that instant, latched topics and all.

Frames are built lazily (:meth:`NavSession.build`) so a long flight's hundreds of
BEV snapshots are never all in memory at once; the last few grids are cached so
stepping back and forth is cheap.
"""
from __future__ import annotations

import bisect
import csv
import glob
import json
import math
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

from sparx_agency.tasks.planning.nav_debug.frame import (
    BevMap, Drift, NavFrame, Quality, ReplanEvent, Routes,
)

_TRAIL_LEN = 48        # localization-trail length (frames)
_HIST_LEN = 64         # command/confidence strip length (frames)
_EVENT_WINDOW_S = 6.0  # a replan banner lingers this long after the event fired
_DEG = math.pi / 180.0


def _f(value) -> Optional[float]:
    """Parse a CSV/JSON scalar to float, or None for '' / None / bad values."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_event(text: str) -> str:
    """Bucket a raw planner event string into a coarse replan ``kind``."""
    t = (text or "").lower()
    if "boxed in" in t:
        return "boxed_in"
    if "blockage" in t or "unseen obstacle" in t:
        return "blockage"
    if "obstacle on route" in t or "collision" in t:
        return "obstacle"
    if "rotat" in t:
        return "rotation"
    if "periodic" in t:
        return "time"
    if "reopened" in t or "resuming" in t:
        return "info"
    return "info"


def _read_jsonl(path: str) -> List[dict]:
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue    # a half-written trailing line must not abort a replay
    return rows


class NavSession:
    """A loaded run: a list of frame records + as-of indexes for map/routes/events."""

    def __init__(self, run_dir: str, csv_path: Optional[str] = None) -> None:
        self.run_dir = run_dir
        self.manifest = self._load_manifest(run_dir)
        self._bev_cache = OrderedDict()      # npy_path -> BevMap

        # Rich per-frame telemetry: the certainty CSV if we can find one, else the
        # recorder's own telemetry.jsonl (pose + our command only).
        self.csv_path = self._resolve_csv(run_dir, csv_path)
        self.rows = self._load_rows(self.csv_path, run_dir)
        if not self.rows:
            raise ValueError(
                "no frames to replay in %r (need a certainty CSV or telemetry.jsonl)"
                % run_dir)
        self._stamps = [r["t"] for r in self.rows]
        # The certainty CSV logs the drone's vertical axis count but not our own
        # cmd_vel.linear.z, so backfill OURS vz from the recorder's telemetry.jsonl
        # (an as-of join) -- else the OURS vertical gauge would read 0 in a climb.
        self._fill_missing_cmd_vz(run_dir)

        # As-of sources.
        self._bev_index = self._index_bev(run_dir)               # [(t, npy, meta)]
        self._bev_stamps = [t for t, _, _ in self._bev_index]
        self._routes = sorted(_read_jsonl(os.path.join(run_dir, "routes.jsonl"))
                              + self._index_routes_dir(run_dir), key=lambda d: d.get("t", 0.0))
        self._route_stamps = [d.get("t", 0.0) for d in self._routes]
        self._events = sorted(_read_jsonl(os.path.join(run_dir, "events.jsonl")),
                              key=lambda d: d.get("t", 0.0))
        self._event_stamps = [d.get("t", 0.0) for d in self._events]

    def __len__(self) -> int:
        return len(self.rows)

    # ── loading ──────────────────────────────────────────────────────────────
    @staticmethod
    def _load_manifest(run_dir: str) -> dict:
        p = os.path.join(run_dir, "manifest.json")
        if os.path.isfile(p):
            try:
                with open(p) as fh:
                    return json.load(fh)
            except ValueError:
                pass
        return {}

    def _resolve_csv(self, run_dir: str, csv_path: Optional[str]) -> Optional[str]:
        if csv_path:
            return csv_path
        m = self.manifest.get("certainty_csv")
        if m and os.path.isfile(m):
            return m
        # Auto-find the newest certainty_*.csv beside the run (in it or its parent).
        cands = (glob.glob(os.path.join(run_dir, "certainty_*.csv"))
                 + glob.glob(os.path.join(os.path.dirname(run_dir.rstrip("/")),
                                          "certainty_*.csv")))
        return max(cands, key=os.path.getmtime) if cands else None

    def _load_rows(self, csv_path: Optional[str], run_dir: str) -> List[dict]:
        if csv_path and os.path.isfile(csv_path):
            return self._rows_from_csv(csv_path)
        return self._rows_from_telemetry(os.path.join(run_dir, "telemetry.jsonl"))

    @staticmethod
    def _rows_from_csv(path: str) -> List[dict]:
        out = []
        with open(path, "r") as fh:
            for r in csv.DictReader(fh):
                t = _f(r.get("ros_stamp"))
                x, y = _f(r.get("pos_x")), _f(r.get("pos_y"))
                if t is None or x is None or y is None:
                    continue
                yaw_deg = _f(r.get("yaw_deg"))
                fwd, lat = _f(r.get("axis_forward")), _f(r.get("axis_lateral"))
                vert, ayaw = _f(r.get("axis_vertical")), _f(r.get("axis_yaw"))
                out.append({
                    "t": t, "x": x, "y": y, "z": _f(r.get("pos_z")),
                    "yaw": (yaw_deg or 0.0) * _DEG,
                    "vx": _f(r.get("cmd_vx")), "vy": _f(r.get("cmd_vy")),
                    "vz": None, "wz": _f(r.get("cmd_wz")),
                    "drone": None if fwd is None else (int(fwd), int(lat or 0),
                                                       int(vert or 0), int(ayaw or 0)),
                    "conf": _f(r.get("confidence")),
                    "quality": Quality(
                        confidence=_f(r.get("confidence")) or 0.0,
                        pos_std_m=_f(r.get("pos_std_m")) or 0.0,
                        cmd_effectiveness=_f(r.get("cmd_effectiveness")) or 0.0,
                        coasting=str(r.get("coasting", "")).lower() in ("true", "1"),
                        age_s=_f(r.get("age_s")) or 0.0,
                        source=str(r.get("state", ""))),
                    "drift": Drift(
                        drift_vx=_f(r.get("drift_vx")) or 0.0,
                        drift_vy=_f(r.get("drift_vy")) or 0.0,
                        drift_wz=_f(r.get("drift_wz")) or 0.0,
                        cross_track_m=_f(r.get("cross_track_m")) or 0.0,
                        along_track_m=_f(r.get("along_track_m")) or 0.0,
                        heading_err_deg=_f(r.get("heading_err_deg")) or 0.0,
                        effort=_f(r.get("effort")) or 0.0,
                        speed_scale=_f(r.get("speed_scale")) or 0.0,
                        authority=str(r.get("authority", "")),
                        state=str(r.get("state", "")),
                        escape_state=str(r.get("escape_state", "")),
                        blocked_axis=str(r.get("blocked_axis", ""))),
                    "wp_idx": _f(r.get("target_wp_idx")),
                    "num_wp": _f(r.get("num_waypoints")),
                    "tx": _f(r.get("target_x")), "ty": _f(r.get("target_y")),
                })
        return out

    @staticmethod
    def _rows_from_telemetry(path: str) -> List[dict]:
        out = []
        for r in _read_jsonl(path):
            t, x, y = _f(r.get("t")), _f(r.get("x")), _f(r.get("y"))
            if t is None or x is None or y is None:
                continue
            out.append({
                "t": t, "x": x, "y": y, "z": _f(r.get("z")), "yaw": _f(r.get("yaw")) or 0.0,
                "vx": _f(r.get("vx")), "vy": _f(r.get("vy")),
                "vz": _f(r.get("vz")), "wz": _f(r.get("wz")),
                "drone": None, "conf": None, "quality": None, "drift": None,
                "wp_idx": None, "num_wp": None, "tx": None, "ty": None,
            })
        return out

    @staticmethod
    def _index_bev(run_dir: str) -> List[Tuple[float, str, dict]]:
        bev_dir = os.path.join(run_dir, "bev")
        index = []
        for npy in sorted(glob.glob(os.path.join(bev_dir, "*.npy"))):
            meta = {}
            side = npy[:-4] + ".json"
            if os.path.isfile(side):
                try:
                    with open(side) as fh:
                        meta = json.load(fh)
                except ValueError:
                    meta = {}
            t = meta.get("t")
            if t is None:
                try:
                    t = float(os.path.splitext(os.path.basename(npy))[0]) / 1000.0
                except ValueError:
                    continue
            index.append((float(t), npy, meta))
        index.sort(key=lambda e: e[0])
        return index

    @staticmethod
    def _index_routes_dir(run_dir: str) -> List[dict]:
        out = []
        for jp in glob.glob(os.path.join(run_dir, "routes", "*.json")):
            try:
                with open(jp) as fh:
                    d = json.load(fh)
            except ValueError:
                continue
            if "t" not in d:
                try:
                    d["t"] = float(os.path.splitext(os.path.basename(jp))[0]) / 1000.0
                except ValueError:
                    continue
            out.append(d)
        return out

    def _fill_missing_cmd_vz(self, run_dir: str) -> None:
        telem = self._rows_from_telemetry(os.path.join(run_dir, "telemetry.jsonl"))
        if not telem:
            return
        ts = [d["t"] for d in telem]
        for r in self.rows:
            if r.get("vz") is None:
                j = self._as_of(ts, r["t"])
                if j is not None:
                    r["vz"] = telem[j].get("vz")

    # ── as-of joins ────────────────────────────────────────────────────────────
    @staticmethod
    def _as_of(stamps: List[float], t: float) -> Optional[int]:
        """Index of the latest stamp <= ``t`` (None if all are later)."""
        if not stamps:
            return None
        j = bisect.bisect_right(stamps, t) - 1
        return j if j >= 0 else None

    def _load_bev(self, npy: str, meta: dict) -> Optional[BevMap]:
        if npy in self._bev_cache:
            self._bev_cache.move_to_end(npy)
            return self._bev_cache[npy]
        try:
            grid = np.load(npy)
        except (OSError, ValueError):
            return None
        geo = meta or self.manifest.get("bev", {})
        bev = BevMap(grid=grid,
                     resolution=float(geo.get("resolution", 0.15)),
                     origin_x=float(geo.get("origin_x", 0.0)),
                     origin_y=float(geo.get("origin_y", 0.0)),
                     frame_id=str(geo.get("frame_id", "world")),
                     stamp=float(geo.get("t", 0.0)))
        self._bev_cache[npy] = bev
        if len(self._bev_cache) > 4:
            self._bev_cache.popitem(last=False)
        return bev

    def _bev_at(self, t: float) -> Tuple[Optional[BevMap], Optional[np.ndarray]]:
        j = self._as_of(self._bev_stamps, t)
        if j is None:
            return None, None
        _, npy, meta = self._bev_index[j]
        bev = self._load_bev(npy, meta)
        conf_path = os.path.join(self.run_dir, "bev_conf", os.path.basename(npy))
        conf = None
        if os.path.isfile(conf_path):
            try:
                conf = np.load(conf_path)
            except (OSError, ValueError):
                conf = None
        return bev, conf

    def _routes_at(self, t: float) -> Routes:
        j = self._as_of(self._route_stamps, t)
        if j is None:
            return Routes()
        d = self._routes[j]

        def _pts(key):
            v = d.get(key)
            return [(float(p[0]), float(p[1])) for p in v] if v else None

        def _pt(key):
            v = d.get(key)
            return (float(v[0]), float(v[1])) if v else None

        return Routes(astar=_pts("astar"), safe=_pts("safe"), final=_pts("final"),
                      goal=_pt("goal"), lookahead=_pt("lookahead"))

    def _replan_at(self, t: float) -> Optional[ReplanEvent]:
        j = self._as_of(self._event_stamps, t)
        while j is not None and j >= 0:
            d = self._events[j]
            et = float(d.get("t", 0.0))
            if t - et > _EVENT_WINDOW_S:
                return None
            text = str(d.get("text", ""))
            kind = str(d.get("kind") or classify_event(text))
            if kind == "info":       # skip pure info; surface the last real replan
                j -= 1
                continue
            xy = None
            if _f(d.get("x")) is not None and _f(d.get("y")) is not None:
                xy = (float(d["x"]), float(d["y"]))
            return ReplanEvent(stamp=et, kind=kind, text=text, age_s=t - et, xy=xy)
        return None

    # ── frame assembly ─────────────────────────────────────────────────────────
    def build(self, i: int) -> NavFrame:
        """Assemble the :class:`NavFrame` for frame ``i`` (0-based)."""
        n = len(self.rows)
        if not 0 <= i < n:
            raise IndexError("frame %d out of range [0, %d)" % (i, n))
        r = self.rows[i]
        t = r["t"]
        vz = r.get("vz")
        our = None
        if r.get("vx") is not None or r.get("wz") is not None:
            our = (r.get("vx") or 0.0, r.get("vy") or 0.0, vz or 0.0, r.get("wz") or 0.0)

        target = None
        if r.get("wp_idx") is not None and r.get("tx") is not None:
            target = (int(r["wp_idx"]), int(r.get("num_wp") or 0), r["tx"], r["ty"])
        advanced = bool(i > 0 and r.get("wp_idx") is not None
                        and self.rows[i - 1].get("wp_idx") is not None
                        and r["wp_idx"] > self.rows[i - 1]["wp_idx"])

        lo = max(0, i - _TRAIL_LEN)
        trail = [(self.rows[k]["x"], self.rows[k]["y"]) for k in range(lo, i)]
        hlo = max(0, i - _HIST_LEN)
        cmd_hist = [self.rows[k].get("vx") or 0.0 for k in range(hlo, i + 1)]
        conf_hist = [self.rows[k].get("conf") or 0.0 for k in range(hlo, i + 1)
                     if self.rows[k].get("conf") is not None]

        bev, conf = self._bev_at(t)
        replan = self._replan_at(t)

        return NavFrame(
            stamp=t, x=r["x"], y=r["y"], yaw=r["yaw"], z=r.get("z"), trail=trail,
            our_cmd=our, drone_cmd=r.get("drone"),
            quality=r.get("quality"), drift=r.get("drift"),
            target=target, advanced=advanced,
            bev=bev, bev_conf=conf, routes=self._routes_at(t), replan=replan,
            cmd_history=cmd_hist, conf_history=conf_hist,
            why=self._why(r, our, replan))

    @staticmethod
    def _why(r: dict, our, replan: Optional[ReplanEvent]) -> str:
        """One-line 'why' for the panel: the drift authority + a climb note."""
        drift = r.get("drift")
        parts = []
        if drift is not None:
            if drift.escape_state and drift.escape_state != "IDLE":
                parts.append("escaping (%s)" % drift.escape_state)
            elif drift.authority:
                parts.append(drift.authority)
            if abs(drift.drift_vy) * 100.0 >= 1.0:
                parts.append("holding %.0fcm/s %s roll vs drift"
                             % (abs(drift.drift_vy) * 100.0,
                                "left" if drift.drift_vy > 0 else "right"))
        if our is not None and our[2] > 0.02:
            parts.append("climbing")
        return "; ".join(parts)
