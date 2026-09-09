"""Compare adjacent segments of a within-flight paired run.

The companion to :mod:`paired_flipper`. That alternates a follower parameter
during a flight; this reads the resulting flight back and compares **neighbouring**
segments, which is what makes the design valid.

Segments are derived from the flight's own trace — the follower records
``gate.yaw_mode`` on every tick — so no external switch log is needed and a
missed or failed parameter write cannot desynchronise the analysis from what the
aircraft was actually doing.

**Adjacent pairs, never pooled.** A flight is not stationary in time: early
segments explore virgin space and late ones revisit, and voxels-per-metre decays
4.2x across a flight regardless of arm (LOOP_BUGS.md B10). Pooling every A
against every B would read that decay as an effect. Comparing each segment with
its immediate neighbour cancels it, because neighbours face near-identical maps.

Reports the median of the per-pair ratios, which is the paired statistic — not
the ratio of the medians.

Usage::

    PYTHONPATH=. .venv/bin/python -m sparx_agency.tools.falcon_campaign.paired_segments runs/<stamp>Z
"""
from __future__ import annotations

import argparse
import math
import statistics as st
from pathlib import Path

from sparx_agency.tools.falcon_campaign import trace_metrics

#: Ticks a segment must contain before it is comparable; shorter ones are
#: switch-boundary fragments, not samples.
MIN_TICKS = 40


def segment(ctrl_rows, key="yaw_mode"):
    """Split ctrl rows into runs of constant ``gate[key]``.

    Args:
        ctrl_rows: The ``ctrl`` records of one flight's trace.
        key: Gate field whose changes delimit segments.

    Returns:
        A list of ``(value, rows)`` in flight order.
    """
    out = []
    current = None
    for row in ctrl_rows:
        gate = (row.get("trace") or {}).get("gate") or {}
        value = gate.get(key)
        if value is None:
            continue
        if current is None or value != current[0]:
            current = (value, [])
            out.append(current)
        current[1].append(row)
    return [(v, r) for v, r in out if len(r) >= MIN_TICKS]


def segment_stats(rows):
    """Per-segment quantities that are meaningful over a short window.

    Args:
        rows: The ctrl rows of one segment.

    Returns:
        A dict of scalars, or None when the segment carries no usable ticks.
    """
    hdg, cross, speed = [], [], []
    for row in rows:
        trace = row.get("trace") or {}
        track = trace.get("tracking") or {}
        gate = trace.get("gate") or {}
        smoothed = (trace.get("terms") or {}).get("smoothed")
        if gate.get("heading_err_rad") is not None:
            hdg.append(abs(math.degrees(gate["heading_err_rad"])))
        if track.get("cross_track_error_m") is not None:
            cross.append(abs(track["cross_track_error_m"]))
        if smoothed:
            speed.append(math.hypot(smoothed[0], smoothed[1]))
    if not speed:
        return None
    return dict(ticks=len(rows),
                hdg_err_deg=st.median(hdg) if hdg else None,
                cross_track_m=st.median(cross) if cross else None,
                cmd_speed_mps=st.median(speed))


def paired_ratios(segments, field):
    """Ratios between each adjacent pair of differing segments.

    Args:
        segments: Output of :func:`segment`.
        field: Key from :func:`segment_stats` to compare.

    Returns:
        ``(label_a, label_b, ratios)`` where each ratio is a-over-b for one
        adjacent pair; an empty list when nothing is comparable.
    """
    stats = [(v, segment_stats(r)) for v, r in segments]
    stats = [(v, s) for v, s in stats if s and s.get(field) is not None]
    ratios, la, lb = [], None, None
    for (va, sa), (vb, sb) in zip(stats, stats[1:]):
        if va == vb or not sb[field]:
            continue
        if la is None:
            la, lb = va, vb
        r = sa[field] / sb[field]
        ratios.append(r if (va, vb) == (la, lb) else 1.0 / r)
    return la, lb, ratios


def main(argv=None):
    """Report the paired comparison for one run."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--key", default="yaw_mode")
    args = ap.parse_args(argv)

    rows = trace_metrics.read_trace(args.run_dir / "flight_trace.jsonl")
    segs = segment(rows.get("ctrl") or [], args.key)
    if not segs:
        print("no segments of >= %d ticks -- was the flip running?" % MIN_TICKS)
        return 1
    print("%d segments: %s" % (len(segs), " ".join(str(v) for v, _ in segs)))
    if len(set(v for v, _ in segs)) < 2:
        print("only one distinct value -- this is not a paired run")
        return 1
    for field in ("hdg_err_deg", "cross_track_m", "cmd_speed_mps"):
        la, lb, ratios = paired_ratios(segs, field)
        if not ratios:
            continue
        print("  %-16s %s/%s  pairs=%d  median ratio %.3f  (%s)"
              % (field, la, lb, len(ratios), st.median(ratios),
                 " ".join("%.2f" % r for r in ratios[:8])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
