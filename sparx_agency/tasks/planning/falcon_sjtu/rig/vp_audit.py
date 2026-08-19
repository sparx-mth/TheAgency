#!/usr/bin/env python3
"""Aggregate the frontier finder's viewpoint-placement audit from a FALCON log.

A cluster retires to dormant when ``sampleViewpoints()`` returns nothing, and
two entirely different failures wear that same verdict:

* **nowhere to stand** -- no candidate position on the sampling ring survives
  the box / occupied / unknown / near-unknown tests, so visibility is never
  even evaluated;
* **nothing to see** -- candidate positions exist, but every ray from them to
  the frontier is judged blocked, so the cluster is rejected on visibility.

They need opposite fixes, and a premature FINISH looks identical either way
from outside. ``falcon_vp_audit.patch`` makes the frontier finder say which,
once per retirement; this reads those lines back.

The ray columns are the sharper half. A ray blocked by OCCUPIED is a wall doing
its job. A ray blocked by UNKNOWN is the visibility test refusing to look at a
frontier *because* it is a frontier -- the boundary of unobserved space is
unobserved by construction -- and that is a self-inflicted wound rather than a
property of the building.

Usage::

    rig/vp_audit.py <run_dir>/falcon.log [--top 15]
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys

# [vp_audit] retire (x,y,z) cells=N relax=R | cand=.. keepout=.. outbox=.. occ=..
# unk=.. nearunk=.. placeable=.. | visib0=.. belowbar=.. maxvisib=.. bar=..
# | rays=.. nofov=.. blockocc=.. blockunk=.. clear=..
_LINE = re.compile(
    r"\[vp_audit\] retire \((?P<x>-?[\d.]+),(?P<y>-?[\d.]+),(?P<z>-?[\d.]+)\) "
    r"cells=(?P<cells>\d+) relax=(?P<relax>\d+) \| "
    r"cand=(?P<cand>\d+) keepout=(?P<keepout>\d+) outbox=(?P<outbox>\d+) "
    r"occ=(?P<occ>\d+) unk=(?P<unk>\d+) nearunk=(?P<nearunk>\d+) "
    r"placeable=(?P<placeable>\d+) \| "
    r"visib0=(?P<visib0>\d+) belowbar=(?P<belowbar>\d+) "
    r"maxvisib=(?P<maxvisib>\d+) bar=(?P<bar>\d+) \| "
    r"rays=(?P<rays>\d+) nofov=(?P<nofov>\d+) blockocc=(?P<blockocc>\d+) "
    r"blockunk=(?P<blockunk>\d+) clear=(?P<clear>\d+)"
)

_INT_FIELDS = ("cells", "relax", "cand", "keepout", "outbox", "occ", "unk",
               "nearunk", "placeable", "visib0", "belowbar", "maxvisib", "bar",
               "rays", "nofov", "blockocc", "blockunk", "clear")


def parse(path):
    """Yield one dict per retirement recorded in ``path``.

    Args:
        path: Path to a FALCON container log.

    Yields:
        A dict of the audit fields, with the position under ``pos``.
    """
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            match = _LINE.search(line)
            if match is None:
                continue
            row = {k: int(match.group(k)) for k in _INT_FIELDS}
            row["pos"] = (float(match.group("x")), float(match.group("y")),
                          float(match.group("z")))
            yield row


_UNK = re.compile(
    r"\[vp_unk\] nostruct=(?P<nostruct>\d+) unk0=(?P<unk0>\d+) "
    r"unk1_2=(?P<unk1_2>\d+) unk3_5=(?P<unk3_5>\d+) unk6_10=(?P<unk6_10>\d+) "
    r"unk11_20=(?P<unk11_20>\d+) unk21p=(?P<unk21p>\d+)"
)


def parse_unknown(path):
    """Yield the per-retirement distribution of unobserved voxels crossed.

    Emitted alongside each retirement by ``falcon_vp_audit.patch``. Only rays
    that meet no OCCUPIED voxel are counted, so these are exactly the lines of
    sight a larger unknown budget could recover.

    Args:
        path: Path to a FALCON container log.

    Yields:
        A dict of bucket counts per retirement.
    """
    with open(path, "r", errors="replace") as handle:
        for line in handle:
            match = _UNK.search(line)
            if match is not None:
                yield {k: int(v) for k, v in match.groupdict().items()}


def classify(row):
    """Name the reason this cluster retired.

    Placeability is decided before visibility ever runs, so the two are not
    competing explanations for one retirement -- they are sequential gates, and
    the first one that empties decides.

    Args:
        row: One parsed audit record.

    Returns:
        A short reason string.
    """
    if row["placeable"] == 0:
        if row["cand"] == 0:
            return "no candidates generated"
        dominant = max((row["keepout"], "keep-out"), (row["outbox"], "outside box"),
                       (row["occ"], "standing in an obstacle"),
                       (row["unk"], "standing in unknown"),
                       (row["nearunk"], "too near unknown"))
        return "nowhere to stand: " + dominant[1]
    if row["maxvisib"] == 0:
        if row["rays"] and row["blockunk"] >= row["blockocc"]:
            return "nothing visible: rays blocked by UNKNOWN"
        if row["rays"] and row["nofov"] >= row["rays"] // 2:
            return "nothing visible: cluster outside the camera cone"
        return "nothing visible: rays blocked by structure"
    return "saw something, under the visibility bar"


def _pct(part, whole):
    return "%5.1f%%" % (100.0 * part / whole) if whole else "    - "


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=pathlib.Path)
    ap.add_argument("--top", type=int, default=15,
                    help="how many repeat offenders to list")
    args = ap.parse_args(argv)

    if not args.log.exists():
        sys.exit("no such log: %s" % args.log)

    rows = list(parse(args.log))
    if not rows:
        sys.exit("no [vp_audit] lines in %s -- is falcon_vp_audit.patch in the image?"
                 % args.log)

    reasons = collections.Counter(classify(r) for r in rows)
    total = len(rows)

    print("viewpoint-placement audit -- %s" % args.log)
    print("%d cluster retirements recorded (%d of them on an amnesty pass)\n"
          % (total, sum(1 for r in rows if r["relax"])))

    print("WHY THE CLUSTER RETIRED")
    for reason, count in reasons.most_common():
        print("  %-46s %6d  %s" % (reason, count, _pct(count, total)))

    placeable = [r for r in rows if r["placeable"] > 0]
    print("\nGATE ONE -- was there anywhere legal to stand?")
    print("  reached the visibility test               %6d  %s"
          % (len(placeable), _pct(len(placeable), total)))
    for key, label in (("keepout", "keep-out"), ("outbox", "outside box"),
                       ("occ", "occupied"), ("unk", "unknown"),
                       ("nearunk", "near unknown")):
        share = sum(r[key] for r in rows)
        cand = sum(r["cand"] for r in rows)
        print("  candidates rejected: %-22s %6d  %s" % (label, share, _pct(share, cand)))

    if placeable:
        rays = sum(r["rays"] for r in placeable)
        print("\nGATE TWO -- from those positions, was anything visible?")
        print("  rays cast to frontier cells              %6d" % rays)
        for key, label in (("nofov", "outside the camera cone"),
                           ("blockocc", "blocked by OCCUPIED (a real wall)"),
                           ("blockunk", "blocked by UNKNOWN (unobserved)"),
                           ("clear", "clear -- cell seen")):
            share = sum(r[key] for r in placeable)
            print("  %-40s %6d  %s" % (label, share, _pct(share, rays)))
        blind = [r for r in placeable if r["maxvisib"] == 0]
        print("  clusters with a legal stance and ZERO visibility  %6d  %s"
              % (len(blind), _pct(len(blind), len(placeable))))

    unk = list(parse_unknown(args.log))
    if unk:
        nostruct = sum(r["nostruct"] for r in unk)
        buckets = [("0", "unk0"), ("1-2", "unk1_2"), ("3-5", "unk3_5"),
                   ("6-10", "unk6_10"), ("11-20", "unk11_20"), ("21+", "unk21p")]
        print("\nWHAT A LARGER UNKNOWN BUDGET WOULD BUY")
        print("  rays with NO known structure in the way    %6d" % nostruct)
        cumulative = 0
        for label, key in buckets:
            share = sum(r[key] for r in unk)
            cumulative += share
            print("  crossing %-6s unobserved voxels: %6d   (budget >= this recovers %s of them)"
                  % (label, share, _pct(cumulative, nostruct)))

    print("\nREPEAT OFFENDERS (same place retiring over and over)")
    by_place = collections.Counter(
        (round(r["pos"][0], 0), round(r["pos"][1], 0)) for r in rows)
    for (x, y), count in by_place.most_common(args.top):
        print("  (%6.1f, %6.1f)  %5d retirements" % (x, y, count))


if __name__ == "__main__":
    main()
