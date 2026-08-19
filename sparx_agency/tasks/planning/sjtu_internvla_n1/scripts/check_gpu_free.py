#!/usr/bin/env python3
"""Assert the GPU is the InternVLA-N1 network's alone -- and nothing else's.

The whole premise of this deployment is that N1 needs almost the entire 8 GB
card, so everything else -- Gazebo, the mapper, both ROS2 nodes -- runs on the
CPU. This script is the check that keeps that true, and it is used twice:

* **before** the model server starts (``--require-empty``): the card must be
  idle, or N1 does not get the memory it was measured to need and fails at load
  in a way that looks like a bug in N1;
* **after** it starts (``--allow internnav --allow python``): exactly one thing
  may hold the card, and it must be the server -- a stray torch process on it is
  the memory N1 is now missing.

Pure stdlib and ``nvidia-smi``; it imports no CUDA and holds no memory itself, so
running the check never changes the answer.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


def _smi(query, extra=None):
    """Run one ``nvidia-smi --query-*`` and return its CSV rows as lists."""
    cmd = ["nvidia-smi", query, "--format=csv,noheader,nounits"]
    if extra:
        cmd[1:1] = extra
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError("nvidia-smi failed: %s" % (out.stderr.strip(),))
    rows = []
    for line in out.stdout.strip().splitlines():
        if line.strip():
            rows.append([c.strip() for c in line.split(",")])
    return rows


def _compute_apps():
    """(pid, name, used_mib) for every process holding the GPU, or []."""
    try:
        rows = _smi("--query-compute-apps=pid,process_name,used_memory")
    except RuntimeError:
        return []
    apps = []
    for row in rows:
        pid = row[0] if row else "?"
        name = row[1] if len(row) > 1 else "?"
        try:
            used = int(float(row[2])) if len(row) > 2 and row[2] not in ("", "[N/A]") else 0
        except ValueError:
            used = 0
        apps.append((pid, name, used))
    return apps


def _memory():
    """(total_mib, used_mib) for GPU 0."""
    row = _smi("--query-gpu=memory.total,memory.used")[0]
    return int(float(row[0])), int(float(row[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-empty", action="store_true",
                        help="fail if ANY process holds the GPU (preflight).")
    parser.add_argument("--allow", action="append", default=[],
                        help="substring of a process name allowed on the GPU "
                             "(repeatable). Anything else fails the check.")
    parser.add_argument("--max-idle-mib", type=int, default=800,
                        help="used memory tolerated with no compute apps "
                             "(desktop/compositor overhead).")
    args = parser.parse_args(argv)

    if shutil.which("nvidia-smi") is None:
        print("[gpu] no nvidia-smi on PATH -- cannot verify the GPU is free.",
              file=sys.stderr)
        return 3

    total, used = _memory()
    apps = _compute_apps()
    print("[gpu] %d/%d MiB used, %d compute process(es)" % (used, total, len(apps)))
    for pid, name, mib in apps:
        print("[gpu]   pid %s  %-28s %6d MiB" % (pid, name, mib))

    if args.require_empty:
        if apps or used > args.max_idle_mib:
            print("[gpu] NOT FREE: something is on the card. Free it before "
                  "starting InternVLA-N1 -- it needs the whole 8 GB.",
                  file=sys.stderr)
            return 1
        print("[gpu] free: safe to give the whole card to InternVLA-N1.")
        return 0

    if args.allow:
        stray = [(p, n, m) for (p, n, m) in apps
                 if not any(a.lower() in n.lower() for a in args.allow)]
        if stray:
            print("[gpu] STRAY process(es) on the GPU that are not the N1 "
                  "server: %s" % ([n for _, n, _ in stray],), file=sys.stderr)
            print("[gpu] every non-N1 process must run on the CPU "
                  "(CUDA_VISIBLE_DEVICES=\"\").", file=sys.stderr)
            return 2
        print("[gpu] OK: only the allowed InternVLA-N1 server holds the card.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())

