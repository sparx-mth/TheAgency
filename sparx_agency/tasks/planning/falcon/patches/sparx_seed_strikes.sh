#!/bin/bash
# ============================================================
# sparx_seed_strikes.sh
#
# Lets a SEEDED blocked region enter at a chosen strike count.
#
# WHY (LOOP_BUGS.md B45). frontier_finder.cpp restores
# /frontier_finder/blocked_regions_runtime with strikes = 1, and its own comment
# says that at strike 1 the blocked radius is deliberately under candidate_rmax
# so the test "cannot fire for the frontier that produced the blocked
# viewpoint... That is correct exactly once: it stops a single early mistake
# retiring a transit route. From the second strike the width exceeds
# candidate_rmax and the frontier retires."
#
# That is right for a RUNTIME veto, which may be a one-off. It is wrong for a
# seed: F21's twelve cells are the distillation of 652 wedging episodes across
# 985 flights, and the worst of them trapped the aircraft on 46 separate
# flights. Loading that evidence at the strength reserved for "might be a
# mistake" means it is deliberately ignored -- which is exactly what F21
# measured: the regions loaded, and nothing changed.
#
# So make the restored strike count a parameter. Default 1 keeps the shipped
# behaviour exactly; 2 lets a seeded region retire frontiers the way the code
# already treats anything that has failed twice.
#
# Self-contained (no .patch file): edits the source in place and verifies.
# ============================================================
set -euo pipefail

F=/catkin_ws/src/FALCON/falcon_planner/exploration_preprocessing/src/frontier_finder.cpp
[ -f "$F" ] || { echo "[sparx-seed-strikes] ERROR: $F not found" >&2; exit 1; }

python3 - "$F" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()

old = "        blocked_regions_strikes_.push_back(1);"
new = """        // SPARX (LOOP_BUGS.md B45): strike 1 is deliberately too weak to retire
        // a frontier -- correct for a runtime veto that might be a one-off,
        // wrong for a seed distilled from hundreds of flights. Configurable so
        // the default is byte-identical to the shipped behaviour.
        int sparx_seed_strikes = 1;
        ros::param::getCached("/frontier_finder/blocked_seed_strikes", sparx_seed_strikes);
        if (sparx_seed_strikes < 1) sparx_seed_strikes = 1;
        blocked_regions_strikes_.push_back(sparx_seed_strikes);"""

if "sparx_seed_strikes" in s:
    print("[sparx-seed-strikes] already applied")
else:
    assert s.count(old) == 1, "anchor found %d times" % s.count(old)
    s = s.replace(old, new, 1)
    io.open(p, "w", encoding="utf-8").write(s)
PY

grep -q "sparx_seed_strikes" "$F" || { echo "[sparx-seed-strikes] ERROR: marker missing" >&2; exit 1; }
echo "[sparx-seed-strikes] OK: seeded strike count is configurable (default 1 = shipped)"
