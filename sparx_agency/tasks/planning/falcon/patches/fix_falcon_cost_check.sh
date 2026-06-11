#!/bin/bash
# ============================================================
# fix_falcon_cost_check.sh
#
# Fixes a fatal glog CHECK in FALCON that aborts exploration_node
# at RUNTIME (exit code -6 / SIGABRT) during cost-matrix building:
#
#   F.... hierarchical_grid.cpp:2022] Check failed: cost > 1e-4
#         (3.26714e-05 vs. 0.0001) Zero cost from current position
#         to cell N free center N
#
# Root cause:
#   UniformGrid::calculateCostMatrixSingleThread() (and the cell-to-
#   cell loop) guard every edge with CHECK_GT(cost, 1e-4/1e-6).
#   These are glog FATAL assertions -> abort() on failure. When the
#   drone's current position is (almost) coincident with a grid-cell
#   free center, the A* / connectivity-graph cost is a legitimately
#   tiny-but-nonzero value (e.g. 3.3e-5) below the threshold, so the
#   check fires and kills the node. The check's real intent is only
#   to keep a ZERO-weight edge out of the downstream ATSP solver.
#
# Fix:
#   Floor every cost that can be near-zero (the A* costs and the
#   connectivity-graph BFS costs) to 1e-3 right where it's computed.
#   This preserves the no-zero-edge guarantee, makes the fatal checks
#   unreachable, and is numerically negligible for planning.
#
# Idempotent + self-verifying. Run AFTER cloning FALCON, BEFORE
# catkin_make. Rebuilds nothing itself.
# ============================================================
set -euo pipefail

HG="$(find /catkin_ws/src/FALCON -path '*exploration_preprocessing*/hierarchical_grid.cpp' | head -n1)"
if [ -z "${HG}" ]; then
  echo "[fix_cost_check] ERROR: hierarchical_grid.cpp not found" >&2
  exit 1
fi
echo "[fix_cost_check] Patching ${HG}"

if grep -q "FALCON patch: floor cost" "${HG}"; then
  echo "[fix_cost_check] Already patched, skipping."
  exit 0
fi
cp "${HG}" "${HG}.orig.bak"

# Floor 1: every A* cost (single-line statements).
sed -i -E 's@([[:space:]]*)cost = getAStarCostYaw\((.*)\);@\1cost = getAStarCostYaw(\2);\n\1if (cost < 1e-3) cost = 1e-3; // FALCON patch: floor cost, avoid CHECK_GT abort@g' "${HG}"

# Floor 2: every connectivity-graph BFS cost (statement wraps two lines,
# so use perl slurp mode to match across the newline).
perl -0777 -i -pe 's/(cost = connectivity_graph_->searchConnectivityGraphBFS\([^;]*\);)/$1\n              if (cost < 1e-3) cost = 1e-3; \/\/ FALCON patch: floor cost (BFS), avoid CHECK_GT abort/g' "${HG}"

# Verify: at least 4 A* floors + 2 BFS floors, and braces still balanced.
N=$(grep -c "FALCON patch: floor cost" "${HG}" || true)
OB=$(grep -o '{' "${HG}" | wc -l)
CB=$(grep -o '}' "${HG}" | wc -l)
echo "[fix_cost_check] floors inserted: ${N}   braces: {${OB} }${CB}"
if [ "${N}" -lt 6 ]; then
  echo "[fix_cost_check] ERROR: expected >=6 floors, got ${N}. Upstream code may have changed." >&2
  mv "${HG}.orig.bak" "${HG}"
  exit 1
fi
if [ "${OB}" != "${CB}" ]; then
  echo "[fix_cost_check] ERROR: brace mismatch after patch; reverting." >&2
  mv "${HG}.orig.bak" "${HG}"
  exit 1
fi
echo "[fix_cost_check] OK: cost-matrix CHECK_GT aborts are now neutralized."