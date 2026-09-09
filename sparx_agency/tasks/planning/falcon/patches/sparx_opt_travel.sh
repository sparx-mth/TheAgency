#!/bin/bash
# ============================================================
# sparx_opt_travel.sh
#
# Reports how far the optimiser actually MOVES the control points.
#
# WHY. B42: 11 of 14 rejected trajectories had a control point inside an
# obstacle, where the clearance term contributes (0-0.55)^2 * 150 = 45 against
# total costs of 33-668 -- a large, clearly-signalled penalty the optimiser
# accepted rather than fixing. B40: every solve terminates on xtol_rel=1e-4 in
# under a millisecond.
#
# That suggests premature convergence -- but "under a millisecond" is not the
# same as "barely iterated", and the distinction decides the fix:
#
#   * moved ~nothing  -> it converged before doing useful work. The lever is
#     xtol_rel (or the initial guess, which it is simply accepting).
#   * moved a lot and still ended inside -> a genuine local minimum or a real
#     trade against the other cost terms. The lever is the cost landscape, and
#     tightening tolerances would only burn CPU.
#
# So log the maximum and mean displacement between the control points handed to
# NLopt and the ones it returns, beside the result code already captured.
#
# Diagnostic only: two vector norms per solve on a path that already logs.
# No control flow changed.
#
# Self-contained (no .patch file): edits the source in place and verifies.
# ============================================================
set -euo pipefail

F=/catkin_ws/src/FALCON/falcon_planner/trajectory/src/bspline/bspline_optimizer.cpp
[ -f "$F" ] || { echo "[sparx-travel] ERROR: $F not found" >&2; exit 1; }

python3 - "$F" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()

# snapshot the variables handed to NLopt, just before the solve
old = "  auto t1 = ros::Time::now();"
new = ("  // SPARX: snapshot the initial variables so the solve's travel can be\n"
       "  // measured (LOOP_BUGS.md B42): converging without moving means the\n"
       "  // tolerance is the lever, moving and still colliding means the cost\n"
       "  // landscape is.\n"
       "  std::vector<double> sparx_q0 = q;\n"
       "  auto t1 = ros::Time::now();")
if "sparx_q0" not in s:
    assert s.count(old) == 1, "t1 anchor found %d times" % s.count(old)
    s = s.replace(old, new, 1)

old2 = """        "[sparx-opt] result=%d budget=%.3fs elapsed=%.3fs cost=%.3f pts=%d","""
new2 = """        "[sparx-opt] result=%d budget=%.3fs elapsed=%.3fs cost=%.3f pts=%d "
        "travel_max=%.4f travel_mean=%.4f","""
old3 = """        static_cast<int>(result), sparx_maxtime,
        (ros::Time::now() - t1).toSec(), final_cost, point_num_);"""
new3 = """        static_cast<int>(result), sparx_maxtime,
        (ros::Time::now() - t1).toSec(), final_cost, point_num_,
        sparx_travel_max, sparx_travel_mean);"""
decl = """    double final_cost;
    nlopt::result result = opt.optimize(q, final_cost);"""
decl_new = """    double final_cost;
    nlopt::result result = opt.optimize(q, final_cost);
    double sparx_travel_max = 0.0, sparx_travel_mean = 0.0;
    {
      const size_t nvar = std::min(sparx_q0.size(), best_variable_.size());
      for (size_t vi = 0; vi < nvar; ++vi) {
        const double dv = std::fabs(best_variable_[vi] - sparx_q0[vi]);
        sparx_travel_mean += dv;
        if (dv > sparx_travel_max) sparx_travel_max = dv;
      }
      if (nvar) sparx_travel_mean /= static_cast<double>(nvar);
    }"""

if "travel_max=" in s:
    print("[sparx-travel] already applied")
else:
    assert s.count(decl) == 1, "decl anchor %d" % s.count(decl)
    s = s.replace(decl, decl_new, 1)
    assert s.count(old2) == 1, "fmt anchor %d" % s.count(old2)
    s = s.replace(old2, new2, 1)
    assert s.count(old3) == 1, "args anchor %d" % s.count(old3)
    s = s.replace(old3, new3, 1)
    io.open(p, "w", encoding="utf-8").write(s)
PY

grep -q "travel_max=" "$F" || { echo "[sparx-travel] ERROR: marker missing" >&2; exit 1; }
grep -q "sparx_q0" "$F" || { echo "[sparx-travel] ERROR: snapshot missing" >&2; exit 1; }
echo "[sparx-travel] OK: control-point travel logged with the result code"
