#!/bin/bash
# ============================================================
# sparx_opt_result.sh
#
# Makes the B-spline optimiser say how it terminated, and lets its time budget
# be changed without restarting the node.
#
# WHY. 41% of everything the planner produces is thrown away by a HARD occupancy
# check (LOOP_BUGS.md B38), while the optimiser treats clearance as a SOFT
# quadratic penalty and is handed opt.set_maxtime(0.01) -- ten milliseconds for
# an L-BFGS over 11-21 control points with six cost terms. The obvious question
# is whether it is running out of time before it can satisfy the distance cost.
#
# NLopt already answers that: opt.optimize() returns a result code, and
# NLOPT_MAXTIME_REACHED means exactly "stopped on the clock". FALCON assigns it
# to a local and never reads it, and its cost reporting sits behind `if (false)`.
# So the question is directly observable and nobody was looking. Logging it
# settles the hypothesis from ONE flight, where inferring it behaviourally needs
# ~90 flights per arm: the rejection count has a CV of 44% because rejections
# are a property of WHERE the aircraft goes, and flights differ in that.
#
# TWO changes, both small:
#   1. Log the result code (throttled) with the solve time and the terminating
#      cost, so a flight reports how often the solver stopped on maxtime.
#   2. Re-read /bspline_opt/max_iteration_time per solve instead of once in the
#      constructor, so the budget can be alternated WITHIN a flight. That is
#      what LOOP_BUGS.md B39's paired design needs, and the same pattern applies
#      to every other C++ constructor-read parameter in this planner.
#
# The per-solve read is a cached rosparam lookup, not an XML-RPC round trip per
# solve: ros::param::getCached costs a local map hit after the first call.
#
# Self-contained (no .patch file): edits the source in place and verifies.
# ============================================================
set -euo pipefail

F=/catkin_ws/src/FALCON/falcon_planner/trajectory/src/bspline/bspline_optimizer.cpp
[ -f "$F" ] || { echo "[sparx-opt-result] ERROR: $F not found" >&2; exit 1; }

# --- 1. per-solve budget re-read, replacing the constructor-cached member -----
python3 - "$F" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()

old = "  opt.set_maxtime(max_iteration_time_);"
new = ("  // SPARX: re-read per solve so the budget can be changed in flight\n"
       "  // (LOOP_BUGS.md B39 paired design). getCached is a local map hit\n"
       "  // after the first call, not an XML-RPC round trip per solve.\n"
       "  double sparx_maxtime = max_iteration_time_;\n"
       "  ros::param::getCached(\"/bspline_opt/max_iteration_time\", sparx_maxtime);\n"
       "  opt.set_maxtime(sparx_maxtime);")
if new.split("\n")[0] in s:
    print("[sparx-opt-result] per-solve budget already applied")
else:
    assert s.count(old) == 1, "maxtime anchor found %d times" % s.count(old)
    s = s.replace(old, new, 1)

# --- 2. log how NLopt terminated ---------------------------------------------
old2 = """  try {
    double final_cost;
    nlopt::result result = opt.optimize(q, final_cost);
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
  }"""
new2 = """  try {
    double final_cost;
    nlopt::result result = opt.optimize(q, final_cost);
    // SPARX: FALCON discarded this code. NLOPT_MAXTIME_REACHED (6) means the
    // solver stopped on the clock rather than converging, which is the whole
    // question behind the 41% pre-publish rejection rate (LOOP_BUGS.md B38).
    ROS_WARN_THROTTLE(2.0,
        "[sparx-opt] result=%d budget=%.3fs elapsed=%.3fs cost=%.3f pts=%d",
        static_cast<int>(result), sparx_maxtime,
        (ros::Time::now() - t1).toSec(), final_cost, point_num_);
  } catch (std::exception &e) {
    std::cout << e.what() << std::endl;
  }"""
if "[sparx-opt] result=" in s:
    print("[sparx-opt-result] result logging already applied")
else:
    assert s.count(old2) == 1, "optimize anchor found %d times" % s.count(old2)
    s = s.replace(old2, new2, 1)

io.open(p, "w", encoding="utf-8").write(s)
PY

grep -q "sparx_maxtime" "$F" || { echo "[sparx-opt-result] ERROR: budget re-read missing" >&2; exit 1; }
grep -q "\[sparx-opt\] result=" "$F" || { echo "[sparx-opt-result] ERROR: result log missing" >&2; exit 1; }
echo "[sparx-opt-result] OK: per-solve budget + NLopt result logging applied"
