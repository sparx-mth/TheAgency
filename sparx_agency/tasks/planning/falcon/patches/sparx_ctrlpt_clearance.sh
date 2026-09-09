#!/bin/bash
# ============================================================
# sparx_ctrlpt_clearance.sh
#
# Reports the clearance of the CONTROL POINTS beside the clearance of the CURVE,
# at the moment a trajectory is rejected.
#
# WHY. B41 established that the ESDF and the occupancy grid agree completely at
# every rejection (p50 0.007 m, max 0.096 m, against safe_distance = 0.55), so
# the 41% pre-publish rejection rate is not a map problem. B40 established the
# optimiser converges every time in under a millisecond. So it converges to
# curves that pass through obstacles its own distance field can see -- and there
# are two ways that happens, which call for OPPOSITE fixes:
#
#   * the optimiser penalises CONTROL POINTS, getDistanceAndGradient(q[i], ...);
#     the checker samples the CURVE. A cubic B-spline lies inside its control
#     hull, so an obstacle protruding into that hull can be nearer the curve
#     than any control point. Every control point can satisfy 0.55 m while the
#     curve clips a corner. Then raising the cost WEIGHT does nothing, because
#     the term being weighted is already satisfied, and the fix is to penalise
#     sampled curve points or widen the margin to cover the hull gap.
#
#   * or the control points are themselves inside, the soft penalty genuinely
#     lost the trade against smoothness/endpoint/feasibility, and the weight
#     (currently 150) is the lever.
#
# One number separates them: the minimum ESDF distance over the rejected
# trajectory's control points. Logged next to the curve value already captured.
#
# Diagnostic only: a loop over control points on a path that already logs and
# returns false. No control flow changed.
#
# Self-contained (no .patch file): edits the source in place and verifies.
# ============================================================
set -euo pipefail

F=/catkin_ws/src/FALCON/falcon_planner/fast_planner/src/planner_manager.cpp
[ -f "$F" ] || { echo "[sparx-ctrlpt] ERROR: $F not found" >&2; exit 1; }

python3 - "$F" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()

old = """      ROS_WARN_THROTTLE(1.0, "[sparx-esdf] reject at (%.2f, %.2f, %.2f) esdf_dist=%.3f t=%.2f",
                        fut_pt.x(), fut_pt.y(), fut_pt.z(), sparx_d, t_now + fut_t);"""
new = """      // SPARX: the same question for the CONTROL POINTS the optimiser actually
      // penalised. If these are clear while the curve is not, the curve is
      // clipping inside the control hull and the cost WEIGHT is irrelevant.
      double sparx_qmin = 1e9;
      Eigen::MatrixXd sparx_q = local_data_.position_traj_.getControlPoint();
      for (int qi = 0; qi < sparx_q.rows(); ++qi) {
        Eigen::Vector3d sparx_qp = sparx_q.row(qi).head(3);
        double sparx_qd = -1.0;
        Eigen::Vector3d sparx_qg;
        map_server_->getESDF()->getDistanceAndGradient(sparx_qp, sparx_qd, sparx_qg);
        if (sparx_qd < sparx_qmin) sparx_qmin = sparx_qd;
      }
      ROS_WARN_THROTTLE(1.0,
          "[sparx-esdf] reject at (%.2f, %.2f, %.2f) esdf_dist=%.3f ctrlpt_min=%.3f npts=%d t=%.2f",
          fut_pt.x(), fut_pt.y(), fut_pt.z(), sparx_d, sparx_qmin,
          static_cast<int>(sparx_q.rows()), t_now + fut_t);"""

if "ctrlpt_min=" in s:
    print("[sparx-ctrlpt] already applied")
else:
    assert s.count(old) == 1, "anchor found %d times" % s.count(old)
    s = s.replace(old, new, 1)
    io.open(p, "w", encoding="utf-8").write(s)
PY

grep -q "ctrlpt_min=" "$F" || { echo "[sparx-ctrlpt] ERROR: marker missing" >&2; exit 1; }
echo "[sparx-ctrlpt] OK: control-point clearance logged beside the curve value"
