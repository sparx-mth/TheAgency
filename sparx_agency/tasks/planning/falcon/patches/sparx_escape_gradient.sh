#!/bin/bash
# ============================================================
# sparx_escape_gradient.sh
#
# Gives the clearance cost a usable gradient INSIDE obstacles.
#
# WHY (LOOP_BUGS.md B44, and F28's own result). FALCON's ESDF is unsigned --
# esdf.cpp seeds `(occupancy == OCCUPIED) ? 0 : max`, with no interior field --
# so a control point inside an obstacle reads distance 0 and gradient 0. The
# clearance cost then does this:
#
#     if (dist_grad.norm() > 1e-4) dist_grad.normalize();     // stays (0,0,0)
#     if (dist < safe_distance_) {
#       cost += pow(dist - safe_distance_, 2);                // +45.4
#       gradient_q[i] += 2.0 * (dist - safe_distance_) * dist_grad;   // += 0
#     }
#
# The penalty is added and the gradient contribution is exactly zero: a large,
# flat cost the solver cannot descend. Measured consequences: 11 of 14 rejected
# trajectories keep a control point inside (B42), and 41% of all produced plans
# are thrown away by the hard occupancy check (B37).
#
# Lifting such points out BEFORE the solve was tried (F28) and does not hold:
# 66% of rejected trajectories had a point back inside afterwards, because
# L-BFGS steps are large (0.216 m median, 1.27 m max -- B43) and one step can
# carry a point from outside the 0.55 m gradient band straight into the flat
# interior. The gradient has to work INSIDE, not just at the start.
#
# THE FIX. When the ESDF gradient vanishes and the point is in violation, build
# an escape direction from the occupancy grid itself: sample the six axis
# neighbours and sum the directions that are not occupied. That is the direction
# out of the obstacle, and it plugs into the existing formula unchanged --
# gradient descent then moves the point toward free space.
#
# Costs six occupancy lookups, and only for points that are actually inside an
# obstacle. Controlled by /bspline_opt/escape_gradient (default false =
# shipped behaviour).
#
# Self-contained (no .patch file): edits the source in place and verifies.
# ============================================================
set -euo pipefail

F=/catkin_ws/src/FALCON/falcon_planner/trajectory/src/bspline/bspline_optimizer.cpp
[ -f "$F" ] || { echo "[sparx-escape] ERROR: $F not found" >&2; exit 1; }

python3 - "$F" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()

old = """    map_server_->getESDF()->getDistanceAndGradient(q[i], dist, dist_grad);
    if (dist_grad.norm() > 1e-4)
      dist_grad.normalize();
"""
new = """    map_server_->getESDF()->getDistanceAndGradient(q[i], dist, dist_grad);
    if (dist_grad.norm() > 1e-4)
      dist_grad.normalize();
    else if (dist < safe_distance_) {
      // SPARX (LOOP_BUGS.md B44): the ESDF is unsigned, so a point inside an
      // obstacle reads distance 0 AND gradient 0 -- the penalty below is added
      // and its gradient contribution is exactly zero, leaving a large flat
      // cost the solver cannot descend. Build the escape direction from the
      // occupancy grid instead: the sum of the axis directions that are free
      // points out of the obstacle. Six lookups, only for trapped points.
      bool sparx_escape = false;
      ros::param::getCached("/bspline_opt/escape_gradient", sparx_escape);
      if (sparx_escape) {
        const double res = map_server_->getResolution();
        Eigen::Vector3d esc(0, 0, 0);
        for (int ax = 0; ax < 3; ++ax)
          for (int sg = -1; sg <= 1; sg += 2) {
            Eigen::Vector3d step(0, 0, 0);
            step[ax] = sg * res;
            const Eigen::Vector3d probe = q[i] + step;
            if (!map_server_->isInBox(probe)) continue;
            if (map_server_->getOccupancy(probe) != voxel_mapping::OccupancyType::OCCUPIED)
              esc += step;
          }
        if (esc.norm() > 1e-6) {
          // Gain < 1 nudges a trapped point outward over several iterations
          // instead of displacing it in one. At full strength (F29) the push is
          // blocky and axis-aligned, and it converted rejected trajectories into
          // FAILED plans: rejections fell 0.67x but plan_fail rose 10.7x and
          // coverage fell to 0.56x. Let smoothness and feasibility keep up.
          double sparx_escape_gain = 1.0;
          ros::param::getCached("/bspline_opt/escape_gain", sparx_escape_gain);
          dist_grad = esc.normalized() * sparx_escape_gain;
          ++sparx_escape_hits_;
        }
      }
    }
"""
if "sparx_escape_hits_" in s:
    print("[sparx-escape] already applied")
else:
    assert s.count(old) == 1, "cost anchor found %d times" % s.count(old)
    s = s.replace(old, new, 1)
    # counter definition, reported with the existing result line
    old2 = "  std::vector<double> sparx_q0 = q;"
    new2 = "  sparx_escape_hits_ = 0;\n  std::vector<double> sparx_q0 = q;"
    assert s.count(old2) == 1, "snapshot anchor %d" % s.count(old2)
    s = s.replace(old2, new2, 1)
    old3 = '"travel_max=%.4f travel_mean=%.4f",'
    new3 = '"travel_max=%.4f travel_mean=%.4f escape_hits=%d",'
    assert s.count(old3) == 1, "fmt anchor %d" % s.count(old3)
    s = s.replace(old3, new3, 1)
    old4 = "        sparx_travel_max, sparx_travel_mean);"
    new4 = "        sparx_travel_max, sparx_travel_mean, sparx_escape_hits_);"
    assert s.count(old4) == 1, "args anchor %d" % s.count(old4)
    s = s.replace(old4, new4, 1)
    io.open(p, "w", encoding="utf-8").write(s)
PY

H=/catkin_ws/src/FALCON/falcon_planner/trajectory/include/bspline/bspline_optimizer.h
grep -q "sparx_escape_hits_" "$H" || python3 - "$H" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()
anchor = "public:"
assert s.count(anchor) >= 1
s = s.replace(anchor, "public:\n  // SPARX: escape-gradient substitutions in the last solve (LOOP_BUGS.md B44).\n  int sparx_escape_hits_ = 0;\n", 1)
io.open(p, "w", encoding="utf-8").write(s)
PY

grep -q "sparx_escape_hits_" "$F" || { echo "[sparx-escape] ERROR: cost marker missing" >&2; exit 1; }
grep -q "sparx_escape_hits_" "$H" || { echo "[sparx-escape] ERROR: member missing" >&2; exit 1; }
echo "[sparx-escape] OK: escape gradient applied (off unless /bspline_opt/escape_gradient)"
