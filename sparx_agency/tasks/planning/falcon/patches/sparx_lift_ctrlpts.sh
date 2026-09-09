#!/bin/bash
# ============================================================
# sparx_lift_ctrlpts.sh
#
# Lifts control points out of obstacles before the B-spline solve.
#
# WHY (LOOP_BUGS.md B44, confirmed from source). FALCON's ESDF is UNSIGNED:
# esdf.cpp seeds it `(occupancy == OCCUPIED) ? 0 : max`, with no interior field.
# Its gradient is a trilinear difference over eight neighbours, so inside an
# obstacle all eight read 0 and the gradient is exactly zero. The optimiser's
# clearance gradient is
#     gradient_q[i] += 2.0 * (dist - safe_distance_) * dist_grad;
# which is therefore also zero. A control point inside an obstacle carries a
# flat (0 - 0.55)^2 * 150 = 45.4 penalty and NO direction in which to reduce it.
# It is invisible to a gradient-based solver -- which is exactly what the flights
# show: the solver converges on xtol in under a millisecond (B40), still moves
# the free points 0.216 m median (B43), and leaves an obstructed point inside in
# 11 of 14 rejected trajectories (B42). 41% of all produced trajectories are then
# thrown away by the hard occupancy check (B37).
#
# THE FIX. Before optimising, relocate any control point whose ESDF distance is
# ~0 to the nearest non-occupied voxel, found by a bounded outward search of the
# occupancy grid (the ESDF cannot help: it is flat there, which is the whole
# problem). The optimiser then starts with every point where the gradient works,
# and the clearance term can keep them out.
#
# Bounded on purpose: the search gives up beyond ~1 m, and a point that cannot
# be lifted is left exactly as it was, so the worst case is today's behaviour.
# Controlled by /bspline_opt/lift_ctrlpts (default false = shipped behaviour)
# and /bspline_opt/lift_radius_m.
#
# Self-contained (no .patch file): edits the source in place and verifies.
# ============================================================
set -euo pipefail

F=/catkin_ws/src/FALCON/falcon_planner/trajectory/src/bspline/bspline_optimizer.cpp
[ -f "$F" ] || { echo "[sparx-lift] ERROR: $F not found" >&2; exit 1; }

python3 - "$F" <<'PY'
import io, sys
p = sys.argv[1]
s = io.open(p, encoding="utf-8").read()

anchor = "  std::vector<double> sparx_q0 = q;"
new = """  // SPARX (LOOP_BUGS.md B44): the ESDF is unsigned, so a control point inside
  // an obstacle sees distance 0 AND gradient 0 -- a flat penalty it cannot
  // descend. Lift such points to the nearest free voxel first, so the solver
  // starts where its gradient actually works. Bounded: a point that cannot be
  // lifted is left untouched, making the worst case the shipped behaviour.
  {
    bool sparx_lift = false;
    ros::param::getCached("/bspline_opt/lift_ctrlpts", sparx_lift);
    double sparx_lift_r = 1.0;
    ros::param::getCached("/bspline_opt/lift_radius_m", sparx_lift_r);
    if (sparx_lift && dim_ == 3) {
      const double res = map_server_->getResolution();
      int sparx_lifted = 0;
      for (int i = 0; i < point_num_; ++i) {
        Eigen::Vector3d cp(q[3 * i], q[3 * i + 1], q[3 * i + 2]);
        if (map_server_->getOccupancy(cp) != voxel_mapping::OccupancyType::OCCUPIED)
          continue;
        Eigen::Vector3d best = cp;
        double best_d = 1e9;
        for (double r = res; r <= sparx_lift_r + 1e-9; r += res) {
          const int steps = std::max(1, static_cast<int>(std::ceil(r / res)));
          for (int ax = -steps; ax <= steps; ++ax)
            for (int ay = -steps; ay <= steps; ++ay)
              for (int az = -steps; az <= steps; ++az) {
                Eigen::Vector3d cand = cp + Eigen::Vector3d(ax, ay, az) * res;
                const double d = (cand - cp).norm();
                if (d > r || d >= best_d) continue;
                if (!map_server_->isInBox(cand)) continue;
                if (map_server_->getOccupancy(cand) == voxel_mapping::OccupancyType::OCCUPIED)
                  continue;
                best = cand;
                best_d = d;
              }
          if (best_d < 1e8) break;
        }
        if (best_d < 1e8) {
          q[3 * i] = best.x();
          q[3 * i + 1] = best.y();
          q[3 * i + 2] = best.z();
          ++sparx_lifted;
        }
      }
      if (sparx_lifted)
        ROS_WARN_THROTTLE(1.0, "[sparx-lift] lifted %d/%d control point(s) out of obstacles",
                          sparx_lifted, point_num_);
    }
  }
""" + anchor

if "[sparx-lift] lifted" in s:
    print("[sparx-lift] already applied")
else:
    assert s.count(anchor) == 1, "anchor found %d times" % s.count(anchor)
    s = s.replace(anchor, new, 1)
    io.open(p, "w", encoding="utf-8").write(s)
PY

grep -q "\[sparx-lift\] lifted" "$F" || { echo "[sparx-lift] ERROR: marker missing" >&2; exit 1; }
echo "[sparx-lift] OK: control-point lifting applied (off unless /bspline_opt/lift_ctrlpts)"
