#!/bin/bash
# Make FALCON replan from where the aircraft IS, not where it assumed it would be.
#
# THE BUG. `ExplorationFSM::FSMCallback`, the non-static replan branch
# (exploration_fsm.cpp, "Replan from non-static state"), picks the start state
# for the next trajectory like this:
#
#     LocalTrajData *info = &planner_manager_->local_data_;
#     double t_r = (time_now - info->start_time_).toSec() + fp_->replan_duration_;
#     fd_->start_pos_ = info->position_traj_.evaluateDeBoorT(t_r);
#     fd_->start_vel_ = info->velocity_traj_.evaluateDeBoorT(t_r);
#     fd_->start_acc_ = info->acceleration_traj_.evaluateDeBoorT(t_r);
#
# Every one of those is read off the PREVIOUS TRAJECTORY. `fd_->odom_pos_` --
# the aircraft's measured position, which the FSM already has and keeps up to
# date -- appears nowhere in this branch. It is used only in the `static_state_`
# (hover) branch above it.
#
# Upstream this is not a bug, it is an identity. Their simulator is
# `poscmd_2_odom`, which feeds the position command straight back as the
# vehicle state, so the point on the trajectory IS the drone's position, exactly
# and forever. Tracking error does not exist there.
#
# Put a real airframe underneath and the two come apart, and the consequence is
# structural rather than gradual:
#
#   * the new curve starts at a point the aircraft is not at;
#   * the aircraft must chase that point, so it is behind before it begins;
#   * the tracking error is INVISIBLE to the planner -- nothing in the replan
#     reads the aircraft's position, so nothing can correct it;
#   * the next replan starts from the new plan's prediction, which inherits the
#     error, so it accumulates rather than decays.
#
# Observed exactly as described: the aircraft cannot keep up with the point
# FALCON believes it occupies, and the gap grows over a flight rather than
# closing. Measured at up to 9.5 m, sustained.
#
# THE FIX, and why it is conditional. Always planning from measured odometry
# would throw away the reason the prediction exists: a curve that starts at the
# aircraft's CURRENT position ignores the `replan_duration_` it takes to compute,
# so by the time it is published the aircraft has already left the start, and
# consecutive curves no longer join smoothly. That is a real cost and it is why
# upstream predicts.
#
# So: predict while the prediction is TRUE, and fall back to measurement once it
# demonstrably is not. There are two ways it stops being true, and the second is
# the one the operator kept hitting:
#
#   1. THE PLAN RAN AHEAD. The predicted start is further than
#      `~replan_start_tolerance` from the aircraft -- it fell behind, the plan
#      marched on, and the smooth join is now worth less than starting where the
#      aircraft actually is.
#   2. THE AIRCRAFT STOPPED. It is essentially motionless -- held after an
#      unsafe trajectory, braked in an emergency, or wedged -- while the plan
#      still expects it moving. Here the position gap may not have grown yet, but
#      planning a fresh curve from a future point a stopped aircraft will never
#      reach is precisely the deadlock to avoid. Detected as measured speed below
#      `~replan_stopped_speed` while the predicted start still carries velocity,
#      and answered the same way: re-anchor to where the aircraft is, now.
#
# In either case velocity comes from odometry and acceleration is zeroed -- the
# same treatment the hover branch already gives, since a measured acceleration is
# not available and a predicted one would belong to the trajectory being
# abandoned.
#
# Both are ROS parameters defaulting to the inert value (1e9 and -1), so an
# unpatched-looking image behaves EXACTLY as upstream until `nav` sets them. That
# keeps a baseline run available for comparison and means the thresholds can be
# tuned from the launch file without another rebuild.
set -e

FSM=/catkin_ws/src/FALCON/falcon_planner/exploration_manager/src/exploration_fsm.cpp
[ -f "$FSM" ] || { echo "exploration_fsm.cpp not found at $FSM"; exit 1; }

python3 - "$FSM" <<'PYEOF'
import sys

path = sys.argv[1]
source = open(path).read()

anchor = """      fd_->start_yaw_(2) = info->yawdotdot_traj_.evaluateDeBoorT(t_r)[2 - 2];
"""
if anchor not in source:
    anchor = """      fd_->start_yaw_(2) = info->yawdotdot_traj_.evaluateDeBoorT(t_r)[0];
"""

addition = anchor + """
      // ---- replan_from_measured_state.sh -------------------------------
      // The block above reads the start state off the PREVIOUS TRAJECTORY.
      // That is exact for upstream's poscmd_2_odom simulator, where the
      // command is fed back as the state, and wrong for any real airframe:
      // the aircraft is not where the old plan predicted, so the new curve
      // starts ahead of it and the error is never corrected because nothing
      // here looks at odom_pos_.
      //
      // Once the prediction is further away than the tolerance it has stopped
      // describing the aircraft, and planning from the measured state is worth
      // more than a smooth join to a curve the aircraft was not flying.
      double replan_start_tolerance = 1e9;
      double replan_stopped_speed = -1.0;
      ros::NodeHandle nh_tol;
      nh_tol.param("/exploration_node/replan_start_tolerance",
                   replan_start_tolerance, 1e9);
      nh_tol.param("/exploration_node/replan_stopped_speed",
                   replan_stopped_speed, -1.0);
      const double predicted_gap = (fd_->start_pos_ - fd_->odom_pos_).norm();
      const double measured_speed = fd_->odom_vel_.norm();
      const double predicted_speed = fd_->start_vel_.norm();
      // Two ways the prediction has stopped describing the aircraft: it ran
      // ahead (gap too big), or the aircraft STOPPED -- held after an unsafe
      // trajectory, braked, or wedged -- while the plan still expects motion.
      // The second is the deadlock the operator kept hitting: a fresh curve from
      // a future point a stopped aircraft will never reach. Re-anchor for either.
      const bool prediction_far = predicted_gap > replan_start_tolerance;
      const bool stopped_but_planned_moving =
          replan_stopped_speed >= 0.0 &&
          measured_speed < replan_stopped_speed &&
          predicted_speed > 2.0 * replan_stopped_speed;
      if (prediction_far || stopped_but_planned_moving) {
        ROS_WARN_THROTTLE(2.0,
                          "[FSM] replanning from the measured state: gap %.2f m "
                          "(tol %.2f), measured %.2f m/s, planned %.2f m/s",
                          predicted_gap, replan_start_tolerance, measured_speed,
                          predicted_speed);
        fd_->start_pos_ = fd_->odom_pos_;
        fd_->start_vel_ = fd_->odom_vel_;
        fd_->start_acc_.setZero();
        fd_->start_yaw_ << fd_->odom_yaw_, 0, 0;
      }
      // ------------------------------------------------------------------
"""

if "replan_from_measured_state.sh" in source:
    print("patch: already applied, nothing to do")
    raise SystemExit(0)

if source.count(anchor) != 1:
    raise SystemExit("patch: expected exactly 1 replan start-yaw block, found %d"
                     % source.count(anchor))

open(path, "w").write(source.replace(anchor, addition))
print("patch: FALCON will replan from the measured state when it strays far OR "
      "stops while the plan expects motion")
PYEOF
