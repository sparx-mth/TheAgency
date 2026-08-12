#!/bin/bash
# Let FALCON run on the simulator's clock instead of the wall clock.
#
# `exploration_node.cpp` opens with
#
#     nh.param("/use_sim_time", use_sim_time, false);
#     CHECK(!use_sim_time) << "Please set use_sim_time to false";
#
# and glog's CHECK is fatal, so a true value does not degrade this stack, it
# aborts the node holding the mapper, the frontier finder and the FSM.
#
# WHY REMOVE IT. Upstream's simulator (`poscmd_2_odom`) feeds the position
# command straight back as the aircraft's state and runs on the wall clock, so
# for them simulated time was pointless and the check documents an assumption.
# Here the aircraft is Isaac Sim + PhysX + PX4, and it manages about **0.63 of
# real time** on this machine. With FALCON on the wall clock the two disagree
# about how fast a second is: FALCON issues a schedule that advances ~1.6 s per
# second of flight the aircraft is actually given, the tracker carries a
# standing along-track lag it can never close, and -- because FALCON starts each
# new curve from its own previous one rather than from the measured position --
# every replan begins a little further ahead than the last. Measured on the
# stub, changing nothing but the clock rate: mean lag -0.08 m at real time
# against +0.69 m at 0.66x, with cross-track tripling.
#
# Compensating for that downstream was tried (`link/sim_clock.py` re-bases each
# trajectory's start onto the aircraft's clock) and recovers about a fifth of
# it. The rest is structural and only goes away when FALCON's own planning
# cadence slows to match, which is what `/use_sim_time` does.
#
# WHAT MAKES IT SAFE. Everything FALCON times -- trajectory starts, FSM timers,
# frontier ages, the depth/pose pairing -- is relative, and all of it is fed
# from one monotonic clock published by `pegasus_bridge_node` from the
# aircraft's own timestamps. Nothing in the planner needs time to be real, only
# consistent. The bridge's pre-connection waits were moved to the wall clock so
# the node cannot deadlock waiting for a clock that has not started yet.
#
# The check is replaced rather than deleted, so an operator who sets the
# parameter by accident still gets told what it means.
set -e

NODE=/catkin_ws/src/FALCON/falcon_planner/exploration_manager/src/exploration_node.cpp
[ -f "$NODE" ] || { echo "exploration_node.cpp not found at $NODE"; exit 1; }

python3 - "$NODE" <<'PYEOF'
import sys

path = sys.argv[1]
source = open(path).read()

check = '  CHECK(!use_sim_time) << "Please set use_sim_time to false";'
replacement = (
    '  // Upstream aborts here on a true value. Allowed deliberately: this\n'
    '  // deployment flies a simulator that does NOT run at real time, and\n'
    '  // pegasus_bridge_node publishes /clock from the aircraft so that FALCON\n'
    '  // plans at the rate the aircraft can actually fly. See\n'
    '  // patches/allow_sim_time.sh.\n'
    '  if (use_sim_time)\n'
    '    ROS_WARN("[exploration_node] running on /clock, not the wall clock -- '
    'see falcon_pegasus/patches/allow_sim_time.sh");'
)

if replacement.splitlines()[0].strip() in source:
    print("patch: already applied, nothing to do")
    raise SystemExit(0)

if source.count(check) != 1:
    raise SystemExit("patch: expected exactly 1 use_sim_time CHECK, found %d"
                     % source.count(check))

open(path, "w").write(source.replace(check, replacement))
print("patch: exploration_node may now run on simulated time")
PYEOF
