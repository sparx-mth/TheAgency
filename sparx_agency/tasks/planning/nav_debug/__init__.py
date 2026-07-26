"""Offline (and, later, live) visual debugging for the FALCON navigation stack.

Loads a recorded run -- the per-tick certainty CSV plus the BEV maps, the three
route layers (raw A* -> corrected -> final flown) and the replan events written
by :mod:`nav_debug_recorder_node` -- and renders, for any moment, one debug
screen that answers "what did A* plan, what did the drone want to do, and why":

  * the BEV map with the raw / corrected / final routes, the target waypoint, the
    drone pose + a localization trail and the drift vector, plus a banner naming
    the active replan reason (time / rotation / blocking) and localization state;
  * a telemetry panel with TWO ROLL/PITCH/YAW gauge stacks -- the command WE send
    (``cmd_vel``) and the command the converter sends the DRONE (``cmd_nav`` axis
    counts) -- confidence, the "why", and short command/confidence history strips.

ROS-free and drone-agnostic: the offline player runs on the dev PC and the same
renderer can be driven live by a viewer node inside the Noetic container. Kept
Python 3.8 compatible for that reason.
"""
