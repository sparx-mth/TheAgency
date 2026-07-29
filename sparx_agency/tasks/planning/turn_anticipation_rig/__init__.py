"""Offline flight rig for the drift-PID turn anticipation (yaw lookahead).

Flies the real controller against a modelled airframe on real routes, with the
anticipation off and on, and reports what actually changed. No ROS, no GPU, no
simulator. See ``README.md``.
"""
