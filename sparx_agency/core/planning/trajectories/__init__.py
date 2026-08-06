"""Time-parameterised trajectory representations.

A *path* is a sequence of points (see ``core.common.types.Path2D``); a
*trajectory* is a curve that also says *when* the vehicle should be at each
point, and therefore carries velocity, acceleration and jerk as exact
derivatives rather than as finite differences.

This package holds the curve forms themselves. The controllers that follow them
live in ``core.control``; the planners that produce them live in
``core.planning.planners``.
"""
