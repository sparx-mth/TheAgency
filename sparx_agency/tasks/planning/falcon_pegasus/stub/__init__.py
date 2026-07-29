"""A stand-in aircraft for exercising the FALCON side without Isaac Sim.

Depth comes from raycasting the surveyed ground-truth voxel map and the airframe
is a first-order velocity lag, but the wire protocol, the handover order, the
outer-loop tracker and the exit conditions are the same code the real aircraft
flies. See :mod:`~.run_stub`.
"""
