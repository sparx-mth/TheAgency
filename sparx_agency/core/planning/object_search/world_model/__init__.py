"""The method's world model: per-storey observed occupancy, where the camera has looked, and the stair surfaces it has seen.

Mapping algorithms that live beside the planner rather than in ``core/mapping``
because their products are the planners' vocabulary (``OccupancyGrid2D`` from
``core/planning/environment``) and their inputs the contract's camera geometry;
``core/mapping`` imports nothing from ``core/planning``, and the line stays that way.
"""
