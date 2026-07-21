"""Shared, ROS-free HUD drawing primitives for the planning debug views.

The object-approach target-lock HUD (:mod:`sparx_agency.tasks.planning.object_approach_offline.overlay`)
and the navigation debug view (:mod:`sparx_agency.tasks.planning.nav_debug.render`)
both draw the same vocabulary -- ROLL/PITCH arrows, a YAW dial, boxed text tags,
a colour palette -- so those primitives live here once instead of being copied
into each. Nothing in this package imports ROS or any mission logic; it is
``numpy`` + ``cv2`` arithmetic only, so it is testable headless and reusable by a
live viewer node inside the FALCON Noetic container.

Python 3.8 compatible (the FALCON ROS1/Noetic adapter renders the live HUD under
3.8): no 3.10+ syntax, annotations deferred via ``from __future__``.
"""
