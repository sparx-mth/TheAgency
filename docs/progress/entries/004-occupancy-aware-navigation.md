# 004 - Smooth, occupancy-aware navigation to the clicked point

**Branch:** `feat/occupancy-aware-nav`
**Status:** planning
**Roadmap item:** "BEV click-to-fly: smooth, occupancy-aware navigation to the clicked point"

## Goal
Clicking a point on the BEV produces a flight path that gets there smoothly and respects the
occupancy map along the way — not a naive straight line, and not a jerky/stop-start path.
"Done" = clicking a point across the mapped room, with a real obstacle in between, produces
a single smooth flight that avoids it, confirmed live.

## Why
The click-to-fly pipeline (`astar_planner_node.py` → `combination_planner_node.py`/NavDP →
`path_corrector_node.py` → `trajectory_simplifier_node.py` → `waypoint_follower_node.py`) is
wired end-to-end already, but today's session never actually tested it against real occupied
geometry with the corrected (post X/Y-fix) map — everything today was raw manual flight,
bypassing this pipeline entirely. This is the first real test of the actual click-to-fly
feature since the localization mirroring fix.

## Steps
- [ ] With a clean, correctly-oriented map (today's fix), click a goal point and confirm
      `astar_planner_node` finds a route using the real occupancy grid, not a stale/mirrored
      one
- [ ] Verify `combination_planner_node`'s NavDP fallback-to-A* behavior (seen today as benign
      "NavDP unreachable" warnings when NavDP's own service isn't running) doesn't produce a
      worse path than pure A* would
- [ ] Check `path_corrector_node`/`trajectory_simplifier_node` output is actually smooth
      (no unnecessary zig-zag) against the corrected map bounds
- [ ] Click a goal with a real obstacle between the drone and the target; confirm the route
      bends around it rather than straight-lining through it
- [ ] Re-verify the rotation-freeze/demo-mode fixes from today don't interfere once
      `waypoint_follower` is actually driving real motion again (today's fixes mostly killed
      `waypoint_follower` outright for manual testing — this is the first time it needs to
      run for real since)

## Open questions
- Is NavDP's own service expected to be running for this test, or is A*-only the intended
  baseline for now?

## Notes

## Result
