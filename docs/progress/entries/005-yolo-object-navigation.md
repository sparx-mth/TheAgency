# 005 - Navigate to a YOLO-detected labeled object (e.g. "barrel")

**Branch:** `feat/yolo-object-navigation`
**Status:** planning
**Roadmap item:** "YOLO-detected object label + position (\"barrel\") as a FALCON navigation goal"

## Goal
Given a target label (e.g. "barrel"), detect it via YOLO in the live camera feed, compute its
real-world position, and drive FALCON to navigate there using the occupancy map for a safe
route — same click-to-fly pipeline as 004, but the goal comes from object detection instead
of a BEV click.

## Why
This is the natural next step after occupancy-aware click-to-fly (004): instead of a human
clicking a point, the system identifies *what* it's looking at and navigates to it.

## Steps
- [ ] Locate the existing YOLO integration (`yolo_world_trt`, referenced in the shared
      GPU-detection module work) and confirm it runs against the Rooster camera feed, not just
      XTEND's
- [ ] Get a 2D detection (label + bounding box) reliably for a test object ("barrel" or
      whatever's actually placed in the Sphera scene — check what's available before assuming
      a barrel exists in `sphera_jail`)
- [ ] Back-project the detection into a real-world 3D/2D position using the depth map +
      camera intrinsics + current pose (same math class as the existing depth→pointcloud step
      in the mapping pipeline — reuse rather than reinvent)
- [ ] Feed that position into the same goal-setting path the BEV click uses
      (`astar_planner_node`'s goal topic), so it inherits 004's occupancy-aware routing for
      free
- [ ] Handle the "no detection" / "label not found" case explicitly — don't let it silently
      do nothing or fly toward a stale/last-known position without saying so
- [ ] Live test: place/confirm a labeled object in the scene, trigger detection, confirm the
      drone navigates to it and the reported stop position is actually next to the real object

## Open questions
- Is a "barrel" actually placed anywhere in the current `sphera_jail` scenario, or does the
  scenario need updating first to have a real target object?
- Single-shot detection-then-navigate, or continuous re-detection while flying (in case the
  object moves or the first detection's position estimate was off)?

## Notes

## Result
