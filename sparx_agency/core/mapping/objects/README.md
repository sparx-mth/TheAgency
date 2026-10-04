# Observed object geometry and landmarks

ROS-free geometry and persistent object bookkeeping, shared by the scene-graph
runtime and simulator adapters. Inputs are predictions and measured geometry,
not privileged scene annotations.

- `geometry.py`: camera-intrinsic box transforms, robust box depth and world
  projection. Camera optical and body/world frame conventions are documented
  there; do not substitute unrotated image coordinates.
- `landmarks.py`: `ObjectLandmarkMap` with per-class radius association and an
  observation-confirmation threshold. Existing defaults retain the original
  first-in-radius behavior. `nearest_match=True` chooses the closest eligible
  same-class landmark instead; it does not merge different semantic classes.
- `class_votes=True` turns association positional: an observation on an
  existing landmark's footprint (centroid within the dedupe radius, or
  footprint discs with `disc_iou` ≥ `footprint_iou`, default 0.15 ≈ equal
  discs up to 1.25 radii apart) is folded into it *as a vote for its class*.
  `ObjectLandmark.votes` is the per-class tally; `class_name` is the plurality
  (a tie keeps the current class), relabels are listed in `map.relabels`, and
  `confirmed()` needs the leader to hold `min_observations` votes and beat the
  runner-up. `observe(..., radius_m=...)` carries the measured footprint
  half-extent; `matches()`/`match()` expose the association so a caller with
  information this 2-D map lacks (object height) can pick the landmark and
  pass it back as `observe(..., landmark=...)`.
- `observe(..., frame_id=step)` counts at most one observation for a landmark
  in that frame. Repeated prompt boxes must not manufacture confirmation.
  Omitting `frame_id` preserves the old behavior. Create/reset the map per
  episode or identity domain so frame ids cannot collide across episodes.

Alias normalization and overlap suppression belong before association, at the
runtime's detector-vocabulary boundary. They must not use fuzzy target matching
or turn distinct object classes into equivalent navigation goals.

Tests: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest
sparx_agency/core/mapping/objects/tests -q`.

