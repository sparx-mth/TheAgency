"""Rigorous baseline-vs-fine-tuned NavDP comparison.

``train/evaluate.py`` answers "did the network imitate its teacher". This package
answers the harder question -- *is the route actually safer* -- by fixing three
problems with that comparison:

1. **Circular target.** The training label is ``correct_navdp_trajectory(...)``
   output, and ``evaluate.py`` scores against that same corrected path. That
   measures imitation, not safety, so ``dist_target`` is reported here only as a
   diagnostic and never as a verdict.
2. **The teacher's own ruler.** The teacher raised clearance against a
   *single-frame* ESDF and the old metrics scored against that same field, so
   part of the gain was guaranteed by construction. :mod:`judge_map` instead
   fuses several posed depth frames into an accumulated map the teacher never
   saw, and scores against that.
3. **Means hide the tail.** Safety is a worst-case property. :mod:`stats` reports
   paired per-sample deltas, win/loss counts, quantiles and a Wilcoxon
   signed-rank test rather than a bare mean.

Modules:
    bag_poses:  ``/xtend/april_tag_pose`` -> per-depth-frame ``world_T_cam``.
    judge_map:  posed multi-frame occupancy + clearance field (the judge).
    metrics:    per-trajectory safety metrics against a clearance field.
    stats:      paired statistics and significance testing.
    compare:    the driver -- three arms (baseline / trained / teacher), CSV+JSON.
    report:     renders a shareable HTML summary from the driver's output.
"""
