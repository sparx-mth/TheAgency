"""Map-supervised, world-goal NavDP fine-tuning: the end-to-end training pipeline.

The difference from :mod:`..pixel_goal` in one sentence: **the supervision comes
from a surveyed ground-truth map of the building, not from NavDP's own output**,
and the goal is a *world* point drawn from that map rather than a pixel clicked
in the camera.

That single change fixes the three things wrong with the earlier fine-tune:

1. **Goals could land on obstacles.** A pixel back-projected by depth is a point
   on whatever surface the ray hit -- usually a wall. Here every goal is drawn
   from the map's free, clear, landable, connected region and is additionally
   required to be *reachable* by A*, so a goal on an obstacle is structurally
   impossible (:mod:`.goal_sampler`).
2. **The label was NavDP's own trajectory, corrected.** Self-distillation caps
   the student at the teacher. Here the label is an independent expert: the
   safest route the global planner can find, re-centred on the corridor's medial
   axis (:mod:`.expert`).
3. **Safety was measured on a single depth frame.** A monocular frame cannot see
   round a corner, so it cannot supervise anticipating one. The clearance term
   here reads the *global* signed ESDF at the sample's world pose, so a waypoint
   heading into a wall the camera cannot see is still penalised (:mod:`.loss`).

Run order (each stage has a ``--help``)::

    build_dataset.py    recordings + surveyed map -> labelled samples, split 3 ways
    cache_features.py   frozen DINOv2 tokens per frame  (optional, ~30x faster training)
    train.py            the fine-tune, logging every term to metrics.jsonl
    plots.py            metrics.jsonl -> loss curves           (train.py runs this)
    evaluate.py         paired baseline-vs-trained on the held-out TEST zone
    fly_navdp.py        closed-loop flights in PEGASUS, untrained vs trained
    report.py           everything above -> one self-contained HTML page

See ``README.md`` next to this file for what NavDP is, what is trained, and why
the objective function looks the way it does.
"""
