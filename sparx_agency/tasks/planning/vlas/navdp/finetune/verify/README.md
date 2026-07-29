# Verify the NavDP → instantaneous-PF/ESDF correction (before training)

This tool lets you **see the training signal on a real flight frame** before spending
GPU-hours fine-tuning. It implements the exact loop you described:

1. take a colour + depth frame (from `~/flight_dataset/<rec>/`, produced by the bag
   extractor);
2. **click a pixel** you want to reach → it becomes a body-frame point goal;
3. run **NavDP** → a trajectory;
4. build the **instantaneous potential field / signed ESDF** from *that same depth
   frame*;
5. **push the NavDP trajectory lightly off the walls** → the corrected target;
6. **compare** NavDP vs corrected, side by side, on a bird's-eye map.

If the corrected trajectory looks right across many clicks/frames, the fine-tune
target is trustworthy — then (and only then) generate labels and train.

## The "connection" (why no coordinate transform is needed)

Everything already lives in **one frame**: the drone **body FLU** frame, `x = forward`,
`y = left`, meters, robot at the origin.

* NavDP emits its trajectory in body FLU (meters).
* `common/frames.py` builds the single-frame occupancy/ESDF in the **same** FLU frame
  (grid origin at the robot, `x=fwd`, `y=left`).

So connecting them is not a transform — we simply hand NavDP's waypoints to
`common/esdf_target.generate_target(..., seed_path=navdp_traj)` as the seed it
corrects (instead of the default straight-to-goal seed). `verify/correction.py` is that
one-line glue. The push off the walls is done by the repo's existing correctors.

## Correct, then smooth

The corrector pushes each waypoint off the walls *independently*, so a strongly-pushed
point next to an un-pushed one leaves a **kink**, and two points pushed opposite ways
leave a **zigzag** — something NavDP (a smooth cumsum of small deltas) would never emit,
and a poor training label. So after correcting we run a **collision-aware smoothing**
pass (`common/smoothing.py`): a count-preserving Gauss-Seidel relaxation that flattens
kinks **only where the relaxed point stays clear** of the (inflated) walls — smoothing
never undoes the wall-avoidance. Endpoints (robot, goal) stay pinned and the 24-waypoint
horizon is preserved. The BEV/comparison panels draw the **raw** correction (tomato,
dashed) under the **smoothed** target (green) so you can see the kink being removed.

## Run it

Runs in the **`navdp` conda env** (it has TensorRT + numpy + matplotlib; **no torch
model, no external NavDP repo, no transformers** are loaded — inference is the built
fp16 TensorRT engines via `core/planning/vlas/navdp/trt`).

```bash
PYTHONPATH=/home/nadavc/GIT/TheAgency \
  ~/miniconda3/envs/navdp/bin/python -m \
  sparx_agency.tasks.planning.vlas.navdp.finetune.verify.interactive_verify \
  --dataset ~/flight_dataset --rec walk_into --frame 40
```

**Click** the colour or depth image to set a goal. The four panels:

| panel | shows |
|---|---|
| top-left | colour image + clicked pixel (red ✕) + goal / critic in the title |
| top-right | depth image + clicked pixel |
| bottom-left | **the field** (signed ESDF or repulsion) + occupancy + NavDP (orange), raw correction (tomato dashed) & smoothed target (green), robot ▲, goal ★ |
| bottom-right | NavDP vs raw-corrected vs smoothed **comparison** with per-waypoint shift vectors + `moved N · smoothed M · max/mean shift` |

## Knobs

* **corrector** — `esdf` (default: per-waypoint distance-gradient ascent, *local and
  gentle*) or `potential_field` (medial-axis re-centering, *more aggressive*).
* **clearance m** — how far off a wall the corrector aims (the push target). Higher →
  it engages sooner / pushes more. Default `0.5` (matches NavDP's own `d_safe`).
* **max shift m** — cap on how far any waypoint may move ("not too hard"). Default `0.8`.
* **smooth** — smoothing strength `0..1` (`0` = off). Relaxes kinks/zigzags out of the
  pushed trajectory so the target is smooth like NavDP. Default `0.5`.
* **pitch deg**, **cam height m** — shape the single-frame occupancy (which pixels are
  floor vs obstacle). **These are hardware constants you must measure** — the live stack
  does not record camera pitch (see the package README). Defaults `0°, 1.0 m` are XTEND
  placeholders; tune them until the occupancy walls match what you see.
* **field** — view the signed ESDF or the derived repulsion `max(clearance − ESDF, 0)`.
* **sample N** — overlay N random pixel goals' corrected trajectories, to preview the
  diversity of the training data ("~100 pixels per image").
* **◀ frame / rec ▶**, **save png**.

Sliders retune the correction **live without re-running NavDP** (NavDP depends only on
the frame + goal, not on the correction knobs), so tuning is instant.

## Headless preview

To render a grid of sampled goals to a PNG (no display, or to share):

```bash
PYTHONPATH=/home/nadavc/GIT/TheAgency ~/miniconda3/envs/navdp/bin/python -m \
  sparx_agency.tasks.planning.vlas.navdp.finetune.verify.batch_preview \
  --rec walk_into --frame 40 --n 6 --out preview.png
```

## Files

| file | role | deps |
|---|---|---|
| `pixel_goal.py` | clicked depth pixel → body `(fwd, left)` goal (deployed convention) | numpy |
| `navdp_infer.py` | single-frame NavDP point-goal inference over TensorRT | tensorrt + numpy |
| `correction.py` | seed `generate_target` with the NavDP trajectory (the connection) | numpy |
| `pipeline.py` | one pixel → goal → NavDP → corrected target; frame IO; pixel sampling | numpy |
| `bev_render.py` | matplotlib drawing for the four panels | matplotlib |
| `interactive_verify.py` | the clickable app (this is the entry point) | matplotlib |
| `batch_preview.py` | non-interactive PNG grid | matplotlib (Agg) |

## Next step (not this tool)

Once the method looks right here, the same `run_pixel` pipeline generates the actual
training labels: for each frame, sample ~100 pixel goals, and for each save the pair
*(NavDP output, corrected target)* encoded with `common/label_format.to_navdp_label`
(and `to_flownav_label`) plus the signed ESDF grid for the differentiable penalty. That
is the fine-tune dataset — see the package README's training section.
