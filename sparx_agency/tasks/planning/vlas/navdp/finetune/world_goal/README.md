# NavDP world-goal fine-tuning

An end-to-end pipeline for teaching NavDP to fly **to a destination by the
safest route**, supervised by a surveyed ground-truth map of the building rather
than by its own output.

Everything runs in the `navdp` conda env:

```bash
conda run -n navdp python -m sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.<stage> --help
```

> Run it in `navdp`, not `.venv`. The pip-installed `ompl` bindings in `.venv`
> corrupt the heap at interpreter shutdown and `core/planning/planners/` imports
> them eagerly, so a `.venv` run prints correct results and *then* aborts with
> exit 134 — and a pytest run there can abort during *collection*, reporting far
> fewer tests than exist. `navdp` has no OMPL and exits cleanly.

Tests (32, ~5 s; the torch half skips cleanly without the checkpoint):

```bash
conda run -n navdp pip install pytest            # not there by default
NAVDP_REPO=~/PycharmProjects/NavDP/baselines/navdp \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD \
  conda run --no-capture-output -n navdp python -m pytest \
  sparx_agency/tasks/planning/vlas/navdp/finetune/world_goal/tests -q
```

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is required: ROS2's `launch_testing` pytest
plugin is discovered even under conda and dies importing `lark`.

---

## 1. What NavDP is, in plain terms

NavDP is a **diffusion policy for navigation**. You give it what the camera sees
and a direction to go; it gives you a short path to fly. 135.7 M parameters in
four pieces:

```
 8 past RGB frames  ──►  DINOv2 ViT-S  ──┐
 (224x224 each)          (frozen)        │
                                         ├──►  Q-Former  ──►  128 tokens
 1 current depth frame ─►  DINOv2 ViT-S ─┘   (128 learned      "the scene"
 (metric, 0.1-5 m)         (frozen)           queries)          (128 x 384)
                                                                    │
 goal (forward, left) ──►  Linear(3, 384)  ──► 1 token  ────────────┤
                                                                    ▼
                                                      16-layer transformer
                                          decoder, shared by two heads:
                                                     │            │
                            action head ◄────────────┘            └──► critic head
                            predicts the noise                    scores a whole
                            in a noisy 24-step action             trajectory (1 number)
```

**How one prediction is made.** The decoder is a *denoiser*. It starts from 24
steps of pure random noise and refines them 10 times, each time predicting "what
part of this is noise", conditioned on the scene and the goal. What comes out is
a `(24, 3)` action tensor of per-step increments; the trajectory is
`cumsum(action / 4)`, so each step is at most 25 cm and the whole prediction
covers a few metres ahead. The third channel is a heading increment.

**Why sixteen of them.** Diffusion is stochastic, so NavDP draws **16 samples**
in parallel and the **critic** scores each one. The highest-scoring sample is the
one that flies. The critic is deliberately goal-blind — its conditioning masks
the goal token out — so it answers "is this trajectory *traversable*", not "does
it go the right way".

Two conventions worth knowing because everything downstream depends on them:

- **Frames.** Body FLU: `x` forward, `y` left. Goals and trajectories are both
  in the aircraft's frame, so they are consistent with each other by
  construction.
- **Depth is short-sighted on purpose.** NavDP zeroes anything past 5 m. In an
  office corridor most of the far wall reads as "no measurement". The depth
  image solves the next few metres; everything beyond it has to arrive through
  the goal token — which is exactly why taking the goal from a map matters.

---

## 2. What we are trying to achieve

Two things at once, which is the whole difficulty:

1. **Reach the destination.** Head toward the goal, including when the goal is
   round a corner and nothing about it is visible.
2. **By the safest route.** Fly down the *middle* of a corridor, not along one
   wall. Start turning *before* a corner, not at it.

A single loss term cannot say both — "get closer to the goal" and "stay away
from walls" pull against each other, and whichever is scaled larger silently
wins. So the objective has separate terms for each and the balance is explicit
and logged (§6).

---

## 3. What changed from the previous fine-tune, and why

The earlier attempt (`../pixel_goal/`) trained on real drone footage with three
structural problems. Each is fixed here by a different piece:

| Problem | Why it mattered | Fix |
|---|---|---|
| Goals were **pixels back-projected by depth** — so a goal *was* whatever surface the ray hit. Roughly a third landed on walls. | The network was taught to fly into geometry. | Goals are drawn from the map's free, clear, landable, connected cells and must be reachable by A*. A goal on an obstacle is not filtered out — it cannot be produced. (`goal_sampler.py`) |
| The **label was NavDP's own trajectory**, pushed off walls. | Self-distillation caps the student at the teacher; it cannot introduce behaviour the teacher never had, like anticipating an unseen corner. | The label is an independent expert: weighted A* on the global map, re-centred on the corridor's medial axis. (`expert.py`) |
| Safety was measured on **one depth frame**. | A monocular frame cannot see round a corner, so it cannot supervise anticipating one — and it was the same ruler the teacher used, making the evaluation circular. | The clearance term reads the **global signed ESDF** at the sample's world pose; evaluation scores against the same map, which no part of training can influence. (`loss.py`, `metrics.py`) |

Two smaller but real ones: the memory is now **eight real frames** instead of
one frame copied eight times (the earlier dataset threw away every motion cue
the architecture exists to use), and the critic now sees **wrong trajectories**
as well as right ones, so its ranking of 16 samples means something.

---

## 4. What is trained and what is frozen

**Trained — 44.5 M of 135.7 M:**

| Block | Params | Why it must adapt |
|---|---|---|
| Fusion decoder (16 layers) | 37.9 M | This *is* the policy — the denoiser and the critic share it. |
| Q-Former (`former_net`, `former_pe`, `former_query`, `project_layer`) | 6.6 M | Decides what in the picture matters. Adapting "what to look at" is cheaper and safer than re-learning "how to see". |
| Point-goal encoder + action/critic heads + embeddings | 0.06 M | The output geometry and the goal encoding. |

**Frozen — always:** the **RGB DINOv2 trunk** (22.1 M). It is a general visual
representation trained on far more imagery than this dataset contains, and it is
the single thing keeping a fine-tune on one office building from collapsing onto
that building.

**Frozen by default, unfreezable:** the **depth DINOv2 trunk** (22.1 M). This is
the viewpoint-sensitive path and the right thing to adapt in a short second
stage at a 10x lower learning rate (`train_depth_encoder: true`, no feature
cache).

Never touched: the image-goal and pixel-goal encoders (44.7 M). NavDP ships
them; this stack does not use them and neither does the TensorRT export.

Two capacity brakes, because 37.9 M trainable parameters against one building is
the main overfitting risk: `train_decoder_last_n` trains only the last N decoder
layers, and **L2-SP** pulls all trained weights back toward their pretrained
values throughout.

### The architecture is not changed, and that is deliberate

No widths, depths or token counts move: 384-dim tokens, 128 Q-Former queries,
16 decoder layers, 8 memory frames, 24 prediction steps, 10 diffusion timesteps.
Several of those are not free parameters at all — the Q-Former's positional table
is sized `(8 + 1) x 256` and the decoder's output positional embedding is sized
24, so changing the memory length or the horizon discards a pretrained tensor.

More generally: **the pretrained weights are the asset here.** A few thousand
frames of one office cannot train a wider network from scratch, and any change
to a layer's shape throws away the part of NavDP that already knows how to
navigate. Every lever in this pipeline is about *what to train and what to teach
it*, not about how big it is. If capacity ever does become the limit, the
evidence would be training loss plateauing well above validation loss — the
opposite of what a small-data fine-tune normally shows.

### Why this is fast

With both trunks frozen, their output for a given frame **never changes**. That
output is ~99 % of a forward pass — nine ViT-S passes per sample. So
`cache_features.py` computes it once and training never runs a ViT again:
roughly **1.5 s per optimiser step becomes 0.05 s** on an 8 GB laptop GPU, which
is what makes a real run possible rather than a token one.

---

## 5. Where the supervision comes from

For one recorded frame at a known world pose, and one sampled world goal:

1. **Weighted A\*** plans a route on the surveyed map. Its soft clearance cost
   already prefers the middle of a corridor to its edge, and because it plans on
   the *whole building* it turns toward a doorway several metres before the
   camera can see it. **This is where corner anticipation comes from.**
2. **Medial-axis centring** samples clearance along the path normal and moves
   each waypoint to the local maximum — the point equidistant from both walls.
   Two guards stop it doing harm: the correction fades in over the first metre
   (a drone cannot step sideways) and switches off once a waypoint already has
   1.5 m of room (chasing a distant clearance maximum across an open hall just
   adds turning). Measured effect: **+7 cm of worst-case clearance on the
   tightest 20 % of routes for +0.8° of turn**. Most of the centring is A*'s;
   this is the refinement where it counts.
3. The first **4.8 m** are resampled to NavDP's 24 steps and encoded as its
   action tensor. The horizon is *arc length*, which gives arrival behaviour for
   free: a goal 20 m away fills all 24 steps at ~0.2 m each; a goal 2 m away
   spreads 2 m over 24 steps and the per-step displacement shrinks toward zero —
   exactly the "stop" signal NavDP's own post-processing looks for.
4. The label is then **decoded again and audited against the map**. Anything
   whose own waypoints come within 30 cm of geometry, or that turns more than
   100° inside the horizon, is discarded rather than taught.

Why 4.8 m and not 6: the action is clamped at ±1 after a ×4 scale, i.e. 0.25 m
per step per axis, so 24 steps can express at most 6 m. 4.8 leaves headroom for
the lateral component of a turn instead of saturating the clamp on every sample.

### Choosing the goal

A frame supports many goals — the picture is shared, only the goal token and the
target route change — which is what multiplies a few thousand frames into a real
dataset. Twelve per frame, drawn in a fixed mixture so the policy sees the whole
job:

| kind | share | range | bearing off the nose | what it teaches |
|---|---|---|---|---|
| `route` | 25 % | 2–20 m along the flown path | (follows the flight) | the most natural goal |
| `near` | 15 % | 1.5–5 m | ≤ 45° | arriving and slowing down |
| `mid` | 25 % | 5–12 m | ≤ 45° | ordinary corridor cruising |
| `far` | 15 % | 12–25 m | ≤ 30° | committing to a direction |
| `corner` | 20 % | 5–12 m | **30–75°** | turning toward something not visible |

**The bearing bands are per kind, and that is the single most important knob in
the sampler.** With one global cap the goal bearing comes out nearly uniform over
its whole range, and since the label ends up pointing at the goal that makes the
*median* training sample a 45° turn with under 10 % of samples going roughly
straight — measured, on the first build of this dataset. A policy needs "keep
going" examples as much as it needs corners. Ask for more turning by raising
`corner`'s **weight**, not by widening everything.

Because the goal comes from a map rather than a camera it does **not** have to be
visible, which is the point. Goals more than 75° off the nose are rejected: NavDP
collapses anything at or behind the camera plane to a fixed stub, so such a
sample carries no information about where it was meant to go.

> Label generation is A*-bound and the `far` band is most of the cost. Two
> lessons from building this the first time, both now fixed in the defaults:
> the far ceiling was 35 m, where a route searches almost the whole building
> (the goal token is clipped to 10 m regardless, so a 35 m goal and a 20 m goal
> on the same bearing usually give the same first 4.8 m anyway); and the
> planner's stock 200k node budget is *smaller than this building has cells*, so
> those long routes were quietly returning NO_PATH and deleting the long-range
> goals from the dataset. `SceneConfig.max_expansions` is now 800k.

---

## 6. The objective function

```
L  =  1.0 · L_act          diffusion epsilon-MSE          (behaviour cloning)
   +  1.0 · L_waypoint     |predicted - expert| in metres (imitate the geometry)
   +  1.0 · L_clearance    hinge on the GLOBAL signed ESDF (the safety floor)
   +  0.1 · L_goal         match the expert's remaining distance to the goal
   +  0.1 · L_critic       value regression, expert vs 3 wrong trajectories
   +  1e-3 · ||θ - θ₀||²   L2-SP, anti-forgetting
```

**`L_act` — behaviour cloning.** `MSE(ε̂, ε)`: predict the noise that was added to
the expert action. This is NavDP's own pretraining objective, the only term that
speaks the network's native language, and it carries the bulk of the learning.
On its own it is scale-free noise regression and a poor *geometric* teacher,
which is why the next two exist.

**`L_waypoint` — reach the destination, and fly down the middle.** Smooth-L1 in
**metres** between the decoded predicted trajectory and the decoded expert one.
Because the expert was already re-centred on the corridor's medial axis, matching
it geometrically *is* learning to fly down the middle. Smooth-L1 rather than MSE
so one bad label cannot dominate a batch.

**`L_clearance` — the safety floor.** `mean(relu(0.55 m − SDF(waypoint))²)`,
where the SDF is sampled from the **global map** after transforming the predicted
waypoints into world coordinates with the sample's pose. This is the term that
could not exist before: it penalises a trajectory heading into a wall **the
camera has not met yet**. The margin is the 0.35 m airframe radius plus a real
safety margin. Note the division of labour: `L_waypoint` teaches centring
(through the corrected expert), `L_clearance` is the hard floor that has to hold
even where the label is imperfect.

**`L_goal` — arrive.** Match the expert's remaining distance to the true world
goal. Route *shape* is already covered by `L_waypoint`; this is about *progress*,
and it survives when two ways round an obstacle are equally good and the waypoint
term is therefore ambiguous. Deliberately low-weighted: it is a tiebreaker, not
a driver.

**`L_critic` — make the ranking mean something.** NavDP flies one of 16 samples
and the critic chooses, so the critic decides what actually flies. Training it
only on expert trajectories teaches nothing — it never sees a bad one. Here each
sample is scored against three deliberately wrong trajectories (the expert
rotated 20–100°, a straight line on an arbitrary bearing, the expert pushed
sideways into a wall), every one valued by the true map:

```
V(τ) = ( −#{waypoints closer than 0.5 m} + 0.1 · Σ clearance gain ) / 24
```

The `/ 24` matters. The earlier fine-tune left this unnormalised, where it
reached 30-plus and numerically drowned every other term in the sum.

**Noise-level weighting.** The three geometric terms are evaluated on the
one-step estimate `x̂₀`, which at high diffusion noise is barely more than a
guess. Penalising the geometry of a guess teaches nothing and adds variance, so
each is weighted by `ᾱ_k` — full weight where the estimate is sharp, vanishing
where it is not. `L_act` is unweighted; it is well-posed at every noise level.

Every term is logged **both raw and weighted**, so a curve can be read.

---

## 7. Three environments, not three random subsets

A policy that has seen a corridor will fly that corridor. Splitting frames at
random would put the same metre of the same corridor — at 10 Hz — in both train
and test, and the resulting numbers would measure memorisation.

So the split is **spatial**. The surveyed office spans y ∈ [−34.5, 40.0] m and is
cut into three disjoint bands (`configs/splits_office.yaml`):

| split | region | share | used for |
|---|---|---|---|
| **train** | y ≥ −8 m — north hall, east spine, open plan | ~66 % | gradient steps |
| **val** | y ∈ [−15, −8) m — the south transition band | ~21 % | picking the checkpoint, nothing else |
| **test** | y < −15 m — the deep south wing | ~13 % | the only number that means anything |

A sample counts for a split only if the aircraft is inside that region **and the
whole expert route stays there**; a 1.5 m buffer strip either side of every
boundary keeps anchors off the line. What *may* cross is the **goal** — it is two
numbers handed to the policy as a direction and leaks no imagery, and far goals
pointing out of the region are exactly the samples worth keeping.

One consequence to know about: a band only 7 m tall against a 4.8 m label
horizon throws away most samples that do not point *along* the band, which
biases that split toward near goals. The val band above is the narrow one, and
val only selects the checkpoint, so the cost is small — but if you re-cut the
split, keep every band comfortably wider than `ExpertConfig.horizon_m`. Widening
val here is not free: the frames simply are not evenly spread down the building,
so a wider val band comes straight out of test.

### Prefer a held-out building once you have more than one

`configs/splits_multiscene.yaml` is the split to use when several buildings have
been flown, and it is strictly better than the y-bands above:

| | | |
|---|---|---|
| **test** | `warehouse_shelves`, entire building | 535 m² the policy never sees |
| **val** | `full_warehouse`, southern fifth | ~310 m², picks the checkpoint |
| **train** | `office` entire + the rest of `full_warehouse` | ~2050 m² |

Two reasons, one of them not obvious:

* **An unseen wing is a weak test.** It shares the building's architecture,
  lighting, renderer and asset set; the corridors either side of the split line
  are built from the same handful of models. A warehouse full of shelves shares
  none of that with an office, so transferring to it is evidence about
  navigation rather than about memorisation.
* **A whole-scene split is also cheaper in samples.** `SplitPlan.route_inside`
  returns `True` unconditionally for a scene assigned whole — there is no
  internal boundary for a route to cross — whereas a band split discards every
  label whose expert route leaves its band, which on the office y-bands was a
  large fraction of them.

Office stays in **train** deliberately: it is the only surveyed scene with
corridors, doorways and desks, and the closest thing here to the building the
real XTEND flies. Validation is a band rather than a fourth building because
with three buildings, spending one on checkpoint selection would drop either the
office or the shelved warehouse out of the experiment; `full_warehouse` is much
the largest at 1570 m² reachable, so a fifth of it is the cheapest band
available. Validation only picks a checkpoint — it never has to be unseen
*architecture*.

---

## 8. How much data

From the 27 recordings on disk (21 Isaac A-to-B episodes + 6 FALCON exploration
runs, ~16.9 k posed frames), after admission filtering:

- **~11 k frames** survive: within ±0.35 m of cruise altitude (this discards
  climb and descent, which look at a completely different building from the one
  the 60 cm map slab describes), under 25° of tilt, inside the map, and not
  standing in geometry.
- at `--frame-stride 2` (consecutive frames are 12 cm apart and nearly the same
  picture) and 12 goals per frame, that is **tens of thousands of labelled
  samples** from a few thousand distinct viewpoints.

**Outcome barely matters, and that is a consequence of the design.** A crashed
episode still contains a hundred good posed observations before it hit anything,
and the label taught at each is the safe route a planner *would* have flown. What
has to be true is that the pose is valid. `--strict-outcomes` restores the
conservative reading for an ablation.

The dataset stores **pointers, not pixels** — recording id, frame index, goal
token, and the 24×3 label, about 350 bytes a sample. Relabelling with a different
horizon or goal mixture costs minutes and no disk.

---

## 9. Running it

```bash
REPO=~/GIT/TheAgency && cd $REPO
WG=sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal
OUT=~/navdp_world_goal
export NAVDP_REPO=~/PycharmProjects/NavDP/baselines/navdp

# 1. labels  (numpy only, ~30 min on 7 cores; no GPU)
conda run -n navdp python -m $WG.build_dataset \
    --recordings ~/sim_flight_recordings ~/sim_flight_recordings_v2 \
                 ~/sim_flight_recordings_v3 ~/sim_flight_recordings_v4 \
                 ~/falcon_pegasus_recordings \
    --scene office --splits $REPO/${WG//.//}/configs/splits_office.yaml \
    --out $OUT/dataset --frame-stride 2 --goals-per-frame 12 --workers 7

# 2. frozen-ViT feature cache  (optional but ~30x faster training; a few GB)
conda run -n navdp python -m $WG.cache_features \
    --dataset $OUT/dataset --out $OUT/features --ckpt ~/Downloads/navdp-cross-modal.ckpt

# 3. train
conda run -n navdp python -m $WG.train \
    --dataset $OUT/dataset --features $OUT/features --out $OUT/run1 \
    --ckpt ~/Downloads/navdp-cross-modal.ckpt
# -> best.pth, last.pth, milestone_{25,50,75}.pth, metrics.jsonl, run.json,
#    training_curves.png, training_nav.png, training_health.png

# 4. evaluate on the held-out TEST wing (paired, against the true map)
conda run -n navdp python -m $WG.evaluate \
    --dataset $OUT/dataset --features $OUT/features --run $OUT/run1
# -> evaluation.json, per_sample.csv, routes.png

# 5. one page with everything
conda run -n navdp python -m $WG.report --run $OUT/run1 --dataset $OUT/dataset
```

**Checkpoints: five, not fifty.** `best.pth` (lowest validation loss), `last.pth`
and three evenly spaced milestones, each holding only the ~44.5 M trainable
tensors (~178 MB) rather than the full 543 MB model. Enough to see the shape of
the run and to roll back.

### Putting it back into the rest of the stack

```bash
conda run -n navdp python -m $WG.export_checkpoint \
    --run $OUT/run1 --base ~/Downloads/navdp-cross-modal.ckpt \
    --out $OUT/navdp-world-goal.ckpt
```

That writes a full NavDP-format checkpoint, which drops straight into everything
already built here: `serve/navdp_trt_server.py --backend torch --ckpt <it>`
serves it over the HTTP contract the FALCON nodes already speak (so
`navdp_click_node`, `hybrid_planner_node` and the rest fly the fine-tune with no
code change), and `trt/export/export_onnx.py` → `trt/engine/build_engine.py`
builds TensorRT engines from it for the real aircraft.

### Closed-loop flights: trained vs untrained

Everything above is open-loop — one prediction, one frame. It cannot see the
failure that matters most, which is a small bias compounding over a hundred
inferences. `fly_navdp.py` closes the loop in PEGASUS: identical missions from
one seed, flown once per arm, scored against the surveyed map, with video.

**The aircraft commits to a plan before asking for another one**, and that is
not a detail. Training and offline scoring are frame-by-frame, and the obvious
way to deploy the result is to match them: re-infer on a timer, steer at
whatever the newest prediction says. In the air that is a pathology. At 3 Hz and
1 m/s the aircraft covers 0.33 m of a 4.8 m plan before that plan is thrown away
— it executes the first 7 % of everything the policy predicts and none of the
route shape, and the only thing that ever compounds is whatever bias lives in
that first segment. So one prediction is anchored where it was made, flown as a
route with pure pursuit until roughly half of it is behind the aircraft, and
only then replaced. `--infer-hz` is now a rate *ceiling*, not a schedule;
`--commit-fraction` (default `0.5`, so waypoint 12 of NavDP's 24) is the rule.
The rule and its four escape hatches live in
[`core/planning/vlas/common/plan_commit`](../../../../../../core/planning/vlas/common/plan_commit/README.md),
so FALCON's `navdp_click_node` gets the same behaviour and the simulator is
flying what the aircraft flies. In the map panel the committed part is solid
green and the speculative tail dashed orange, because they are two different
claims.

**And the aircraft could not turn.** Worth its own paragraph, because it hid
behind the first fault and looked exactly like a bad policy. The yaw setpoint
was built as `slew_towards(pose[2], yaw_command, rate, dt)` — slewing from the
*measured* heading rather than from the previous command. That caps the setpoint
at `rate · dt` ahead of where the aircraft already is: 0.24° at the 250 Hz
physics step, so PX4 was never handed a heading error worth turning for. A
measured flight yawed at **0.6 °/s** against the 40 °/s asked for; the aircraft
flew wherever it happened to be pointing, and on a mission whose goal was 22 m
north of a start facing west it stalled 20 m short having never turned. With the
command slewed instead — which is what `episode.py` and `falcon_pegasus`'s
mission have always done — the same mission, same seed, same untrained weights,
reaches the goal. Anything measured before this is measuring the harness.

One command does all of it — both arms, one Isaac session per mission, the
inference server swapped between arms, the flights copied out and the video cut:

```bash
bash sparx_agency/tasks/planning/vlas/navdp/finetune/world_goal/run_comparison.sh \
    --missions-to-fly 0,1,2 --out ~/navdp_world_goal/flights
```

Read `run_comparison.sh --help` before doing it by hand: it encodes three things
that are easy to get wrong — one mission per session (the aircraft cannot be
repositioned, so a second mission in the same session starts wherever the first
ended), one server at a time (two NavDP policies and Isaac Sim do not fit in
8 GB, and the failure looks nothing like its cause), and `ffmpeg`/`ffprobe`
coming from the `navdp` env rather than the bare host. The long way, for
reference:

```bash
# host, two servers (Isaac's Python has torch but not diffusers, so inference
# stays on the host and reaches the container over the existing HTTP contract)
conda run -n navdp python -m sparx_agency.tasks.planning.vlas.navdp.serve.navdp_trt_server \
    --backend torch --port 8888 --ckpt ~/Downloads/navdp-cross-modal.ckpt &
conda run -n navdp python -m sparx_agency.tasks.planning.vlas.navdp.serve.navdp_trt_server \
    --backend torch --port 8889 --ckpt $OUT/navdp-world-goal.ckpt &

bash sparx_agency/tasks/planning/sim_flight_recording/run_collection.sh --help  # for the docker cp pattern
docker exec isaac-sim /isaac-sim/python.sh \
    /tmp/dev/repo/sparx_agency/tasks/planning/vlas/navdp/finetune/world_goal/fly_navdp.py \
    --scene office --missions 6 --seed 4242 --arm baseline --server http://127.0.0.1:8888 \
    --out /tmp/dev/navdp_flights --video
# ... and again with --arm trained --server http://127.0.0.1:8889
# then copy the two arm directories back and fold them into the report:
docker cp isaac-sim:/tmp/dev/navdp_flights $OUT/flights
conda run -n navdp python -m $WG.report --run $OUT/run1 --dataset $OUT/dataset \
    --flights $OUT/flights
```

> **Status:** the offline pipeline (stages 1–5) has been run end to end. The
> closed-loop script had never been *executed*, only written, and a static audit
> found nine independent first-run failures — all now fixed: it unpacked five
> values from a four-value `bring_up`, never called `boot_isaac`, called a
> `px4_launch.launch` that does not exist (so PX4 never started and the run died
> 300 simulated seconds later in the heartbeat wait), got `configure_px4` and
> `settle_estimator` arity wrong, was missing six arguments `bring_up` reads off
> the namespace, handed `client.reset` a raw 3×3 list where an `Intrinsics` is
> required, returned a three-key dict on `arm_timeout` that `KeyError`d the
> caller and lost every mission already flown, and tore nothing down on failure
> so the second arm inherited the first arm's ports and PX4 parameters.
>
> One of the nine was in the server, not the script: `--backend torch` passed
> `render_cam_height` to upstream's `NavDP_Agent`, which does not accept it. The
> agent is built lazily inside `/navigator_reset`, so the server started
> cleanly and then returned 500 on the first reset and 409 on every step after
> it — indistinguishable from a dead policy. **Anything serving a fine-tuned
> checkpoint over the torch backend hit this**, not just closed-loop flights.

### The comparison video lied twice, and it is worth knowing how

The first comparison videos looked like the aircraft was badly mislocalised — the
position marker displaced from the takeoff point before the drone left the ground,
and the flown trail detached from it for most of the clip. Neither was real. Both
were composition faults, and between them they nearly cost a dataset:

* **`track_video.py` drew the flown path by *fraction of the flight*.** Inference
  is held off until cruise altitude, so the inferences cover the last ~26 s of a
  36 s flight; spreading the flown path evenly across them put the trail up to ten
  seconds behind the aircraft. It now renders on the flight's own clock, from
  `started_s` and `flown_dt` in the track log (schema 2), and the marker is the
  aircraft's position at that instant rather than the pose of the last inference.
* **`compare_videos.py` stacked clips of different lengths and played both from
  zero.** A 26 s panel over a 36 s camera means the map runs a third faster than
  the view. The panel is now rendered at the camera's own capture rate over the
  whole flight, and `check_alignment` warns if a pair still disagrees.

A track log written before this has no clock; it still renders, one frame per
inference, with a warning. **The takeoff drift in those videos was partly real,
though** — see `--no-vision` in
`tasks/planning/sim_flight_recording/README.md`, and the station-keeping fix in
`fly_navdp.fly_mission`, which used a heading-locked travel law as a position hold
and so corrected drift into the rear hemisphere at exactly zero speed.

---

## 10. Reading the results

**`training_curves.png`** — every loss term, train against validation, on its own
axes. A single falling total says almost nothing when it is a weighted sum of
five things pulling in different directions; the interesting failure — clearance
improving while goal-reaching quietly rots — is invisible unless the terms are
separated.

**`training_nav.png`** — the same run in metres and percent: worst clearance of
the predicted trajectories, how often a prediction enters geometry, and how far
from the goal the trajectory ends *against how far the expert's ends*. This is
whether the flying got better, which the loss does not say.

**`evaluation.json` / the report table** — three arms on the held-out test wing,
all answering the same frames with the same diffusion seed: `baseline`
(pretrained), `trained`, and `expert` (the label itself — the ceiling imitation
can reach, not a competitor). Per metric: the paired mean delta oriented so
positive is always better, **win/loss counts** (a mean gain built from a few big
wins alongside many small regressions is not a safety improvement), a Wilcoxon
signed-rank p, and a rank-biserial effect size.

Then everything again **split by how hard the label turns** (straight / gentle /
sharp). "Safer on straight corridors, worse at corners" and "safer everywhere"
produce the same overall mean and are completely different outcomes.

Be precise about what `turn_deg` is: the largest heading deviation from straight
ahead anywhere in the label. For most samples that is close to the *goal's*
bearing, because the label ends up pointing at the goal — so the buckets read as
"how far off-axis was it asked to go", which includes both a genuine corridor
corner and a goal simply off to one side. Both are cases the pretrained policy
handles worse than a straight run, which is why the axis is worth reporting; it
is just not a pure corner detector.

`centre_offset_m` is the direct measurement of the behaviour being trained for:
the distance from the corridor's medial axis, found by sampling clearance along
the path normal. Zero is dead centre, and it is independent of the expert — a
trajectory can be perfectly centred while looking nothing like the label.

**`routes.png`** — held-out samples drawn on the map: green dashed the expert,
orange the baseline, blue the fine-tune, from the same frame. Panels are chosen
by turn magnitude, because a straight corridor looks the same whatever the
policy does.

### The failure mode to watch for

Read `goal_gap_m` next to the clearance metrics. **Buying safety by becoming less
committal is real, easy, and looks like success** — a policy that stops early is
trivially safer. The earlier fine-tune did exactly this: wall-hits fell from 37 %
to 11 % while the distance from the goal grew by 13–20 cm. `L_goal` and the
expert arm exist to make that visible rather than convenient.

---

## 11. Known limits

- **One building.** The test wing is unseen geometry, but it is the same
  architecture, lighting and renderer. Surveying a second scene and using
  `scene_split:` is the single biggest improvement available.
- **Open-loop by default.** Stages 1–5 score one prediction per frame. Compounding
  error only appears in the closed-loop section, which needs a sim run.
- **Simulated imagery.** Nothing here says how the policy behaves on the real
  XTEND camera. The frames are geometrically interchangeable (the PEGASUS camera
  config is deliberately identical to the XTEND crop) but not photometrically.
- **The heading channel saturates** at roughly 8 % of steps, concentrated at
  corners: a per-step heading change above 14° clamps. Harmless — the deployed
  followers use only x/y — but it inflates `L_act` a little and is why the
  geometric terms are x/y only.
- **Colour order.** The deployed server converts RGB→BGR before the encoder, so
  the ImageNet statistics are applied positionally to BGR channels. That is an
  upstream quirk, but it is what the pretrained weights learned, so training
  matches it (`color_order: bgr`). Note `../verify/navdp_infer.py` does the
  opposite and is therefore one channel swap away from the deployed stack.
- **The feature cache is invalid once the depth trunk is unfrozen.** The trainer
  detects that and refuses, rather than training on stale tokens.

---

## 12. Files

| file | what it owns |
|---|---|
| `scene.py` | the surveyed map: occupancy, signed ESDF, traversable/goal regions, planner, centring field |
| `splits.py` | train/val/test as three disjoint places |
| `sources.py` | which recorded frames are admissible, and why the rest are not |
| `goal_sampler.py` | reachable-only goals, in a fixed mixture |
| `polyline.py` | arc length, resampling, truncation, body↔world, action decode (numpy) |
| `expert.py` | A* → medial-axis centring → NavDP action label, audited against the map |
| `build_dataset.py` | the label-generation CLI |
| `preprocess.py` | the one true image/depth preprocessing, shared by cache, dataset and inference |
| `cache_features.py` | frozen-ViT patch tokens, computed once |
| `dataset.py` | torch `Dataset`, cached-token or live-pixel |
| `model.py` | freeze policy, the cached-feature encoder, trainable-only checkpoints |
| `scene_field.py` | the surveyed ESDF resident on the GPU, sampled differentiably |
| `loss.py` | the six-term objective |
| `logger.py` | `run.json` + `metrics.jsonl` + the console table |
| `train.py` | the training CLI |
| `plots.py` | `metrics.jsonl` → the three figures |
| `infer.py` | NavDP inference exactly as deployed: 16 samples, 10 steps, critic picks |
| `metrics.py` | trajectory scoring against the true map, including corridor-centre offset |
| `evaluate.py` | the paired three-arm held-out comparison |
| `figures.py` | BEV route panels and split-coverage maps |
| `report.py` | one self-contained HTML page |
| `export_checkpoint.py` | merge the fine-tune into a full NavDP checkpoint |
| `fly_navdp.py` | closed-loop flights in PEGASUS, trained vs untrained |
| `track_log.py` / `track_video.py` | what the policy proposed, in world coordinates, drawn on the surveyed map |
| `aggregate_flights.py` | one-mission-per-session results → one `summary.json` per arm |
| `compare_videos.py` | the two arms side by side, camera over map |
| `run_comparison.sh` | the whole closed-loop campaign in one command: both arms, N missions, one Isaac session each, servers swapped between arms, video cut at the end |
