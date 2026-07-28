# Fine-tuning NavDP & FlowNav for drone flight (PF/ESDF-guided)

This package fine-tunes the two vision→trajectory policies (NavDP, FlowNav) from
their **~0.3 m ground-robot** training viewpoint to our **~1.0 m drone** viewpoint,
using a small set of flight recordings and a **potential-field / ESDF** signal that
pushes the predicted trajectory away from walls.

It is torch-heavy and dev/host-only: it lives under `tasks/` and must never be
imported by `core/` (which stays ROS-free and Python-3.8). The numpy foundation
(target generation, label encoding, augmentation, recordings) runs in the plain
`.venv`; the training loops run in the `navdp` / `flownav_trt` conda envs (torch +
the external model repos), exactly like the existing TensorRT export tooling.

---

## a. Architecture of the two networks

Both are **generative trajectory policies**: an image (+depth/goal) is encoded once,
then a small generative head produces a short future trajectory in the robot's body
frame. They differ in the generative mechanism and the goal type.

### NavDP — point-goal **diffusion** policy (deployed mode)

```
8 RGB frames ─▶ DINOv2 ViT-S/14 (RGB) ─┐
                                        ├─▶ Q-Former (128 queries, 2-layer) ─▶ rgbd_embed (128,384)
1 depth frame ─▶ DINOv2 ViT-S/14 (depth)┘
goal (fwd,left,0) ─▶ Linear(3→384) ─▶ goal token
                                                    ┌─ denoiser: 16-layer causal TransformerDecoder ─▶ ε̂ (24,3)
rgbd_embed + goal + noised action ─▶ shared decoder ┤
                                                    └─ critic:  same decoder, bidirectional ─▶ value (1)
```

| block | in → out | role | fine-tune |
|---|---|---|---|
| RGB DINOv2 ViT-S/14 | `(8,3,224,224)`→`(1,2048,384)` | frozen visual backbone | **freeze** |
| depth DINOv2 ViT-S/14 | `(1,224,224)`→`(1,256,384)` | viewpoint-sensitive depth features | PEFT / unfreeze |
| Q-Former (2-layer) | `(1,2304,384)`→`(1,128,384)` | RGBD fusion | train |
| shared decoder (16-layer) | — | denoiser **and** critic | train |
| `action_head`/`critic_head` | `(…,384)`→`(24,3)` / `(1)` | ε-prediction / value | train |

* **Output:** action `(24, 3)` = per-step `(Δfwd, Δleft, Δyaw)` **×4**; the trajectory
  is `cumsum(action / 4)` → 24 waypoints, body **FLU** frame (x=fwd, y=left), meters.
  `clip_sample=True` caps steps at ~0.25 m. Yaw is emitted but ignored downstream.
* **Sampling:** DDPM, **10 steps**, `squaredcos_cap_v2`, ε-prediction; 16 samples fanned
  out, ranked by the critic (index 0 executed); stop when `critic.max() < −3`.

### FlowNav — image-goal **flow-matching** policy

```
4 RGB obs ─▶ EfficientNet-B0 ─┐
current+goal ─▶ EfficientNet-B0 (6ch) ─┤─▶ 4-layer self-attn ─▶ obsgoal_cond (256)
current RGB ─▶ frozen DINOv2 (depth prior)┘
                                             ┌─ ConditionalUnet1D velocity field ─▶ v (8,2)   (run K−1×)
obsgoal_cond + noisy action ────────────────┤
                                             └─ DenseNetwork ─▶ temporal distance (1)
```

| block | in → out | role | fine-tune |
|---|---|---|---|
| obs EfficientNet-B0 | `(1,12,96,96)`→`(1,4,256)` | obs encoder (GroupNorm) | freeze |
| goal EfficientNet-B0 | `(1,6,96,96)`→`(1,1,256)` | obs+goal encoder | freeze |
| DINOv2 depth prior | RGB→`(1,1,256)` | **frozen** feature prior (no depth input!) | **freeze** |
| self-attn (4-layer) | `(1,6,256)`→`(1,256)` | fusion → `obsgoal_cond` | train |
| `ConditionalUnet1D` | `(N,8,2)+t+cond`→`(N,8,2)` | velocity field | train |
| `DenseNetwork` | `(1,256)`→`(1,1)` | temporal distance | train |

* **Output:** `(8, 2)` = 8 waypoints `(dx, dy)`, body frame, in **waypoint units**
  (meters ÷ `metric_waypoint_spacing`); de-norm = `(v+1)/2·(max−min)+min` then `cumsum`,
  with `action_min=[-2.5,-4]`, `action_max=[5,4]`.
* **Sampling:** deterministic Euler over `linspace(0,1,K)`, `K−1` velocity evals,
  `sigma=0` (rectified flow). K is a runtime knob (deploy default 4–10).

> FlowNav has **no depth/ESDF input port** — its "depth prior" is a frozen DINOv2 on
> RGB. So the ESDF only enters FlowNav via the **label** and the **penalty**, never as
> a network input.

---

## b. The loss function

Two independent uses of the same per-frame ESDF. They **never share a code path**:
one produces a fixed label (numpy, offline); the other is a live differentiable
regularizer (torch).

### (i) The corrected trajectory as a behavior-cloning label — *what it should output*

From one `(depth, intrinsics, goal)` frame we build a **single-frame local map** and
push a seed route (the drone's flown-future, or a straight shot to the goal) off the
walls with the repo's existing correctors:

```
depth ─▶ body-FLU cloud ─▶ height-banded occupancy grid ─▶ signed ESDF (compute_sdf)
                                     │
seed path ─▶ PotentialFieldPathCorrector / EsdfPathCorrector ─▶ corrected Path2D
                                     │
              label_format ─▶ NavDP (24,3 ×4 deltas)  /  FlowNav (8,2 wp-units)
```

This is the standard supervised target for each generator:

* **NavDP** — DDPM ε-loss: `noise~N(0,I)`, `k~U{0..9}`, `x_k = add_noise(x0, noise, k)`,
  `L_act = MSE(ε_θ(x_k,k,goal,rgbd), noise)` with `x0` = the corrected label.
* **FlowNav** — flow-matching: `x_t = (1−t)·noise + t·x1`, `u_t = x1 − noise`,
  `L_flow = MSE(v_θ(x_t,t,cond), u_t)` with `x1` = the normalized corrected label.

### (ii) A differentiable ESDF hinge — *penalize entering walls*

The signed ESDF grid is a **fixed lookup** (stop-gradient w.r.t. pixels). We sample it
at the network's **own decoded waypoints** with `grid_sample` (differentiable to the
coordinates) and apply a hinge below a clearance `margin`:

```
L_esdf = mean_i  relu(margin − SDF(waypoint_i))²          (margin = 0.35 m)
```

Signed SDF (not unsigned distance) so a waypoint driven *inside* a wall keeps a
monotone push-out gradient. Total loss per model:

```
NavDP  : L = L_act  + λc·L_critic + λe·L_esdf + λsp·L2SP
FlowNav: L = L_flow + α·L_dist    + λe·L_esdf + λsp·L2SP
```

* **NavDP critic** is the natural ESDF consumer: its privileged value target is
  `V(τ) = −Σ 1[d^k<0.5] + 0.1·Σ(d^{k+1}−d^k)`, computed from the same ESDF along the
  trajectory (`d_safe=0.5 m`). We regress the critic head to it.
* **Cutoff** = `margin` (0.35 m, between the 0.25 m tube radius and NavDP's 0.5 m
  `d_safe`); the ESDF is clamped to ±4 m so far-field gradients stay finite.
* **Per-frame** ESDF: yes — the field is computed from a single RGB-D frame's local
  occupancy, not the persistent BEV map, exactly as you asked.

### L2-SP (anti-forgetting)

With a handful of flights vs. 100k+ sim hours, we penalize `‖θ − θ₀‖²` toward the
pretrained weights (a Fisher-free EWC that needs no source data), so the model adapts
without catastrophically forgetting its strong navigation prior.

---

## c. Our training method (and whether to change the architecture)

**Do not change the core architecture for the first two picks.** Both models expose a
*test-time* steering surface and a *small-data-safe* fine-tune surface; adding layers
is a documented fallback, not the default. Never change image size, context length,
horizon, action normalization, or the noise/Euler schedule — those are what make the
pretrained checkpoint usable.

Ranked, per model:

| rank | NavDP | FlowNav | data | forgetting |
|---|---|---|---|---|
| 1 | **ESDF critic-rerank / score guidance** (no training) | **ESDF velocity guidance** in the Euler loop (no training) | none | none |
| 2 | fine-tune **critic head** on ESDF labels, backbone frozen | freeze EffNets+DINOv2, train **UNet + self-attn + dist** | tens | low |
| 3 | **DoRA/LoRA** on depth-ViT + Q-Former, train decoder+heads | **viewpoint-FiLM** + LoRA on the UNet (add a layer) | tens–100s | very low |
| 4 | preference (DPO/RWR) fine-tune of the decoder | Adjoint-Matching reward fine-tune | 100s+ | medium |

**Answering your questions directly:**

* *Add a layer / few layers, freeze the rest, then train all at low LR?* — The
  freeze-head-then-unfreeze schedule is exactly rank 2→3 (`train_depth_encoder: false`
  → `true`, discriminative LRs `1e-4` head / `1e-5` backbone). Adding a layer is the
  rank-3 **FiLM/adapter** fallback (`use_film: true`), used only when data is tiny.
* *Freeze layers?* — Yes: **always freeze the RGB DINOv2 (NavDP) and both EfficientNets
  + DINOv2 (FlowNav)**; train the fusion + generative head; adapt the depth path with
  PEFT if the height gap needs it.
* *Change the architecture at all?* — Recommended **no** for the first pick. The
  best small-data recipe is: freeze backbones + **PEFT** + aggressive **viewpoint/height
  augmentation** + **L2-SP** + **EMA**. Architecture changes (FiLM adapter, input
  homography) are reversible fallbacks.
* *Use the potential field / ESDF to shape the trajectory?* — Yes, two ways, as above:
  as the **corrected label** and as a **differentiable hinge**. ESDF is preferred over
  the raw potential field for the penalty because its signed distance gives a clean,
  non-saturating gradient and a natural clearance cutoff.
* *Per-frame field vs. the BEV map?* — Implemented per-frame from `(RGB, depth,
  intrinsics)`; no persistent map needed (`common/frames.py` + `common/esdf_target.py`).

### The height gap, concretely

Training cameras sit ~0.3 m and level; the drone flies ~1.0 m. The nearest visible
ground moves from ~0.8 m to ~1.6 m ahead, the horizon shifts up (XTEND `cy≈90` of 294),
and useful depth pushes toward NavDP's 5 m clip. We close this with **pitch-rotation
homography augmentation** (`common/augment.py`, exact for rotation) + depth-scale
jitter, re-generating the ESDF label for the warped frame so input and label stay
consistent. **Camera pitch is not encoded anywhere in the live stack — measure it on
hardware** and set `LocalMapConfig.pitch_deg`.

---

## Infrastructure layout

The **model-agnostic** half lives here; the **per-model** half lives with its
policy, so each VLA owns its `trt/` and `finetune/` side by side.

```
tasks/planning/vlas/common/finetune/          # ← this package: model-agnostic
  common/
    frames.py                # depth → body-FLU cloud → single-frame occupancy   [numpy]
    esdf_target.py           # occupancy → signed ESDF + PF/ESDF-corrected Path2D [numpy]
    label_format.py          # Path2D → NavDP (24,3) / FlowNav (8,2)             [numpy]
    augment.py               # pitch-homography viewpoint/height augmentation    [numpy]
    esdf_penalty.py          # differentiable ESDF hinge (grid_sample)           [torch]
    l2sp.py  ema.py          # anti-forgetting + weight EMA                       [torch]
  datasets/
    recording.py             # flight-recording schema + reader + synth          [numpy]
    esdf_label_gen.py        # offline: recording → per-frame labels + SDF        [numpy CLI]
    flight_dataset.py        # torch Dataset → per-model samples                  [torch]
    bag_extract.py           # XTEND rosbag → synced rgb/depth pairs             [numpy CLI]
  tests/                     # numpy tests (run in .venv) + torch tests

tasks/planning/vlas/navdp/finetune/            # ← NavDP-specific
  loss.py  finetune_model.py  train.py
  configs/navdp_finetune.yaml
  verify/                    # click-a-pixel UI: see the training signal before training
  eval/                      # trained-vs-untrained comparison + report
  pixel_goal/                # the pose-free pixel-goal fine-tune variant
  world_goal/                # the map-supervised pipeline — see below

tasks/planning/vlas/flownav/finetune/          # ← FlowNav-specific
  loss.py  finetune_model.py  train.py
  configs/flownav_finetune.yaml
```

`verify/` and `eval/` sit under `navdp/` rather than here because they run NavDP
inference (TensorRT / `TorchNavDP`) — they are NavDP tooling, not shared machinery.

**Data flow:** flight recording → (offline, numpy) PF/ESDF labels + SDF grids →
(training, torch) forward → `L_bc + λe·L_esdf (+ L_critic)` + L2-SP, EMA. Checkpoints
stay in the native model format so each policy's `trt/export/` consumes them unchanged.

The numpy half is validated end-to-end (`pytest tests/` → 25 passed); the torch half
compiles and its unit tests run in the model conda env.

---

## When a surveyed map exists, use `navdp/finetune/world_goal/` instead

Everything above builds its supervision from **one depth frame**: the occupancy
is what that frame can see, the ESDF is derived from it, and the label is NavDP's
own trajectory pushed off those walls. That is the right design when all you have
is a monocular recording, and it is still the path for the real XTEND bags.

For the PEGASUS simulator there is a **surveyed ground-truth map of the whole
building**, and `navdp/finetune/world_goal/` uses it instead. Three consequences,
each of which this package structurally cannot provide:

* goals are drawn from the map's free/clear/landable cells rather than
  back-projected from a pixel, so a goal can never land on an obstacle;
* the label is an independent expert (global A* + medial-axis centring), not the
  student's own output corrected — so it can teach turning toward a doorway
  several metres before the camera can see it;
* the clearance term and the evaluation both read the **global** ESDF, so a
  trajectory heading into an unseen wall is penalised, and the ruler is not the
  one the teacher optimised against.

It shares this package's `label_format`, `esdf_penalty`, `l2sp`, `ema` and
`recording` modules. See `navdp/finetune/world_goal/README.md`.

---

## ⚠️ Status & what you must provide

* **No usable flight recording exists yet.** The only bag in the repo
  (`rosbag2_2026_06_02-15_03_21`, 4.6 MB) is a depth-only AprilTag test — **no RGB, no
  intrinsics, no goal**. You must record synchronized flights: RGB + co-registered
  depth (same resolution), fused pose (`/xtend/localization`), static intrinsics, and
  the pursued goal (a body point for NavDP, a target image for FlowNav). See
  `datasets/recording.py` for the exact on-disk schema (`synthesize_recording` builds a
  tiny example).
* **NavDP training is reconstructed, not read** (upstream ships no trainer). The critic
  target and the ×4 metric scale come from the paper + the shipped scheduler — verify
  against `InternRobotics/NavDP` before trusting the critic recipe. The `torch.no_grad`
  guard is bypassed via the export wrappers (`finetune_model.py`).
* **Pin two numbers on hardware:** the camera **pitch** and the **K-vs-P intrinsic**
  choice (use raw K; P over-scales metric distance ×1.27).
* **Ship the no-training guidance first** (rank 1) for immediate wall-avoidance while
  you collect flights for the weight fine-tune.

## Running

```bash
# 1) (numpy, .venv) generate labels from a recording
PYTHONPATH=$PWD .venv/bin/python -m sparx_agency.tasks.planning.vlas.common.finetune.datasets.esdf_label_gen \
    --recording <flight_dir> --out-dir <flight_dir>/labels

# 2) (torch, navdp conda env) fine-tune NavDP
python -m sparx_agency.tasks.planning.vlas.navdp.finetune.train \
    --config sparx_agency/tasks/planning/vlas/navdp/finetune/configs/navdp_finetune.yaml \
    --recording <flight_dir> --ckpt <navdp-cross-modal.ckpt> --navdp-repo $NAVDP_REPO

# 3) (torch, flownav_trt conda env) fine-tune FlowNav
python -m sparx_agency.tasks.planning.vlas.flownav.finetune.train \
    --config sparx_agency/tasks/planning/vlas/flownav/finetune/configs/flownav_finetune.yaml \
    --recording <flight_dir> --ckpt <flownav_weights.pth> --flownav-repo $FLOWNAV_REPO
```
