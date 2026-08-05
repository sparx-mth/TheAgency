# FlowNav — TensorRT build + run (x86 **and** Jetson)

Optimizes the [FlowNav](https://github.com/utn-air/flownav) image-goal navigation
policy for TensorRT — best FPS with minimal loss of network capacity — and lets us
run it **with and without** optimization to compare runtime and quality. Mirrors
the NavDP TRT infrastructure (`tasks/planning/vlas/navdp`): a torch-free numpy runtime
in `core`, the torch/onnx/trt builder here in `tasks`.

This README is a complete runbook for **both** an x86 dev box (RTX 5070, sm120,
TensorRT 11) **and** a Jetson AGX Orin (sm87, JetPack / TensorRT 10). Engines are
**not portable across devices** — you build on each target.

---

## TL;DR

```bash
# (once, per machine) build env -- see "0. Build environment" for your platform
# then, from the repo root, with PYTHONPATH set:
export PYTHONPATH=$PWD
export FLOWNAV_REPO=~/PycharmProjects/flownav
export CKPT=$FLOWNAV_REPO/flownav/checkpoints/flownav_weights.pth
ONNX=sparx_agency/tasks/planning/vlas/flownav/trt/engines/onnx

# Stage 1 (any box w/ torch, CPU ok) -- export + FP32 parity
python -m sparx_agency.tasks.planning.vlas.flownav.trt.export.export_onnx   --ckpt $CKPT --flownav-repo $FLOWNAV_REPO --out-dir $ONNX   # add --no-slim on Jetson
python -m sparx_agency.tasks.planning.vlas.flownav.trt.export.validate_parity --onnx-dir $ONNX --ckpt $CKPT --flownav-repo $FLOWNAV_REPO

# Stage 2 (ON the target device, its TRT venv) -- build + benchmark/select
python -m sparx_agency.tasks.planning.vlas.flownav.trt.engine.build_engine --onnx-dir $ONNX --precision fp16
TAG=$(python -c "from sparx_agency.tasks.planning.vlas.common.hardware.detect import detect; print(detect().target_tag)")
python -m sparx_agency.tasks.planning.vlas.flownav.trt.benchmark.bench   --engine-dir sparx_agency/tasks/planning/vlas/flownav/trt/engines/$TAG   --ckpt $CKPT --flownav-repo $FLOWNAV_REPO
```

---

## Architecture — 3 engines + a numpy loop

FlowNav is a NoMaD-derived model that swaps DDPM diffusion for **flow matching**:
the action trajectory is produced by integrating a learned velocity field with an
explicit Euler ODE solver over only a few steps. It decomposes into three engines:

| engine | what | static shapes | runs |
|---|---|---|---|
| `flownav_encoder` | `NoMaD_ViNT`: EfficientNet-B0 (obs) + EfficientNet-B0 (obs+goal) + a frozen DINOv2 / DepthAnythingV2-ViT-S "depth prior" on the current RGB frame + 4-layer self-attention | `obs_img (1,12,96,96)`, `goal_img (1,3,96,96)` → `obsgoal_cond (1,256)` | 1× |
| `flownav_vfield` | one velocity-field eval (`ConditionalUnet1D`, continuous-time sinusoidal embedding inside the graph) | `sample (N,8,2)`, `timestep (1,)`, `global_cond (N,256)` → `vfield (N,8,2)` | **K−1×** |
| `flownav_dist` | temporal-distance head (small MLP) | `obsgoal_cond (1,256)` → `distance (1,1)` | 1× |

Everything stochastic / data-dependent stays in numpy (`core.planning.vlas.flownav.trt`):
the initial noise `x0 ~ N(0,I)`, the **deterministic Euler integration**
`x ← x + dt·vfield`, and the action de-normalization (`(d+1)/2·(max−min)+min` then
`cumsum`). No critic ranking — the reference executes sample 0; the `dist` head is
only for topomap-node localization.

**The "K" (the speed lever).** `K = num_steps` is the `linspace(0,1,K)` Euler grid;
the velocity field runs **K−1** times, so K dominates latency. Flow matching stays
accurate at low K, so the benchmark sweeps K and picks the **smallest** K whose
trajectory stays within tolerance of the high-K reference **and** whose engines
pass the fidelity gate. K is a *runtime* knob (no rebuild).

**Depth note (DA2 vs DA3).** FlowNav's "depth prior" is a frozen DINOv2
(DepthAnythingV2-ViT-S) run on the **RGB** frame (`obs_img[:, 9:]`); it has **no
depth-map input port** and its weights are baked into the checkpoint. So it cannot
consume DA3's depth — it is exported as part of `flownav_encoder`.

## Where the code lives

- **`core/planning/vlas/flownav/trt/`** — the runtime. ROS-free, **numpy-only at
  import**, Python-3.8 compatible (FALCON's Noetic adapter imports `core` under
  3.8). `tensorrt`/`pycuda` lazy-imported. `FlowNavTRTPolicy` runs inference.
- **`tasks/planning/vlas/flownav/`** — the builder + benchmark (here). Imports torch /
  onnx / tensorrt / the FlowNav model; **dev/host only**, never imported by `core`.

---

## Platform matrix

| | **x86 dev box** | **Jetson AGX Orin** |
|---|---|---|
| GPU / SM | RTX 5070, **sm120** | Orin iGPU, **sm87** |
| `target_tag` (engine dir) | `nvidiageforcertx_sm120` | `orin_sm87` |
| TensorRT | **11.x** (pip), strongly-typed → FP16 via FP16-ONNX | **10.x** (JetPack apt), weak-typed `BuilderFlag.FP16` |
| TRT env | conda clone of `navdp` | `--system-site-packages` venv (TRT is a system pkg) |
| `onnxslim` in export | ok | **skip** (`--no-slim`; onnxruntime SIGABRTs on aarch64) |
| Power | always-on | **set `nvpmodel` + `jetson_clocks`** before build & run |
| Memory | dedicated VRAM (8 GB here) | unified with CPU → small workspace cap |

The builder auto-detects all of this (`hardware/detect.py`) and dispatches the
TRT-10 vs TRT-11 path automatically (`engine/build_engine.py`). You do **not** pass
a platform flag — just run on the target.

---

## 0. Build environment

Both platforms need: **torch + tensorrt + pycuda + onnx**, the **FlowNav repo**, and
FlowNav's pinned deps (`efficientnet-pytorch`, `einops`, `torchdiffeq`) **+ the two
debOliveira forks** (`depth_anything_v2`, `diffusion_policy`). **xformers must be
ABSENT** (else DINOv2 attention exports to a non-traceable `memory_efficient_attention`).

### 0a. x86 (RTX 5070 / sm120) — clone the NavDP TRT env

```bash
conda create --clone navdp -n flownav_trt -y          # torch 2.11+cu128, TRT 11.1.0.106, pycuda, onnx*
PY=/home/$USER/miniconda3/envs/flownav_trt/bin/python
```

### 0b. Jetson AGX Orin (sm87 / JetPack) — system-site-packages venv

JetPack ships TensorRT as a system apt package (not pip-installable), so the venv
must see system site-packages:

```bash
python3 -m venv --system-site-packages ~/venvs/flownav_trt
PY=~/venvs/flownav_trt/bin/python
$PY -m pip install pycuda                              # if not already present
# torch/torchvision: use the matched aarch64 wheels (Jetson AI Lab index), NOT PyPI
#   (PyPI drags in an x86 CPU torch). Example for JetPack 6.x / torch 2.x:
#   $PY -m pip install --no-deps torchvision==<matched> --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
$PY -c "import tensorrt, pycuda, torch; print('TRT', tensorrt.__version__, '| torch', torch.__version__)"
# Do NOT pip install onnxruntime/onnxslim on the Jetson (they SIGABRT on aarch64);
# run Stage-1 parity on x86 instead, and use --no-slim in Stage 1 here.
```

### 0c. FlowNav deps + forks (BOTH platforms)

```bash
$PY -m pip install --no-deps efficientnet-pytorch==0.7.1 einops==0.8.1 torchdiffeq==0.2.5
$PY -m pip install --no-deps \
    "git+https://github.com/debOliveira/depth-anything-V2.git@7885bbc0647bc64d55ff5803561ea2c7dea1af72" \
    "git+https://github.com/debOliveira/diffusion_policy.git@db1434cc256b53deb0ad7228c129c0ce7c733822"
$PY -c "import torch,tensorrt,pycuda,onnx,efficientnet_pytorch,einops,torchdiffeq, \
        depth_anything_v2.dinov2, diffusion_policy.model.diffusion.conditional_unet1d; \
        import importlib.util as u; assert u.find_spec('xformers') is None, 'remove xformers'; print('deps OK')"
```

---

## 1. Stage 1 — export ONNX + FP32 parity

Runs on **CPU** and is hardware-agnostic (the ONNX is portable). Do it once on the
x86 box; you can copy `engines/onnx/` to the Jetson, or re-run it there with
`--no-slim`.

```bash
export PYTHONPATH=$PWD                                  # repo root (has sparx_agency/)
export FLOWNAV_REPO=~/PycharmProjects/flownav
export CKPT=$FLOWNAV_REPO/flownav/checkpoints/flownav_weights.pth
ONNX=sparx_agency/tasks/planning/vlas/flownav/trt/engines/onnx

# x86:
$PY -m sparx_agency.tasks.planning.vlas.flownav.trt.export.export_onnx \
    --ckpt $CKPT --flownav-repo $FLOWNAV_REPO --out-dir $ONNX
# Jetson: add --no-slim
#   $PY -m ...export_onnx --ckpt $CKPT --flownav-repo $FLOWNAV_REPO --out-dir $ONNX --no-slim

# Authoritative FP32 proof (x86 only -- needs onnxruntime):
$PY -m sparx_agency.tasks.planning.vlas.flownav.trt.export.validate_parity \
    --onnx-dir $ONNX --ckpt $CKPT --flownav-repo $FLOWNAV_REPO
```

Writes the three `.onnx`, `flownav_head_params.npz` (action min/max), and a
`manifest.json`. The export op-gate fails loud if a `Resize` (un-baked DINOv2
pos-embed) or a fused `*Attention` op survives.

## 2. Stage 2 — build engines + benchmark/select (ON the target device)

Run with the **same python `tensorrt` the runtime imports**. Hardware is
auto-detected; the engine dir is named for the device (`<target_tag>`).

```bash
ONNX=sparx_agency/tasks/planning/vlas/flownav/trt/engines/onnx
$PY -m sparx_agency.tasks.planning.vlas.flownav.trt.engine.build_engine --onnx-dir $ONNX --precision fp16

TAG=$($PY -c "from sparx_agency.tasks.planning.vlas.common.hardware.detect import detect; print(detect().target_tag)")
$PY -m sparx_agency.tasks.planning.vlas.flownav.trt.benchmark.bench \
    --engine-dir sparx_agency/tasks/planning/vlas/flownav/trt/engines/$TAG \
    --ckpt $CKPT --flownav-repo $FLOWNAV_REPO
```

**Jetson: set the power mode first** — the engine tactics are tuned to the clocks
present at build time, so build and fly in the same mode:

```bash
sudo nvpmodel -m 0      # MAXN (id may vary; check `sudo nvpmodel -q`)
sudo jetson_clocks      # pin GPU/CPU/EMC clocks
```

The benchmark is **mandatory**: it writes `selected.json` (precision + engine
filenames + chosen **K** + N), which `FlowNavTRTPolicy` requires (it never
guesses). It prints, per K: TRT latency, eager-torch latency, the speed-up, the
TRT-vs-torch fidelity, and the low-K quality drift vs the high-K reference — so the
chosen K is data-driven. INT8 is intentionally **not** wired (don't over-quantize
on top of low-K flow matching; FP16 is the validated default).

## 3. Run with vs without TensorRT

- **With TRT** (optimized):
  ```python
  from sparx_agency.core.planning.vlas.flownav.trt import FlowNavTRTPolicy
  policy = FlowNavTRTPolicy(engine_dir, head_params_npz)      # K + N come from selected.json
  actions, distance = policy.predict(obs_img, goal_img)        # actions: (N,8,2)
  ```
- **Without TRT** (baseline): `build_flownav_model(ckpt, flownav_repo)` then the
  eager-torch path (the `bench.py` `TorchReference`, or the reference
  `torchdiffeq.odeint` loop). The benchmark measures both and reports the speed-up.
- **Live FALCON server, either backend** (same routes -> the node + window are
  unchanged; just swap how you start the server):
  ```bash
  # optimized (engines):   ... --engine-dir .../engines/<target_tag>
  # UNOPTIMIZED (eager torch):
  python -m sparx_agency.tasks.planning.vlas.flownav.serve.flownav_trt_server --backend torch \
      --ckpt $CKPT --flownav-repo $FLOWNAV_REPO --goal-image ~/Downloads/goal_image.jpg
  ```
  `server/torch_policy.FlowNavTorchPolicy` is a drop-in twin of `FlowNavTRTPolicy`
  (same numpy Euler loop + de-normalization; only the 3 forward passes differ).

`obs_img` / `goal_img` are produced by FlowNav's `transform_images` (resize to
96×96, ToTensor, ImageNet-normalize, context frames concatenated on the channel
axis → 12 channels).

---

## Measured results (x86 / RTX 5070 Laptop, sm120, TensorRT 11.1, FP16)

Built and benchmarked end-to-end (16 scenarios). FP32 parity first: encoder
`rel_l2=1.3e-06`, vfield `3.0e-07`, dist `1.0e-07`, numpy-Euler vs torchdiffeq
`0.0`. Then TRT vs eager-torch, per K:

| K | TRT | eager torch | speed-up | TRT-vs-torch waypoint err | drift vs K=10\* |
|---|---|---|---|---|---|
| 2 | 1.84 ms (543 Hz) | 10.8 ms | **5.9×** | 5.7 mm | 0.29 m |
| 3 | 3.44 ms | 12.8 ms | 3.7× | 5.0 mm | 0.23 m |
| **4** (selected) | 4.00 ms (250 Hz) | 14.8 ms | 3.7× | 4.6 mm | 0.18 m |
| 5 | 4.19 ms | 17.2 ms | 4.1× | 3.7 mm | 0.14 m |
| 8 | 5.51 ms | 22.4 ms | 4.1× | 3.9 mm | 0.04 m |
| 10 | 6.55 ms | 25.5 ms | 3.9× | 3.8 mm | 0.00 m (ref) |

The **TRT FP16 engines are near-lossless vs FP32 torch at every K** (waypoint error
~4–6 mm), so no encoder-FP16 drift. K is a runtime knob: the default is **K=4**
(honoring "relatively low K"); set lower (K=2 → ~6× and 540 Hz) for max speed, or
higher for max fidelity-to-10-step — no rebuild, just `FlowNavTRTPolicy(num_steps=K)`
or edit `num_steps` in `selected.json`.

\* The drift column is measured on **synthetic random inputs**, so it is only a
coarse proxy for real low-K quality — re-evaluate on real images during integration
before committing to a final K.

## Output layout (`engines/`)

```
engines/
  onnx/                                  # portable across devices
    flownav_encoder.onnx  flownav_vfield.onnx  flownav_dist.onnx
    flownav_head_params.npz  manifest.json
  nvidiageforcertx_sm120/                # x86 build (device-locked)
    flownav_{encoder,vfield,dist}.fp16.engine (+ .engine.json manifests)
    flownav_head_params.npz  timing_*.cache  selected.json  bench_report.json
  orin_sm87/                             # Jetson build (device-locked)
    flownav_{encoder,vfield,dist}.fp16.engine (+ .json, npz, caches, selected.json, ...)
```

`FlowNavTRTPolicy(engine_dir=engines/<target_tag>, head_params_npz=engines/<target_tag>/flownav_head_params.npz)`.

## Running with FALCON (image-goal, `vla:=flownav`)

FALCON's local-planner slot serves either VLA via one launch arg, **`vla`**:
`nav_mode:=navdp vla:=navdp` (NavDP point-goal — click a pixel) or
`nav_mode:=navdp vla:=flownav` (FlowNav image-goal — steer toward a target image).
Switching is just the `vla` arg + starting the matching host server. Because the
FALCON Noetic container has no TensorRT, FlowNav inference runs in a **host
process** over loopback HTTP, exactly like NavDP.

```bash
# 1) Start the FlowNav TRT host server (GPU host, flownav_trt env). Pass your
#    target image with --goal-image: the HOST can read paths the container can't
#    mount (e.g. ~/Downloads), so the in-container node needs no goal file at all.
PY=/home/$USER/miniconda3/envs/flownav_trt/bin/python \
    sparx_agency/tasks/planning/vlas/flownav/serve/run_server.sh --goal-image ~/Downloads/goal_image.jpg
#   (or directly:)
#   PYTHONPATH=$PWD $PY -m sparx_agency.tasks.planning.vlas.flownav.serve.flownav_trt_server \
#       --engine-dir sparx_agency/tasks/planning/vlas/flownav/trt/engines/<target_tag> \
#       --port 8889 --goal-image ~/Downloads/goal_image.jpg

# 2) Launch FALCON with the FlowNav VLA (NO goal needed here — the server has it):
roslaunch falcon_adapter real_drone.launch \
    map_name:=office nav_mode:=navdp vla:=flownav
#   optional: flownav_arrival_distance:=3.0  -> hold once the goal-distance head
#   drops below 3.0 (watch the node's "goal-distance N.NN" log to pick a value).
# back to NavDP: drop vla:=flownav and start the NavDP server instead.
```

A good goal image is a **view from the destination** — what the drone's own camera
would see once it has arrived (same camera / aspect ratio helps). The goal can
instead be set on the node (`flownav_goal_image:=<container-visible path>`), live
via `flownav_goal_image_topic`, or swapped at runtime with the server's `/set_goal`
route — but `--goal-image` on the host is the simplest given the container mount.

Responsibilities — one per file, so the navdp ⇄ flownav switch is trivial:

| file | responsibility |
|---|---|
| `core/planning/vlas/flownav/client.py` | loopback HTTP wire client (`FlowNavImageGoalClient`); numpy-only, container-safe |
| `core/planning/vlas/flownav/preprocess.py` | RGB frames → model tensors (`transform_images` parity) |
| `tasks/planning/vlas/flownav/serve/flownav_trt_server.py` | host TRT inference service (rolling frame buffer + `FlowNavTRTPolicy`) |
| `adapter/scripts/flownav_node.py` | in-container ROS node: RGB + goal image → body waypoints → world `nav_msgs/Path` |
| `launch/nav_stack.launch` (`vla` arg) | selects `navdp_click_node` vs `flownav_node` and routes the corrector input |

**Identical to NavDP downstream** (so nothing else changes): the node publishes a
latched world-frame `nav_msgs/Path` on `/path/waypoints_flownav` →
`path_corrector_node` → `trajectory_simplifier_node` → `waypoint_follower_node`,
unchanged. It reuses `anchor_trajectory_to_world`, the frame-path transport, and
the pose handling.

**The only new input:** the goal is a **target image** (`flownav_goal_image:=<file>`,
or a live `flownav_goal_image_topic`) instead of a clicked point; FlowNav runs
**continuously** at `flownav_rate_hz` (reactive) and needs no depth or intrinsics.

## Troubleshooting

- **Export error: `... contains forbidden ops ['Resize']`** — the DINOv2 pos-embed
  pre-bake didn't take (its `interpolate_pos_encoding` signature differs). Check
  `export/wrappers.py:bake_pos_embed`.
- **Export error mentioning a custom autograd Function / `swish`** — EfficientNet's
  memory-efficient swish leaked; `wrappers.py` calls `set_swish(memory_efficient=False)`.
- **`memory_efficient_attention` / xformers in the graph** — uninstall xformers in
  the build env; DINOv2 then uses the plain (exportable) attention.
- **Engine fails to load with an SM / TensorRT-version error** — engines are
  device-locked; rebuild on this machine (the runtime version-locks against the
  `.engine.json` manifest).
- **Jetson `onnxslim`/`onnxruntime` SIGABRT ("Unknown CPU vendor")** — use
  `--no-slim` in Stage 1 and run `validate_parity` on x86.
- **Encoder FP16 drift flagged by the gate (TRT-11/x86)** — add `"flownav_encoder"`
  to `strongly_typed_fp32_engines` in `configs/build_policy.json` (builds the
  encoder FP32, keeps vfield/dist FP16); the Orin's TRT-10 weak-typed path is
  unaffected.

Config knobs (FP-keep layers, optimization levels, the accuracy gate, parity
tolerances, the K sweep + low-K quality bound) live in `configs/build_policy.json`.
