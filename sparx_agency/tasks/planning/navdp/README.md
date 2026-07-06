# NavDP point-goal — TensorRT build + inference infrastructure

Optimizes the NavDP cross-modal point-goal policy for TensorRT on a Jetson AGX
Orin — best FPS with minimal loss of network capacity — and provides a
drop-in TRT inference server that honors the existing HTTP contract, so
`navdp_click_node.py` and the core `NavDPPointgoalClient` run **unchanged**.

The PyTorch model decomposes into **three TensorRT engines** plus a numpy control
loop:

| engine | what | shapes (static) | runs |
|---|---|---|---|
| `navdp_encoder` | 2× DINOv2 ViT-S (8 RGB memory frames + 1 depth frame) + Q-Former | `images (1,8,3,224,224)`, `depth (1,1,224,224)` → `rgbd_embed (1,128,384)` | 1× |
| `navdp_denoise` | one DDPM denoise step (16-layer decoder, causal) | `last_actions (16,24,3)`, `time_token (16,1,384)`, `goal_embed (16,1,384)`, `rgbd_embed (16,128,384)` → `noise_pred (16,24,3)` | 10× |
| `navdp_critic` | trajectory critic (cross-attn cond mask) | `predict_trajectory (16,24,3)`, `rgbd_embed (16,128,384)` → `critic (16,1)` | 1× |

Everything stochastic / data-dependent stays in numpy (`core.planning.navdp.trt`):
the DDPM scheduler, the `sample_num=16` fan-out, the `cumsum(/4)`, the `<0.5`
zeroing, and the critic ranking — so the result matches the PyTorch reference up
to the engines' precision.

## Where the code lives

- **`core/planning/navdp/trt/`** — the runtime. ROS-free, **numpy-only at
  import**, Python-3.8 compatible (the FALCON Noetic adapter imports `core` under
  3.8). `tensorrt`/`pycuda` are lazy-imported. `NavDPTRTPolicy` is the drop-in
  for `NavDP_Policy.predict_pointgoal_action`.
- **`tasks/planning/navdp/`** — the builder + server (this directory). Imports
  torch / onnx / tensorrt / the external NavDP repo; **dev/host only**, never
  imported by `core`.

## Two-stage build (engines are SM + TensorRT-build locked → build per device)

The exported ONNX is portable; the built `.engine` is **not** (it deserializes
only on the exact GPU compute capability + TensorRT build that wrote it). So:

### Stage 1 — export ONNX (once; any box with torch — x86 dev box *or* the Jetson)

Export runs on **CPU** and is hardware-agnostic, so it can be done on the Jetson
itself; it does not need CUDA or TensorRT. It does, however, import the *real*
NavDP model: `build_policy` loads the external repo's `policy_network.py`, which
pulls in `diffusers` (DDPM scheduler) and, via `policy_backbone` →
`depth_anything/.../dpt.py`, `torchvision` + `opencv`. A bare `torch`-only venv
is missing these — that is the `ModuleNotFoundError: No module named 'diffusers'`.

```bash
# 1) NavDP model deps the exported graphs are built from. `onnx` is required
#    (the export op-gate uses its pure-Python checker).
pip install diffusers transformers onnx opencv-python

# torchvision must match the installed torch. On Jetson do NOT use the PyPI wheel
# (it drags in an x86/CPU torch). Use the matched aarch64 wheel from the Jetson AI
# Lab index and --no-deps so it can't replace torch. Example (verified): JetPack
# 6.2 / L4T R36.4 with torch 2.9.1+cu126 -> torchvision 0.24.1:
#   pip install --no-deps torchvision==0.24.1 \
#       --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
#   pip install pillow                      # tv runtime dep skipped by --no-deps
# (torch 2.x <-> torchvision: 2.9->0.24, 2.8->0.23, 2.7->0.22; pick the matching cuXXX.)

# verify the model stack imports AND torch is untouched before exporting:
python -c "import torch, torchvision, cv2, diffusers, onnx; \
print('model deps OK; torch', torch.__version__, 'tv', torchvision.__version__)"

# Optional graph-simplify + FP32 parity tooling. x86 dev box ONLY:
#   pip install onnxslim onnxruntime
# Do NOT install these on a Jetson — onnxruntime's CPU-feature detection fails on
# aarch64 ("Unknown CPU vendor") and SIGABRTs; onnxslim calls it during export and
# crashes the whole run. The export auto-skips onnxslim if it is absent, but a
# `--system-site-packages` venv may still see a system onnxslim, so pass
# `--no-slim` (below) to skip it unconditionally. Run the onnxruntime-based
# `validate_parity` on x86; on the Jetson the Stage-2 bench gate (TRT vs the torch
# reference) is the accuracy proof.

export NAVDP_REPO=~/GIT/NavDP/baselines/navdp   # dir containing policy_network.py
export PYTHONPATH=<repo-root>                  # dir containing sparx_agency/

python -m sparx_agency.tasks.planning.navdp.export.export_onnx \
    --ckpt   $NAVDP_REPO/checkpoints/navdp-cross-modal.ckpt \
    --navdp-repo $NAVDP_REPO \
    --out-dir sparx_agency/tasks/planning/navdp/engines/onnx \
    --no-slim          # Jetson/aarch64: skip the onnxslim pass (it SIGABRTs)

# authoritative numeric proof (FP32, CPU EP, deterministic) -- x86 ONLY,
# needs onnxruntime (skip on Jetson; see the note above):
python -m sparx_agency.tasks.planning.navdp.export.validate_parity \
    --onnx-dir sparx_agency/tasks/planning/navdp/engines/onnx \
    --ckpt $NAVDP_REPO/checkpoints/navdp-cross-modal.ckpt --navdp-repo $NAVDP_REPO
```

This writes the three `.onnx`, a `manifest.json`, and `navdp_head_params.npz`
(point-encoder weights, the 10-row sinusoidal time table, `alphas_cumprod`).

### Stage 2 — build engines + gate (on the target device, in its TRT venv)

Run with the **same python `tensorrt` the server imports** (engines are locked to
the build). Hardware is auto-detected (x86 dGPU vs Jetson Orin: power mode, DLA,
memory, compute capability → `target_tag`, e.g. `orin_sm87`).

```bash
# 0) TensorRT + pycuda must be importable from THIS python (engines deserialize
#    only on the exact TRT build that wrote them). JetPack's TensorRT is a system
#    apt package, not pip-installable -- create the venv with --system-site-packages
#    (or use the system python). pycuda may still need `pip install pycuda`.
python -c "import tensorrt, pycuda.autoinit; print('TRT', tensorrt.__version__)"

# 1) build (use the REAL onnx dir from Stage 1, not a literal '...'):
python -m sparx_agency.tasks.planning.navdp.engine.build_engine \
    --onnx-dir sparx_agency/tasks/planning/navdp/engines/onnx --precision fp16
# -> engines/<target_tag>/navdp_{encoder,denoise,critic}.fp16.engine (+ .json,
#    and navdp_head_params.npz is copied in for the gate/server)

# 2) FPS + accuracy gate; picks the precision and writes selected.json:
python -m sparx_agency.tasks.planning.navdp.benchmark.bench \
    --engine-dir sparx_agency/tasks/planning/navdp/engines/<target_tag> \
    --ckpt $NAVDP_REPO/checkpoints/navdp-cross-modal.ckpt --navdp-repo $NAVDP_REPO
```

The gate is **mandatory**: it writes `selected.json` (chosen precision + engine
filenames), and the Stage-3 server fails loud if that file is absent — it never
guesses a precision. If no precision passes the gate, `bench` raises instead of
shipping a bad engine.

INT8 is an optional stretch: build with `--precision int8 --calib-npz <frames>`
and it is selected **only if** it clears a stricter on-device accuracy gate
(`argmax`-flip / stop-decision / `<0.5`-zeroing vs the FP32 torch reference).

#### Power mode matters more than workspace

The builder logs e.g. `workspace=1.0GiB ... opt-target=orin_sm87`. That **1 GiB is
a function of the active `nvpmodel` power mode, not your RAM**: the Jetson workspace
cap (`hardware/detect.py:_workspace_bytes`) is 1 GiB at ≤15 W, 2 GiB above it, then
`min(cap, total_mem/4)` — small on purpose because Jetson GPU memory is unified with
the CPU. Workspace is only a tactic-selection scratch ceiling; for these small
DINOv2 ViT-S graphs 1–2 GiB rarely changes the chosen kernels. The real throughput
lever is the **power mode itself**. For best FPS on a 64 GB AGX:

```bash
sudo nvpmodel -q          # show the active mode (a *W mode => 1 GiB cap)
sudo nvpmodel -m 0        # MAXN (id may vary; check -q). Then:
sudo jetson_clocks        # pin GPU/CPU/EMC clocks to max
```

**Build and run the gate in the same power mode you will fly in** — the engine's
tactics are tuned to the clocks present at build time.

#### TensorRT 10 vs 11 (the builder handles both)

The Orin (JetPack) ships **TensorRT 10.x**, which has weak typing: `BuilderFlag.FP16`
lets TensorRT mix FP16/FP32 per layer and accumulate matmuls in FP32 — so all three
graphs build FP16 and stay accurate. **TensorRT 11** (what `pip install tensorrt`
currently pulls) **removed weak typing**: precision comes from the network types via
a *strongly-typed* network, so the builder feeds it an FP16-converted ONNX. The
catch: the deep DINOv2 ViT encoder drifts badly when its 24-block residual stream is
*forced* to FP16 (no per-layer fallback) — measured ~0.69 rel-L2, which flips the
critic ranking. So on the strongly-typed path the **encoder is built FP32** and only
denoise/critic FP16 (`configs/build_policy.json: strongly_typed_fp32_engines`); the
TRT-10 Orin build ignores that list and gets mixed FP16 on all three.

Validated on x86 (RTX 5070, sm_120, TRT 11): encoder FP32 + denoise/critic FP16
passes the gate at **0% argmax-flip**, ~20 Hz end-to-end, and a full client
round-trip returns a 24-waypoint trajectory. The Orin's all-FP16 TRT-10 build should
match or beat this — re-run the gate there to confirm before flying.

### Engine variants & A/B comparison

The denoiser inner loop is the **measured bottleneck** (~71% of per-decision
compute: ~3.9 ms/step × 10 steps vs ~12 ms for the encoder, on the 5070). So
denoiser variants are worth A/B-testing. Export a variant alongside the baseline,
build them together, and compare:

```bash
# 1) export the baseline + a tgt_is_causal denoiser variant
python -m sparx_agency.tasks.planning.navdp.export.export_onnx \
    --ckpt .../navdp-cross-modal.ckpt --navdp-repo $NAVDP_REPO \
    --out-dir .../engines/onnx --with-causal-denoise

# 2) build all engines found (encoder, denoise, denoise_causal, critic)
python -m sparx_agency.tasks.planning.navdp.engine.build_engine \
    --onnx-dir .../engines/onnx --precision fp16

# 3) A/B every navdp_denoise*.engine: step latency + Hz + accuracy gate,
#    writes selected.json to the fastest variant that still passes the gate
python -m sparx_agency.tasks.planning.navdp.benchmark.compare_engines \
    --engine-dir .../engines/<target_tag> --ckpt ... --navdp-repo ...
```

**`navdp_denoise_causal`** is the one concrete optimization from an alternative
export script (`tgt_is_causal=True`, a causal-attention kernel hint) — but with
our FP16-safe `-1e4` mask instead of its `-inf` (which risks NaN in FP16).
Measured on the 5070 it was **identical accuracy (0% flip) and the same speed**
(3.09 vs 3.08 ms/step) — the masked self-attention is only 24 tokens, so the hint
doesn't move the needle. Re-measure on the Orin; it is unlikely to differ.

The genuinely high-leverage win is **fewer sampler steps** (DDIM / DPM-Solver at
2–4 steps vs 10 DDPM steps) — that attacks the 71% directly. It is a runtime
change (same denoise engine, fewer calls), not an engine variant, and is not yet
implemented.

### Stage 3 — run the drop-in server (FALCON host)

The FALCON Noetic container has no TensorRT, so the server is a **host process**;
FALCON reaches it over `--network host` + `127.0.0.1:<port>` loopback (run it on
the FALCON host).

The TRT server needs `tensorrt` + `pycuda` (the same ones that built the engines)
plus `flask opencv-python pillow numpy`. `--navdp-repo` (or `NAVDP_REPO`) **is**
required even on the default `trt` backend: the agent inherits `NavDP_Agent`'s image
preprocessing, which is imported from that repo on the first navigate request (the
server still *starts* without it, then fails on the first reset). It is only the
`torch` fallback that additionally needs `--ckpt`.

The engine directory name is the device's hardware tag (`hardware/detect.py`), so
the same checkout has a different `--engine-dir` per machine. Pick the one that
matches where you're running — engines are **not** portable across devices.

```bash
# --- AGX Orin (the Jetson at the office) -------------------------------------
# Run in MAXN + jetson_clocks, the same power mode the engines were built in.
export NAVDP_REPO=~/GIT/NavDP/baselines/navdp/
cd ~/GIT/TheAgency
PYTHONPATH=$PWD python \
    -m sparx_agency.tasks.planning.navdp.server.navdp_trt_server \
    --engine-dir sparx_agency/tasks/planning/navdp/engines/orin_sm87 \
    --navdp-repo "$NAVDP_REPO" \
    --port 8888

# --- x86 dev box (RTX 5070 / sm120) ------------------------------------------
cd ~/GIT/TheAgency
NAVDP_REPO=${NAVDP_REPO:?set NAVDP_REPO to the NavDP repo (has policy_agent.py)} \
PYTHONPATH=$PWD python \
    -m sparx_agency.tasks.planning.navdp.server.navdp_trt_server \
    --engine-dir sparx_agency/tasks/planning/navdp/engines/nvidiageforcertx_sm120 \
    --navdp-repo "$NAVDP_REPO" \
    --port 8888
# navdp_click_node.py points at it unchanged (~port 8888)
```

`--backend trt` (default) fails loud if engines are missing/locked;
`--backend torch` (or `--allow-torch-fallback`) runs the original model for
comparison. The image/pixel/nogoal routes return **501** (point-goal only).

## Accuracy strategy (minimal capacity loss)

- **FP16-first.** Near-lossless for ViT/transformers on Orin; sensitive layers
  (LayerNorm, heads, attention softmax, encoder pos-embed) are pinned high under
  INT8.
- The DINOv2 bicubic positional-embedding interpolation is **pre-baked** (the
  `Resize` is deleted from the graph) and `-inf` attention masks become finite
  `-1e4` (FP16-safe) — both blessed by `validate_parity`.
- The gate measures **decision** failures (which trajectory is executed, stop
  decision, the zeroing of the chosen sample), not raw MSE, with identical
  injected noise on both sides.

## What is validated where

- **At home (x86):** the whole numpy runtime (`core/.../trt`, unit-tested
  torch-free), the ONNX export + FP32 parity (authoritative), hardware detection,
  and — with an x86 TRT venv — the FP16 engine build + FPS/gate for that GPU.
- **On the Orin (must wait):** the Orin engines (home `.engine` is invalid there),
  the on-device FPS numbers (at the chosen `nvpmodel` power mode), INT8
  calibration/blessing, and the FALCON loopback integration.

## File map

```
core/planning/navdp/trt/      engine_runner · scheduler · point_encoder · postprocess · policy · errors  (+ tests, golden .npz)
tasks/planning/navdp/
  export/    wrappers · build_policy · export_onnx · validate_parity · io_spec · gen_scheduler_golden
  hardware/  detect
  engine/    inspect_onnx · calibrator · build_engine · fp16_onnx
  benchmark/ bench · compare_engines
  server/    trt_agent · navdp_trt_server
  configs/   build_policy.json          engines/  (gitignored output)
```

Config knobs (precision pins, optimization levels, gate thresholds, parity
tolerances) live in `configs/build_policy.json`.
