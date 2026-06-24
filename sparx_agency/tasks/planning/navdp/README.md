# NavDP point-goal — TensorRT build + inference infrastructure

Optimizes the NavDP cross-modal point-goal policy for TensorRT on a Jetson AGX
Orin (15 W) — best FPS with minimal loss of network capacity — and provides a
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

### Stage 1 — export ONNX (once, any x86/dev box with torch)

```bash
# in the navdp conda env (torch + the external NavDP repo); add onnx tooling:
pip install onnx onnxruntime onnxslim          # export + parity deps

export NAVDP_REPO=~/PycharmProjects/NavDP/baselines/navdp
export PYTHONPATH=<repo-root>                  # dir containing sparx_agency/

python -m sparx_agency.tasks.planning.navdp.export.export_onnx \
    --ckpt   $NAVDP_REPO/checkpoints/navdp-cross-modal.ckpt \
    --navdp-repo $NAVDP_REPO \
    --out-dir sparx_agency/tasks/planning/navdp/engines/onnx

# authoritative numeric proof (FP32, CPU EP, deterministic):
python -m sparx_agency.tasks.planning.navdp.export.validate_parity \
    --onnx-dir sparx_agency/tasks/planning/navdp/engines/onnx \
    --ckpt $NAVDP_REPO/checkpoints/navdp-cross-modal.ckpt --navdp-repo $NAVDP_REPO
```

This writes the three `.onnx`, a `manifest.json`, and `navdp_head_params.npz`
(point-encoder weights, the 10-row sinusoidal time table, `alphas_cumprod`).

### Stage 2 — build engines + gate (on the target device, in its TRT venv)

Run with the **same python `tensorrt` the server imports** (engines are locked to
the build). Hardware is auto-detected (x86 dGPU vs Jetson Orin 15 W: power mode,
DLA, memory, compute capability).

```bash
python -m sparx_agency.tasks.planning.navdp.engine.build_engine \
    --onnx-dir .../engines/onnx --precision fp16
# -> .../engines/<target_tag>/navdp_{encoder,denoise,critic}.fp16.engine (+ .json)

# FPS + accuracy gate; picks the precision and writes selected.json:
python -m sparx_agency.tasks.planning.navdp.benchmark.bench \
    --engine-dir .../engines/<target_tag> \
    --ckpt $NAVDP_REPO/checkpoints/navdp-cross-modal.ckpt --navdp-repo $NAVDP_REPO
```

INT8 is an optional stretch: build with `--precision int8 --calib-npz <frames>`
and it is selected **only if** it clears a stricter on-device accuracy gate
(`argmax`-flip / stop-decision / `<0.5`-zeroing vs the FP32 torch reference).

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

### Stage 3 — run the drop-in server (FALCON host)

The FALCON Noetic container has no TensorRT, so the server is a **host process**;
FALCON reaches it over `--network host` + `127.0.0.1:<port>` loopback (run it on
the FALCON host).

```bash
python -m sparx_agency.tasks.planning.navdp.server.navdp_trt_server \
    --engine-dir .../engines/<target_tag> --navdp-repo $NAVDP_REPO --port 8888
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
  the @15 W FPS numbers, INT8 calibration/blessing, and the FALCON loopback
  integration.

## File map

```
core/planning/navdp/trt/      engine_runner · scheduler · point_encoder · postprocess · policy · errors  (+ tests, golden .npz)
tasks/planning/navdp/
  export/    wrappers · build_policy · export_onnx · validate_parity · io_spec · gen_scheduler_golden
  hardware/  detect
  engine/    inspect_onnx · calibrator · build_engine
  benchmark/ bench
  server/    trt_agent · navdp_trt_server
  configs/   build_policy.json          engines/  (gitignored output)
```

Config knobs (precision pins, optimization levels, gate thresholds, parity
tolerances) live in `configs/build_policy.json`.
