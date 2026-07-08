# Open-set YOLO-World TensorRT (backbone→DLA + head→GPU, Jetson AGX Orin @ 15 W)

Turns the ultralytics **YOLO-World** checkpoints (`s`/`m`/`l`/`x`) into TensorRT
engines that stay **fully open-vocabulary** while pushing the heavy convolutional
work onto the Jetson AGX Orin **DLA**. You give a prompt (or a list of objects) at
run time — never at build time — and can change it whenever you like, with no
rebuild.

This task owns only the **build + runtime + benchmark**. The detection algorithm
stays ROS-free in `core.mapping.detection`; the TRT engine is exposed through the
same `DetectionModel` ABC (`runtime.YoloTRTDetector`), so `yolo_detector_node`
consumes it unchanged.

---

## 0. The idea (why this stays open-set *and* uses the DLA)

YOLO-World has two data paths with opposite runtime profiles:

```
 prompts ─► [ CLIP text encoder ]* ─► txt_feats [N,512]   (* runs ONLY on re-prompt)
                                          │  cached
 image ─► [ backbone: conv/C2f/SPPF ] ─► feature maps      (static, text-free, EVERY frame)
                                          │
                    txt_feats ───────────┤
                                          ▼
              [ neck+head: RepVL-PAN + WorldDetect ] ─► detections  (text-fused, EVERY frame)
```

Your key insight drives the whole design: **the text side only runs when the
object list changes.** So we never bake the prompts (that would kill open-set).
Instead we split the model in two and feed the text embeddings as a **runtime
input**:

| Engine | Contents | Shapes | Device | Runs |
|---|---|---|---|---|
| **backbone** | CSPDarknet conv/C2f/SPPF | fully **static**, text-free | **DLA** (+GPU fallback) | every frame |
| **head** | RepVL-PAN neck + WorldDetect | **dynamic N** (prompt count) | **GPU** | every frame |
| **text** (`TextEmbedder`) | CLIP text encoder | — | torch (CPU/GPU) | only on re-prompt |

- The **backbone** is the bulk of the compute and is a pure CNN → the part that
  *can* go on DLA. DLA needs static shapes, and the backbone is N-independent, so
  it qualifies cleanly.
- The **head** fuses text and its class dimension follows the (runtime, variable)
  prompt count → **dynamic shape → GPU** (DLA cannot do dynamic shapes). It is the
  lighter part.
- The **text encoder** is expensive and a poor TRT/DLA fit, but it runs *rarely*
  (once per prompt change), so it stays torch and its cost is off the frame path.

Net effect: arbitrary prompts at run time, the conv backbone offloaded to the two
DLA cores (a big win at 15 W where the GPU is the power/thermal bottleneck), and a
per-frame path that touches no torch.

> **"Truly unlimited N":** TensorRT dynamic shapes need a maximum. The head is
> built with a profile `N ∈ [n_min, n_opt, n_max]` (default `1 / 8 / 256`).
> Re-prompt with **≤ n_max** classes needs no rebuild — 256 is far above any real
> mission list. Raise it with `--n-max` / `N_MAX` if you ever need more.

---

## 1. Nothing is left to TensorRT defaults

`hardware.py` probes the board; `build_policy.py` derives an explicit per-role
`BuildPolicy` that `build_engine.py` applies literally:

| Decision | backbone | head | Why |
|---|---|---|---|
| Precision | FP16 | FP16 | DLA runs FP16/INT8 only; FP16 is the floor. INT8 is opt-in (§7). |
| Device | **DLA** core 0 + **GPU fallback** | **GPU** | A few backbone tail ops (SPPF variants, some Slice/Concat) aren't DLA-supported → fallback is mandatory. The head is dynamic → GPU. |
| Shapes | static | dynamic `N` profile | Static is what lets DLA accept the backbone; the head's class dim is the runtime prompt count. |
| DLA pools | SRAM 1 MiB / local 1 GiB / global 512 MiB | — | Explicitly sized (Orin DLA managed-SRAM ≈ 1 MiB/core). |
| Workspace | ≤ ¼ shared RAM @ 15 W | same | LPDDR is shared with the CPU. |
| Opt level | 5 | 5 | Offline build-search knob, **not** runtime power (nvpmodel clamps that). Max search only costs a longer one-time build. |
| Input size | `288×512` (H×W) default | — | Stride-32, matches the 504×294 landscape frame with minimal padding, ~2.8× fewer pixels than 640². `640×640` for max accuracy (§6). |

DLA engines build **only on a Jetson**. On the x86 laptop the builder finds no DLA
and produces a **GPU** backbone engine instead, so the whole pipeline is still
exercisable off-target (the DLA path just isn't taken).

---

## 2. Install

### 2a. Export box (laptop or Orin) — torch + ultralytics + onnx
Export is CPU-only and portable.
```bash
pip install "ultralytics>=8.2" onnx onnxslim      # torch comes with ultralytics
```

### 2b. Target Orin — TensorRT + pycuda (build + runtime + benchmark)
JetPack ships these; use a venv that can see the system packages:
```bash
python3 -m venv --system-site-packages ~/venvs/trt && source ~/venvs/trt/bin/activate
python -c "import tensorrt, pycuda; print(tensorrt.__version__)"
pip install pycuda numpy opencv-python          # if missing
```
> Build **on the target with the same TensorRT the runtime imports** — engines are
> locked to the exact GPU + TensorRT build and are **not portable**.

Run everything from the repo root with it on `PYTHONPATH`:
```bash
cd /path/to/TheAgency && export PYTHONPATH=$PWD
```

---

## 3. Where the weights go, and what gets produced

### Input weights (you provide these)
Put the four checkpoints in **one folder** (nothing is hardcoded; `worldv2`
preferred, `world` also works):
```
<WEIGHTS_DIR>/
  yolov8s-worldv2.pt   yolov8m-worldv2.pt   yolov8l-worldv2.pt   yolov8x-worldv2.pt
```
`YOLOWorld("yolov8s-worldv2.pt")` auto-downloads on first use if you have internet.
Point the tools at them with `--weights <path>` (or `WEIGHTS_DIR=<folder>` for
`build_all.sh`).

### Outputs (created for you)
```
sparx_agency/tasks/mapping/yolo_world_trt/engines/
  onnx/
    yolo_world_s.backbone.onnx        # image -> feature maps (static)
    yolo_world_s.head.onnx            # feats + txt_feats -> detections (dynamic N)
    yolo_world_s.io.json              # feat names/shapes, txt axis, N profile (do not edit)
    ...
  <target_tag>/                       # e.g. orin_sm87  or  nvidiageforcertx_sm120
    yolo_world_s.backbone.fp16.dla0.engine   (+ .json manifest: DLA layer count, IO, ...)
    yolo_world_s.head.fp16.gpu.engine        (+ .json manifest: dynamic N bounds, ...)
    timing_<target_tag>.cache
    ...
```

---

## 4. Quick start — build and compare (one command)

Yes — this one command does the whole chain end to end: for each variant it
**exports both ONNX graphs → builds both TRT engines → benchmarks the run speeds**,
then prints a ranked FPS summary and writes `/tmp/yolo_world_trt_compare.csv`.

```bash
cd /path/to/TheAgency && export PYTHONPATH=$PWD
export WEIGHTS_DIR=/path/to/yolo_world_weights
export IMAGES=/path/to/bench_frames        # optional → also runs the benchmark
./sparx_agency/tasks/mapping/yolo_world_trt/build_all.sh          # all four: s m l x
```

**Pick which models to build** — pass them as arguments (start small if you like):
```bash
./…/build_all.sh s          # just the small model
./…/build_all.sh s m        # small + medium
./…/build_all.sh            # no args → all four
./…/build_all.sh --help     # usage
```

Two things to know about "does everything":
- The **benchmark step only runs if `IMAGES` is set** (a folder of RGB frames).
  Without it you still get the ONNX + engines, just no speed comparison.
- It runs in **whatever environment you launch it in**: the export step needs
  `ultralytics`+`torch` and the build/benchmark steps need `tensorrt`+`pycuda`. On
  the laptop you get ONNX + a GPU engine (no DLA); the DLA engines and the real
  15 W numbers come from running it on the Orin (§2b).

No prompts are needed at build time (open-set). Overrides: `VARIANTS` (same as the
positional args), `IMGSZ=640x640`, `N_MAX=256`, `NUM_PROMPTS=4`, `DLA=on|off|auto`,
`PYTHON`.

---

## 5. The steps, run manually

**Step 1 — Export the split to ONNX** (export box; NO prompts baked):
```bash
python -m sparx_agency.tasks.mapping.yolo_world_trt.export_onnx \
    --weights $WEIGHTS_DIR/yolov8s-worldv2.pt --variant s --imgsz 288x512
```
This writes the backbone + head ONNX and runs a **parity gate** — it compares
`head(backbone(image))` against ultralytics' own full-model forward and aborts if
they differ. That gate is your safety net: the one fragile part is cutting the
graph correctly on *your* ultralytics version, and it is checked numerically every
export.

**Step 2 — Build both engines** (on the Orin; backbone→DLA, head→GPU chosen for you):
```bash
python -m sparx_agency.tasks.mapping.yolo_world_trt.build_engine \
    --onnx-dir sparx_agency/tasks/mapping/yolo_world_trt/engines/onnx --variant s --role both
# --role backbone|head to build one; --no-dla to force a GPU backbone; --dla to force DLA
```

**Step 3 — Benchmark & compare** (on the Orin):
```bash
python -m sparx_agency.tasks.mapping.yolo_world_trt.benchmark \
    --images /path/to/frames --num-prompts 4 \
    --pair s:.../yolo_world_s.backbone.fp16.dla0.engine,.../yolo_world_s.head.fp16.gpu.engine \
    --pair m:.../yolo_world_m.backbone.fp16.dla0.engine,.../yolo_world_m.head.fp16.gpu.engine \
    --pair l:.../..backbone..engine,.../..head..engine \
    --pair x:.../..backbone..engine,.../..head..engine
```
Breaks latency into **preprocess / backbone+head / decode+NMS / total**, prints
FPS and the **DLA-eligible layer count** per variant, and ranks them. It uploads
random embeddings for `--num-prompts` classes to drive the head's dynamic-N cost
(no torch needed on the target).

**Use it in code / the ROS node (open-set at run time):**
```python
from sparx_agency.tasks.mapping.yolo_world_trt.runtime import YoloTRTDetector
det = YoloTRTDetector(
    ".../yolo_world_s.backbone.fp16.dla0.engine",
    ".../yolo_world_s.head.fp16.gpu.engine",
    text_weights="yolov8s-worldv2.pt")     # the .pt drives the CLIP text branch
det.set_prompts(["refrigerator", "chair"]) # any prompts, any time, no rebuild
boxes = det.detect(rgb_hwc_uint8)          # -> List[core Detection2D]; torch-free
```
`set_prompts` runs the text encoder once (torch) and caches the embeddings;
`detect` is pure TensorRT + numpy. For a **torch-free runtime**, precompute the
embeddings offline and call `det.set_text_features(embeddings, labels)` instead.
To swap this into `yolo_detector_node`, construct a `YoloTRTDetector(...)` in place
of `YoloWorldDetector(...)` — same ABC.

---

## 5b. PyTorch vs TensorRT — is it actually faster?

`compare_torch_vs_trt.py` runs the **PyTorch** YOLO-World and the **TensorRT
split** over the same frames, with the same labels, at the same input size, and
prints a side-by-side latency / FPS table and the speed-up. You give it exactly:
the `.pt` weights, the two TRT engines, the image folder, and a label string.

```bash
python -m sparx_agency.tasks.mapping.yolo_world_trt.compare_torch_vs_trt \
    --torch-weights /path/to/yolov8s-world.pt \
    --backbone .../orin_sm87/yolo_world_s.backbone.fp16.dla0.engine \
    --head     .../orin_sm87/yolo_world_s.head.fp16.gpu.engine \
    --images   /path/to/frames \
    --labels   "weapon, chair, refrigerator"
```

Output (both timed over the full per-frame path: preprocess → inference → NMS):
```
model      |   mean ms      std      min      max |      FPS | dets/img
--------------------------------------------------------------------------
pytorch    |    ...          ...     ...      ... |    ...   |   ...
tensorrt   |    ...          ...     ...      ... |    ...   |   ...
--------------------------------------------------------------------------
TensorRT speed-up: N.NNx  (... -> ... ms/frame, ... -> ... FPS)
```
`dets/img` (mean detections per frame) lets you confirm the two agree, not just
that TRT is faster. Both run at the engine's built size by default (`--imgsz` to
override); needs `ultralytics`+`torch` **and** `tensorrt`+`pycuda` present.

## 6. Input size (the biggest lever after DLA)

The backbone input is **static**. Two sane choices, both multiples of 32:
- **`288×512` (default)** — matches the XTEND `504×294` resize with almost no
  letterbox padding; ~2.8× fewer pixels than 640². Fastest sensible option for
  15 W. **Recommended for the drone.**
- **`640×640`** — YOLO-World's native size; best open-vocab accuracy on small /
  ambiguous objects, but heavier.

The future `720×420` full frame only changes the letterbox source; pick a
stride-32 size for it (e.g. `416×704`) and re-export. Benchmark both — that's what
this task is for.

---

## 7. INT8 (opt-in, future)

DLA is far more efficient in INT8, but it needs a representative calibration set
and must pass an on-target accuracy check before flight. The precision knob is
plumbed; `build_engine.py` stops with a clear message because the calibrator isn't
wired yet. Plan: entropy calibrator fed real XTEND frames (mirroring NavDP's
`engine/calibrator.py`), FP16 stays the default, INT8 gated on a measured mAP delta.

---

## 8. Files

| File | Purpose |
|---|---|
| `hardware.py` | Probe GPU / Jetson / power → `HardwareProfile` (DLA-aware). |
| `build_policy.py` | Per-role (`backbone`/`head`) explicit `BuildPolicy`. |
| `wrappers.py` | Cut the ultralytics model at the first text-aware layer → backbone / head modules. |
| `text_embed.py` | `TextEmbedder`: prompts → text embeddings (the re-prompt-only text branch). |
| `export_onnx.py` | Export backbone + head ONNX + `io.json`, with the parity gate. |
| `build_engine.py` | ONNX → backbone(DLA,static) + head(GPU,dynamic-N) engines + manifests. |
| `preprocess.py` | Letterbox a frame into the static backbone input (pure numpy). |
| `postprocess.py` | Decode raw head + class-wise NMS + un-letterbox (pure numpy, dynamic nc). |
| `runtime.py` | `TwoStageYoloTRT` (shared feature buffers) + `YoloTRTDetector(DetectionModel)`. |
| `benchmark.py` | Compare `s/m/l/x` TRT engines: latency / FPS / DLA-vs-GPU. |
| `compare_torch_vs_trt.py` | PyTorch vs TensorRT speed on a folder + labels. |
| `build_all.sh` | One-shot export + build + benchmark for all variants. |
| `configs/build_policy.json` | Knobs (variants, imgsz, precision, DLA pools, head N profile). |
| `tests/` | Numpy-only tests (geometry, NMS, decode, policy, graph-cut) — run anywhere. |

Run the tests (no torch/TRT needed):
```bash
.venv/bin/python -m pytest sparx_agency/tasks/mapping/yolo_world_trt/tests/ -q
```

---

## 9. The one risk to know about

Cutting the ultralytics graph cleanly (`wrappers.py`) is the only version-fragile
part. It is written generically (find the first text-aware layer; reuse the model's
own `.f`/`.save` routing — no hard-coded indices) and is **guarded by the numerical
parity gate** in `export_onnx.py`, which fails loudly if the cut is wrong for your
ultralytics version. If it ever fires, inspect `wrappers.find_cut` and the head
routing against that version's `WorldModel.predict`.
```
