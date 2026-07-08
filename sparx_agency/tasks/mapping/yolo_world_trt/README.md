# YOLO-World TensorRT optimization (DLA-first, for Jetson AGX Orin @ 15 W)

Turns the ultralytics **YOLO-World** checkpoints (`s` / `m` / `l` / `x`) into
TensorRT engines tuned for the exact board they run on. The headline target is the
**Jetson AGX Orin 64 GB pinned to 15 W**, where the two **DLA** cores are used to
run the YOLO CNN off the GPU — the GPU is the throughput *and* power bottleneck at
that cap, so offloading the convolutional backbone/neck to the fixed-function DLA
is the single biggest win available.

This task owns only the **build + runtime + benchmark**. The detection algorithm
stays ROS-free in `core.mapping.detection` (`YoloWorldDetector`, the torch path).
The engine here is exposed through the same
`DetectionModel` ABC (`runtime.YoloTRTDetector`), so the existing
`yolo_detector_node` can consume it unchanged.

---

## 0. The one thing to understand first: prompts are *baked in*

YOLO-World is open-vocabulary because it fuses a **CLIP text encoder** into the
image neck. That text transformer is expensive and is *not* a good TensorRT/DLA
citizen. The standard, fast deployment path — the one used here — calls
`model.set_classes(prompts)` **before export**, which **freezes the text
embeddings into the head as constants**. The exported graph is then a pure-vision
CNN with a fixed number of classes `nc = len(prompts)`. That is exactly what makes
it fast and DLA-able.

**The trade-off:** a baked engine detects **only the prompts you baked**. To
detect a new class you must re-export + re-build. So bake the **full mission
vocabulary** you might ever target in one shot, e.g.:

```
--prompts refrigerator chair door person "fire extinguisher" backpack
```

At runtime you can *restrict* to a subset cheaply (`set_prompts(["refrigerator"])`),
but you cannot add a class that was not baked. `runtime.YoloTRTDetector` enforces
this and tells you to rebuild if you ask for an un-baked class.

---

## 1. Why these hardware choices (nothing is left to TensorRT defaults)

`hardware.py` probes the board (`detect()`), and `build_policy.py` turns that plus
`configs/build_policy.json` into an explicit `BuildPolicy` that `build_engine.py`
applies literally:

| Decision | Value | Why |
|---|---|---|
| **Precision** | FP16 (default) | DLA runs **FP16 or INT8 only, never FP32**. FP16 is the floor; INT8 is opt-in (see §7). |
| **Device** | `DLA` core 0 + **GPU fallback** | A prompt-baked YOLO is a CNN → DLA. A few tail ops (head decode Reshape/Transpose, some Slice/Concat, the max-sigmoid class step) are not DLA-supported, so `GPU_FALLBACK` is **mandatory** — without it the build fails; with it, the conv-heavy 90 % stays on DLA. |
| **DLA memory pools** | SRAM 1 MiB, local DRAM 1 GiB, global DRAM 512 MiB | Explicitly sized; the Orin DLA managed-SRAM is ~1 MiB/core. |
| **Workspace** | ≤ ¼ of shared RAM at 15 W | LPDDR is shared with the CPU; stay well under the cap. |
| **Optimization level** | 5 (max) | This is an **offline build-search** knob, *not* runtime power — nvpmodel clamps power regardless. Max search only costs a longer one-time build. |
| **Input size** | `288×512` (H×W) default | Stride-32, matches the 504×294 landscape frame with minimal letterbox padding, ~2.8× fewer pixels than 640². Use `640×640` for max open-vocab accuracy. See §6. |

DLA engines can be built **only on a Jetson**. On the x86 dev laptop the builder
detects no DLA and transparently produces a **GPU FP16** engine instead, so the
whole pipeline is still exercisable off-target (just not the DLA path).

---

## 2. Install

### 2a. Export box (laptop or Orin) — needs torch + ultralytics + onnx
Export is CPU-only and portable; you can do it on the laptop.

```bash
# In the project venv (or any py>=3.10 env):
pip install "ultralytics>=8.2" onnx onnxslim
# torch comes as an ultralytics dependency; a CPU wheel is fine for export.
```

### 2b. Target Orin — needs TensorRT + pycuda (for build + runtime + benchmark)
On JetPack these ship with the system Python. Use a venv **with system site
packages** so it sees the JetPack `tensorrt`:

```bash
python3 -m venv --system-site-packages ~/venvs/trt
source ~/venvs/trt/bin/activate
python -c "import tensorrt, pycuda; print(tensorrt.__version__)"   # sanity
pip install pycuda numpy opencv-python   # if not already present
```

> Build **on the target with the same TensorRT the runtime imports** — engines are
> locked to the exact GPU + TensorRT build and are **not portable**.

Always run modules from the **repo root** with it on `PYTHONPATH`:
```bash
cd /path/to/TheAgency
export PYTHONPATH=$PWD
```

---

## 3. Where to put the weights, and where outputs land

### Input weights (the original `.pt` you provide)
Download the four ultralytics YOLO-World checkpoints and put them in **one folder**
of your choice (nothing is hardcoded). `worldv2` is preferred; `world` also works:

```
<WEIGHTS_DIR>/
  yolov8s-worldv2.pt
  yolov8m-worldv2.pt
  yolov8l-worldv2.pt
  yolov8x-worldv2.pt
```
Get them from the ultralytics releases (e.g. `YOLOWorld("yolov8s-worldv2.pt")`
auto-downloads on first use if you have internet). Point the tooling at them with
`--weights <path>` or `WEIGHTS_DIR=<folder>` for `build_all.sh`.

### Outputs (created for you)
```
sparx_agency/tasks/mapping/yolo_world_trt/engines/
  onnx/                          # exported ONNX + baked-class sidecars
    yolo_world_s.onnx
    yolo_world_s.classes.json    # the baked prompt order + shapes (do not edit)
    ...
  <target_tag>/                  # e.g. orin_sm87  or  nvidiageforcertx_sm120
    yolo_world_s.fp16.dla0.engine
    yolo_world_s.fp16.dla0.engine.json   # manifest: TRT ver, SM, prompts, DLA layer count
    timing_<target_tag>.cache
    ...
```
The `<target_tag>` subfolder keeps engines from different boards from clobbering
each other. The `.json` manifest lets the runtime version-lock and records how many
layers were DLA-eligible.

---

## 4. Quick start — build all four and compare (one command)

```bash
cd /path/to/TheAgency && export PYTHONPATH=$PWD
export PROMPTS="refrigerator chair door person"     # your baked mission vocabulary
export WEIGHTS_DIR=/path/to/yolo_world_weights
export IMAGES=/path/to/bench_frames                 # optional → also runs the benchmark
./sparx_agency/tasks/mapping/yolo_world_trt/build_all.sh
```

This exports → builds → benchmarks `s`, `m`, `l`, `x` with the same prompts and
input size, then prints a ranked FPS summary and writes
`/tmp/yolo_world_trt_compare.csv`.

Useful env overrides: `VARIANTS="s m"`, `IMGSZ=640x640`, `PRECISION=fp16`,
`DLA=on|off|auto` (default `auto`), `PYTHON=/path/to/python`.

---

## 5. The steps, run manually (what `build_all.sh` does)

**Step 1 — Export to ONNX** (export box; bakes the prompts):
```bash
python -m sparx_agency.tasks.mapping.yolo_world_trt.export_onnx \
    --weights $WEIGHTS_DIR/yolov8s-worldv2.pt --variant s \
    --prompts refrigerator chair door person \
    --imgsz 288x512 \
    --out-dir sparx_agency/tasks/mapping/yolo_world_trt/engines/onnx
```

**Step 2 — Build the engine** (on the Orin; DLA + FP16 chosen automatically):
```bash
python -m sparx_agency.tasks.mapping.yolo_world_trt.build_engine \
    --onnx sparx_agency/tasks/mapping/yolo_world_trt/engines/onnx/yolo_world_s.onnx \
    --variant s
# add --no-dla to force a GPU-only engine, or --dla to force DLA (errors off-Jetson)
```

**Step 3 — Benchmark & compare** (on the Orin):
```bash
python -m sparx_agency.tasks.mapping.yolo_world_trt.benchmark \
    --images /path/to/frames \
    --engine s:.../orin_sm87/yolo_world_s.fp16.dla0.engine \
    --engine m:.../orin_sm87/yolo_world_m.fp16.dla0.engine \
    --engine l:.../orin_sm87/yolo_world_l.fp16.dla0.engine \
    --engine x:.../orin_sm87/yolo_world_x.fp16.dla0.engine
```
The report breaks latency into **preprocess / inference / decode+NMS / total**,
prints **FPS** and the **DLA-eligible layer count** per variant, and ranks them.

**Use it in code / the ROS node:**
```python
from sparx_agency.tasks.mapping.yolo_world_trt.runtime import YoloTRTDetector
det = YoloTRTDetector(".../orin_sm87/yolo_world_s.fp16.dla0.engine")
det.set_prompts(["refrigerator"])          # subset of the baked classes
boxes = det.detect(rgb_hwc_uint8)          # -> List[core Detection2D]
```
To swap the torch detector in `yolo_detector_node` for this, construct a
`YoloTRTDetector(engine_path)` instead of `YoloWorldDetector(...)` — same ABC.

---

## 6. Choosing the input size (the biggest performance lever after DLA)

The engine input is **static** (a fixed `H×W`). Two sane choices:

- **`288×512` (default)** — matches the XTEND `504×294` resize (landscape,
  aspect 1.78 vs 1.71) with almost no letterbox padding. ~2.8× fewer pixels than
  640², i.e. the fastest sensible option for the 15 W budget. **Recommended for the
  drone.**
- **`640×640`** — YOLO-World's native training size; best open-vocabulary accuracy
  (small/ambiguous objects), but heavier. Use if you see missed detections at
  288×512.

Both must be **multiples of 32**. The future `720×420` full frame just changes the
letterbox source; pick a stride-32 engine size for it (e.g. `416×704`) and re-export.
Benchmark both and let the numbers decide — that is exactly what this task is for.

---

## 7. INT8 (opt-in, future)

DLA is dramatically more efficient in INT8, but it needs a **representative
calibration set** and must pass an on-target accuracy check before it can be
trusted for flight. The precision knob and manifest already carry `int8`;
`build_engine.py` currently stops with a clear message because the calibrator is
not wired yet. Plan: add an entropy calibrator fed a folder of real XTEND frames
(mirroring the NavDP `engine/calibrator.py`), keep FP16 as the safe default, and
gate INT8 selection on a measured mAP delta.

---

## 8. Files

| File | Purpose |
|---|---|
| `hardware.py` | Probe GPU / Jetson / power → `HardwareProfile` (DLA-aware). |
| `build_policy.py` | Derive the explicit `BuildPolicy` from hardware + `configs/`. |
| `export_onnx.py` | ultralytics `YOLOWorld.pt` → static ONNX (bakes prompts). |
| `build_engine.py` | ONNX → TensorRT engine with explicit DLA/FP16 config + manifest. |
| `preprocess.py` | Letterbox a frame into the fixed engine input (pure numpy). |
| `postprocess.py` | Decode raw head + class-wise NMS + un-letterbox (pure numpy). |
| `runtime.py` | TRT engine runner + `YoloTRTDetector(DetectionModel)`. |
| `benchmark.py` | Compare `s/m/l/x`: latency / FPS / DLA-vs-GPU breakdown. |
| `build_all.sh` | One-shot export + build + benchmark for all variants. |
| `configs/build_policy.json` | Version-controlled knobs (variants, imgsz, precision, DLA pools). |
| `tests/` | Numpy-only tests (geometry, NMS, decode, policy) — run anywhere. |

Run the tests (no torch/TRT needed):
```bash
.venv/bin/python -m pytest sparx_agency/tasks/mapping/yolo_world_trt/tests/ -q
```
