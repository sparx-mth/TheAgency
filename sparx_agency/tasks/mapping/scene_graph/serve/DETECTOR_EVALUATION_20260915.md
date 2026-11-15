# Detector upgrade: measured development results — 2026-09-15

## Current selection

**YOLO-World X-v2 (`yolov8x-worldv2.pt`) is the default by explicit user request.** LLMDet-L remains selectable; S/L remain explicit checkpoint overrides. Confidence thresholds, exploration, camera control, target evidence and navigation algorithms are unchanged. No ApexNav, SG-Nav or OSG Navigator logic has been implemented.

This operational choice is **not a held-out benchmark or navigation-improvement claim**. The initial S-versus-LLMDet study recommended keeping S; the later X comparison informed the user's separate default choice. See the [sourced comparison and runbook](README.md).

## Configuration and provenance

The work was developed on `feat/objnav-open-vocabulary-detector-nadav` from shared infrastructure **60d91714**, without changing other worktrees.

- S baseline: `yolov8s-worldv2.pt`; X follow-up/default: `yolov8x-worldv2.pt`. Both use image size 640 and NMS IoU 0.5.
- LLMDet: `iSEE-Laboratory/llmdet_large`, revision `bec37f296f05b22f6c6b39bc05a6c611239f4e31`; Swin-L/BERT, resize 800/1333, 900 queries, caption chunk cap 80. This vocabulary uses one 129-token caption. Portable PyTorch deformable attention, custom kernels off.
- Matched settings: same 54-class ordered vocabulary and RGB frames; JPEG90; batch one; 0.05 emission floor; max 100 boxes; four CPU intra-op threads.
- Linux host: RTX 5070 Laptop GPU (8 GB), NVIDIA driver 595.84. Simulator rendering and GPU inference ran separately; navigation checks used CPU detection. No paid service or dataset upload was used.
- Model environment: Python 3.10.20; Torch 2.11.0+cu128; torchvision 0.26.0+cu128; Transformers 5.17.0; tokenizers 0.23.2; safetensors 0.8.0; huggingface-hub 1.8.0; Ultralytics 8.3.253; numpy 1.26.0; Pillow 11.2.1; OpenCV 4.9.0.

| Checkpoint | SHA-256 |
|---|---|
| YOLO-World S v2 | `9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792` |
| YOLO-World X v2 | `41e771bfbbb8894dd857f3fef7cac3b3578dffd49fd3547101efa6a606a02a0e` |
| LLMDet safetensors | `6799bea5a4fe67ddda9ba55cf46cff06ae82653f3415e2d7d08428e02f3a0766` |
| Auxiliary CLIP ViT-B/32 | `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` |

Per-run metadata records versions, preprocessing, configuration and captions. The HF bundle has a separate aggregate identity; the final service also hashes actual YOLO prompt features. No Torch/Transformers was installed into `.venv`. The model overlay inherits one unrelated `pip check` issue: `generate-parameter-library-py 0.7.3` lacks `typeguard`. Its ROS provisioning was not changed; detector imports and inference passed.

## Data and calibration

**48 training RGB-D frames**, collected with AI2-THOR 5.0.0 and reference build `bad5bc2b250615cb766ffb45d455c211329af17e`. No validation/test episode was used.

- Calibration: 24 frames, **29 usable target instances**, from `FloorPlan_Train1_1` and `FloorPlan_Train2_1`.
- Measurement: 24 frames, **45 usable target instances**, including **20 small boxes**, from `FloorPlan_Train3_1` and `FloorPlan_Train4_1`.
- Fixed capture: first training start, bearings 0/60/120/180/240/300, horizons 0 and 30 degrees. Level views are data collection, not a camera-policy change.
- GT is visible instance-mask bounds at all ranges. Two masks with fewer than four pixels are ignored. Small means original-frame box area <32×32. Occlusion can make these differ from amodal detector boxes.
- Counts are correlated per-frame instances, not 74 independent objects. Matching is score-ordered one-to-one IoU≥0.5, with recorded aliases and negative frames retained. These are point precision/recall metrics, **not AP**. No garbage-can GT occurred, so its recall is unmeasured.
- A predeclared 0.10–0.80 grid, step 0.05, maximizes calibration **F0.5 after unchanged shared suppression**. Selected thresholds: **S 0.50, X 0.50, LLMDet 0.60**. Calibration TP/FP: S 10/0, X 20/2, LLMDet 15/0. No measurement-set threshold tuning was performed.

These are inspected development inputs, **not held-out benchmark results**. The earlier five Gibson episodes remain development examples; S/L success of 2/5 is not detection accuracy or proof of an L gain.

## Localization on the 24 measurement frames

Rows include existing shared suppression. CPU FP32 and GPU FP16 LLMDet produce the same aggregate counts at these points, not necessarily identical boxes/scores or universally lossless FP16.

| Detector / threshold | TP | FP | Misses | Precision | Recall | Small recall |
|---|---:|---:|---:|---:|---:|---:|
| S, original 0.35 | 12 | 3 | 33 | 80.0% | 26.7% | 0/20 |
| S, calibrated 0.50 | 11 | 0 | 34 | 100% (11/11) | 24.4% | 0/20 |
| X, 0.35 | 17 | 2 | 28 | 89.5% | 37.8% | 2/20 (10%) |
| X, calibrated 0.50 | 12 | 1 | 33 | 92.3% | 26.7% | 0/20 |
| LLMDet, diagnostic 0.35 | 28 | 13 | 17 | 68.3% | 62.2% | 8/20 (40%) |
| LLMDet, calibrated 0.60 | 9 | 0 | 36 | 100% (9/9) | 20.0% | 0/20 |

The 0.35 LLMDet row is diagnostic, not an inherited deployment threshold. “100%” over 9 or 11 predictions is not a safety guarantee.

At 0.35, S has 7 raw FP, including 4 duplicates; shared suppression leaves 3 FP without losing its 12 TP. LLMDet has 13 raw FP and zero counted duplicates, unchanged by suppression. Its target confusions include **mug → bowl twice** and **vase → apple once**. Fewer duplicates did not imply higher precision.

At calibrated thresholds, LLMDet improves basketball recall (4/4 vs S 1/4), but loses television (0/6 vs 4/6), mug (1/3 vs 2/3), and house-plant recall (3/6 vs 4/6). It misses all small targets at that conservative point.

## Follow-up: YOLO-World X-v2

X was tested after the initial comparison, with capture hashes, frame order, vocabulary and settings checked against S. The [official checkpoint](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-worldv2.pt) required no new detector or navigation implementation.

At 0.35, X improves all three measured accuracy metrics over S on this subset at approximately **3.5× CPU HTTP latency**. Raw and suppressed X counts agree at both thresholds, with zero counted duplicates. No cross-target confusion was counted on measurement frames at those points; calibration included a vase/alarm-clock confusion. This does not establish general absence of confusion.

**No X-specific navigation/false-STOP check was run.** The later user-selected X default does not establish general benchmark improvement. A real CPU inference through the final default CLI, without a model/backend override, verified the exact X-v2 checkpoint above.

## Latency and memory

48 matched requests per successful configuration after warm-up. HTTP timing includes health/provenance checks, JPEG, transfer, inference and parsing.

| Configuration | HTTP median / p95 | Server median | Peak host RSS | CUDA peak allocated / reserved |
|---|---:|---:|---:|---:|
| S, CPU FP32 | 148 / 165 ms | 60.6 ms | 2,102 MiB | none |
| X, CPU FP32 | 518 / 549 ms | 429 ms | 2,359 MiB | none |
| LLMDet, CPU FP32 | 17,664 / 17,898 ms | 17,577 ms | 8,179 MiB | none |
| S, GPU FP32 | 96 / 100 ms | 6.0 ms | 2,196 MiB | 734 / 800 MiB |
| LLMDet, GPU FP16 | 772 / 880 ms | 682 ms | 2,795 MiB | 3,569 / 4,864 MiB |

GPU cold-start plus warm-load: S 3.73 s; LLMDet FP16 4.85 s. RSS includes loading; CUDA values are allocator peaks, not whole-board use. The roughly 86–90 ms HTTP overhead is material for YOLO and is not omitted.

**LLMDet FP32 GPU failed explicitly** under the safety cap: another 1.02 GiB was requested with 4.18 GiB allocated, about 332 MiB reserved-unused, and 5.27 GiB allowed by the 70% cap. This establishes failure at that budget, not an exact universal minimum VRAM. FP16 was a separate explicit experiment, never fallback; no simulator GPU reserve was taken.

## Downstream checks and limitations

S at 0.35 and verified LLMDet at 0.60 each completed **12 actual actions** from the same `FloorPlan_Train3_1` start, target AlarmClock. HeadlessObjNavAgent, depth projection, observed map, scene graph, RPT*, weighted A* and execution ran unchanged with CPU detection.

- Both: **0 STOPs, 0 false STOPs, no target-confirmation evidence**. Movement, turns and four A* calls occurred; this does not show false STOPs are solved.
- No LOOK actions: the downward-camera issue remains.
- Room-language responses were an explicit deterministic fixture, not a live LLM: two calls for S and three for LLMDet. No cold LLM took the rendering GPU.
- Final object landmarks: S 10, LLMDet 1. Target-only calibration also affects room context.
- Door observations/depth-accepted/confirmed: S **10/5/0**, LLMDet **133/13/3**. Doors were not annotated; these are behavior diagnostics, not accuracy gains.
- Synthetic tests preserve alias suppression, nearest association, distinct-frame support and viewpoint separation. A persistent false-label fixture still triggers STOP after sufficient separated observations.
- No paired annotated Gibson, MP3D, HM3Dv1 or HM3Dv2 accuracy study was completed. The integration supports those adapters, but these results do not establish scanned-scene or navigation improvement.

## Invalidated loader experiment

Stock Transformers reported no missing/mismatched keys but substituted **35 decoder tensors**, tying six independent heads to head zero. Comparing the loaded state with safetensors exposed the defect. `llmdet_loading.py` corrects only alias bookkeeping and verifies **42 decoder tensors** before serving. Architecture, forward computation and checkpoint are unchanged.

`llmdet_cpu.json` and `llmdet_navigation_smoke.json` remain **invalid loader diagnostics**, excluded from conclusions. Use the verified artifacts below; a clean loading-info report alone was insufficient.

## Artifacts and validation

Under `$HOME/objnav_benchmark/detector_upgrade_20260915/`:

- `train_capture/manifest.json` and immutable hashed RGB-D `.npz` files.
- `yolo_world_cpu_4threads.json`, `yolo_world_cuda_fp32.json`.
- `yolo_world_x_cpu_fp32.json`: X raw predictions, identity and calibration grid.
- `llmdet_verified_cpu_fp32.json`, `llmdet_verified_cuda_fp16.json`.
- `llmdet_verified_cuda_fp32.json`: explicit capacity failure.
- `yolo_world_navigation_smoke.json`, `llmdet_verified_navigation_smoke.json`.
- `verified_summary.json` and `artifact_status.json`: identities, hashes and validity classification; the later X run is separate.

Checkpoints/source records, licensed data, PDFs and upstream clones remain outside git. Existing services on 18092/18093/18094 and other worktrees were not reconfigured. The unrelated `.run/` configuration was preserved.

The initial study passed 1,702 tests. After the X-default change, **1,709 tests passed**, along with the HTTP self-test, shell syntax check and real default-CLI inference. One existing Matplotlib/Axes3D environment warning remains.

A larger frozen development study with hard negatives, small objects, scanned scenes, separately calibrated target/context/door evidence, and positive and false-STOP cases is still needed. No additional paper algorithms are enabled.

