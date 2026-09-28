# Selectable open-vocabulary detector service

**YOLO-World X-v2 (`yolov8x-worldv2.pt`) is the default**, selected by the user
after the development comparison. `--backend llmdet` selects
the official pretrained **LLMDet Swin-L**, not a newly trained detector. Both
implement the same RGB → `Detection2D` interface and the existing JPEG HTTP
contract. No navigation, frontier, camera-control or STOP algorithm is replaced.
YOLO S/L remain explicit checkpoint overrides. The default change does not
change confidence thresholds or add fusion/reasoning from the reviewed papers.

## Optional grounded visual verification

The new [Grounding DINO + BLIP-2 runbook](GROUNDED_VLM.md) adds `grounded_vlm`
(DINO + BLIP-2, no YOLO) and `hybrid` (all three), with one `backend` field in
[`configs/gibson_perception.json`](configs/gibson_perception.json). Set it to
`yolo_world` for YOLO only. The no-config default above is unchanged.
Grounding DINO Base and BLIP-2 FLAN-T5 XL use pinned, verified local snapshots.
Verification is experimental, defaults to all classes and never replaces depth,
floor-transition or multi-view safety checks. Its evidence is recorded per frame.

## Selection review — 2026-09-15

Here *open vocabulary* means localizing category names supplied as text. Unknown
object discovery, closed-set detection and single referring-expression grounding
are not interchangeable evidence. There is no defensible universal SOTA winner.

| Candidate / primary source | Relevant evidence (authors' results, not ours) | Availability, cost and suitability |
|---|---|---|
| [SAM 3](https://arxiv.org/abs/2511.16719), [official code](https://github.com/facebookresearch/sam3) | Revised paper Table 1: LVIS **53.6 box AP**, COCO **56.4**, very strong concept discrimination/negative examples. SA-Co concept F1 is NOT LVIS AP. | Released 860M-parameter checkpoint, but [manual access gate](https://huggingface.co/facebook/sam3) returned HTTP 401 here. Approximately 3.44 GB FP32 parameters alone, not total VRAM. Authors report **30 ms/image on H200**, not on this laptop or for 54 independent category prompts. Custom [SAM license](https://github.com/facebookresearch/sam3/blob/main/LICENSE), including redistribution and restricted-end-use terms; not Apache. Needs approved checkpoint access before it can be evaluated here. |
| [SAM 3.1](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md), March 2026 | Object Multiplex improves **video multi-object tracking**; ~7× at 128 objects on H100. | Not evidence of a 7× still-image category detector or improved ObjectNav accuracy. Same gated checkpoint family. Official runtime requires Python ≥3.12, PyTorch ≥2.7, CUDA ≥12.6. No verified 8 GB full-vocabulary memory figure. |
| [DINO-X Pro](https://github.com/IDEA-Research/DINO-X-API), [paper](https://arxiv.org/abs/2411.14347) | LVIS minival **59.8 AP / 63.3 rare**, full val **52.4 / 56.5 rare**; COCO **56.0**. Grounding DINO 1.5/1.6 Pro are older API alternatives. | Official repository supplies API examples, **not these Pro weights**. Apache example code does not license hosted weights. Requires DeepDataSpace access/token, quotas and remote image transfer; service latency and local VRAM not established. No paid calls or data uploads were made. |
| [WeDetect-L](https://github.com/WeChatCV/WeDetect), [paper](https://arxiv.org/abs/2512.12309), CVPR 2026 | Table 1: LVIS minival **55.0 / 51.1 rare**, full val **49.4 / 43.3 rare**; COCO **54.5**. 490M vision parameters, 1280² input, reported **6 FPS** (hardware not specified in that table). | Public [weights](https://huggingface.co/fushh7/WeDetect), GPLv3 code **and weights**, official standalone PyTorch deployment available June 2026. ~1.96 GB FP32 vision parameters alone, plus text tower/activations. **Released detector requires Chinese class names**. Stronger published LVIS result than LLMDet, but not demonstrated with our arbitrary English vocabulary. An audited translation/prompt-mapping evaluation remains necessary; English was not silently sent to this model. |
| **[LLMDet-L](https://github.com/iSEE-Laboratory/LLMDet), [paper](https://arxiv.org/abs/2501.18954)** | Official model zoo: LVIS minival **51.1 / 45.1 rare**, full val **42.0 / 31.6 rare**; chunk-80 protocol **52.4 / 44.3**, **43.2 / 32.8** respectively. | Public official [Transformers checkpoint](https://huggingface.co/iSEE-Laboratory/llmdet_large), Apache-2.0 code/weights. ~343M parameters / 1.37 GB FP32 parameter storage; actual peak is larger. No authoritative latency for this HF implementation/host: measured locally below. Native English, exact category scoring, offline CPU support. **Selected runnable candidate**, not claimed superior to SAM 3 or WeDetect on their protocols. |
| [OWLv2-L](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit) | Official README highlights **44.6 LVIS rare AP**. Self-training, LVIS-base fine-tuning and weight-ensemble checkpoints have different results and supervision. | Apache-2.0 code/weights; Transformers implementation available. Strong rare-category option. Do not call LVIS-base+ensemble evaluation fully unseen-category training. Latency plots precompute text and use particular checkpoints/resolutions; no measured host latency here. |
| [YOLOE](https://github.com/THU-MIG/yoloe), ICCV 2025 | Text-prompt YOLOE-v8-S: **27.9 fixed AP / 22.3 rare** on LVIS minival. Published +3.5 AP over Worldv2-S with Objects365+GoldG training. | Released weights; Ultralytics-derived AGPL/commercial licensing must be reviewed. **305.8 FPS** is T4 **TensorRT**, not PyTorch HTTP latency. Credible efficient upgrade, not the strongest accuracy candidate. |
| [YOLO-World](https://github.com/AILab-CVC/YOLO-World) / [Ultralytics implementation](https://docs.ultralytics.com/models/yolo-world/) | Existing `yolov8s-worldv2.pt` baseline; larger L did not establish a navigation gain in the five previously inspected Gibson development episodes. | Current Ultralytics implementation/weights have AGPL-3.0/commercial terms. Preserve explicit S/L checkpoint identity; do not equate all paper YOLO-World variants with the Ultralytics v2 release. This pipeline's actual CPU baseline is measured, not inferred from TensorRT FPS. |
| [NVIDIA LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B), [May 2026 paper](https://arxiv.org/abs/2605.27365) | Strong dense/general grounding, parallel box decoding. Reports box F1/IoU and boxes/sec; these are not directly comparable to AP or calibrated detection scores. | Public 3.83B-parameter BF16 checkpoint (~7.66 GB parameters alone), **academic/non-profit research only**. Official 4K batch-4 A100 probe: **11.71 GB** peak reserved with sparse attention, **35.12 GB** with dense masks. Not an 8 GB co-resident renderer deployment; no unapproved quantization or smaller-model substitution. |

Also checked [OV-DEIM](https://github.com/wleilei/OV-DEIM) (March 2026): L reports
35.9 fixed AP / 36.8 rare at 640 on its LVIS protocol, Apache code but **CC-BY-NC
weights**; it does not displace the accuracy candidates above. Original Grounding
DINO/MM-GDINO remain useful mature baselines; the selected LLMDet is their
caption-supervised successor, not the API-only Pro model.

### Comparability and unresolved risks

* LVIS **minival vs full val**, fixed AP vs standard AP, all-category vs novel
  category evaluation, mask vs box metrics, and prompt chunk sizes must be kept
  separate. The large minival/full-val gaps above are not transcription errors.
* LLMDet adds **GroundingCap-1M caption supervision** to MM-GDINO pretraining;
  WeDetect uses ~20M samples (15M self-collected); OWLv2 uses web self-training;
  SAM 3 uses a much larger concept/mask data engine. These are not controlled
  architectural comparisons. COCO training exposure also differs; “zero-shot
  transfer” does not guarantee every queried category was absent from pretraining.
* Small-object success on RoboTHOR is not established by aggregate AP or rare AP.
  Native resize resolutions/backbones differ. We measure visible small boxes
  separately and leave detector misses distinct from the known downward-camera
  policy issue. No evidence here warrants removing multi-view confirmation.
* LLMDet's shipped HF processor uses **800 / 1333**, whereas a published large
  model comparison uses 1000 / 1560. Our operating point is recorded explicitly;
  these measurements are **not** a reproduction of paper AP.

## Setup (separate model environment)

Do not install Torch/Transformers into the lightweight `.venv`. Provision a
dedicated Python ≥3.10 environment with matching Torch/torchvision wheels for the
device. On this machine an isolated overlay can reuse the existing Torch install
read-only, without modifying `navdp` or the other simulators:

```bash
"$HOME/miniconda3/envs/navdp/bin/python" -m venv --system-site-packages "$HOME/.venvs/objnav-detector"
DETECT_PY="$HOME/.venvs/objnav-detector/bin/python"
"$DETECT_PY" -m pip install -r sparx_agency/tasks/mapping/scene_graph/serve/requirements-llmdet.txt
"$DETECT_PY" -m sparx_agency.tasks.mapping.scene_graph.serve.provision_llmdet \
  --output "$HOME/models/objnav/llmdet-large"
```

The explicit provisioning command pins `iSEE-Laboratory/llmdet_large` revision
`bec37f296f05b22f6c6b39bc05a6c611239f4e31`, verifies the publisher's weight SHA-256,
and writes `source.json`. Existing directories are not overwritten. Inference is
**local-files-only**, safetensors-only, no remote code, no implicit downloads.
Transformers 5.17.0 / hub 1.8.0 / safetensors 0.8.0 / tokenizers 0.23.2 passed
the dependency vulnerability check on the review date. Other inherited packages
still require their own deployment security review.

**Do not bypass our checkpoint loader.** The stock Transformers 5.17.0
MM-GDINO loader (also inspected in 5.10.0) incorrectly ties distinct LLMDet
decoder stages: 35 saved tensors were silently substituted despite an empty
loading-error report. `llmdet_loading.py` corrects only the alias map, retains
the upstream architecture/forward, and checks all 42 saved decoder tensors
against the actual loaded values. The six heads remain distinct. This fix and
the verification count are included in service provenance. It is not training,
a detector redesign, or a modification to installed Transformers files.

YOLO-World also uses Ultralytics' OpenAI CLIP ViT-B/32 text encoder. Provision
its auxiliary weights in the model environment in advance (on this host they
are in `$HOME/.cache/clip/ViT-B-32.pt`); Ultralytics can otherwise download them
while setting classes. The strict safetensors/offline guarantee above applies
to **LLMDet**; the legacy YOLO auxiliary-loader behavior is unchanged.

## Run and rollback

Obtain `VOCAB` from the simulator adapter's `--print-vocabulary` (same order for
both models). Do not change another mission's `/set_classes`. Use separate ports:

With no `--model`, the YOLO service uses `yolov8x-worldv2.pt` in its working
directory. Provision that local file first; a missing X checkpoint fails instead
of downloading or selecting S/L. On this machine the verified download is in
`$HOME/models/objnav/`; from the repo root, a local gitignored symlink enables
the default invocation:

```bash
ln -s "$HOME/models/objnav/yolov8x-worldv2.pt" yolov8x-worldv2.pt
```

Alternatively pass `--model "$HOME/models/objnav/yolov8x-worldv2.pt"`.
The [official X-v2 release](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-worldv2.pt)
and its SHA-256 are recorded in the evaluation report. LLMDet still requires
both `--backend llmdet` and an explicit `--model` directory and `--conf` value.

```bash
# Low emission floor for calibration and existing door processing, NOT STOP confidence.
CUDA_VISIBLE_DEVICES="" "$DETECT_PY" -m sparx_agency.tasks.mapping.scene_graph.serve.detection_server \
  --device cpu --host 127.0.0.1 --port 18095 --conf 0.05 --torch-threads 4 --classes "$VOCAB"

CUDA_VISIBLE_DEVICES="" "$DETECT_PY" -m sparx_agency.tasks.mapping.scene_graph.serve.detection_server \
  --backend llmdet --model "$HOME/models/objnav/llmdet-large" --device cpu \
  --host 127.0.0.1 --port 18096 --conf 0.05 --torch-threads 4 --classes "$VOCAB"

# Explicit S rollback: use instead of the default service above.
CUDA_VISIBLE_DEVICES="" "$DETECT_PY" -m sparx_agency.tasks.mapping.scene_graph.serve.detection_server \
  --backend yolo_world --model yolov8s-worldv2.pt --device cpu \
  --host 127.0.0.1 --port 18095 --conf 0.05 --torch-threads 4 --classes "$VOCAB"
```

Point the adapter at the desired `--detector-url`. Set the independently
calibrated downstream **`RPTSettings.detection_confidence`** through its
`--policy-config` JSON. Do not remap LLMDet confidence onto YOLO's score scale.
Keep the service emission floor ≤ both object and door thresholds; existing
adapters check this. Door thresholds are **not calibrated by the target-only
study**. `HttpDetector(..., expected_backend="llmdet")` adds an optional strict
endpoint-type check. Normal clients also pin complete health/inference identity.

The RGB contract is unchanged: client RGB → OpenCV BGR/JPEG90 → server RGB →
PIL processor RGB/resize/ImageNet normalization. LLMDet boxes are normalized
CXCYWH → original-image XYXY, clipped and integer-truncated. Labels come from
exact category token spans, not arbitrary decoded fragments. More than 256 BERT
tokens causes deterministic caption chunking; a single overlong label fails.
Backend, checkpoint files, source revision, actual preprocessing/captions,
configuration, wire codec and package versions are returned atomically.

For GPU inference pass `--device cuda:0` (also the legacy CLI default; always
pass `--device cpu` in simulator launches). The service refuses a card with
>512 MiB resident use **before CUDA context creation**, and caps the allocator
at 70%. This conservative check is not a scheduler or a guarantee against a
concurrent launch. **Do not co-launch with simulator rendering**. OOM/device
errors fail rather than switching model, precision or device.

The verified model fits the tested GPU budget with **explicit FP16**; FP32
exceeded the 70% allocator budget. FP16 is not selected automatically. Use a
dedicated GPU host or stop the simulator before this isolated profile:

```bash
"$DETECT_PY" -m sparx_agency.tasks.mapping.scene_graph.serve.profile_detector \
  --manifest "$HOME/objnav_detector_dev/train_capture/manifest.json" \
  --output "$HOME/objnav_detector_dev/llmdet-cuda-fp16.json" \
  --backend llmdet --model "$HOME/models/objnav/llmdet-large" \
  --device cuda:0 --dtype float16 --conf 0.05 --torch-threads 4
```

`profile_detector` uses a fresh real HTTP server and the ObjectNav HTTP client,
records cold start and CUDA allocator peaks, and releases the model on exit.
Failures are recorded, not replaced with a successful weaker configuration.

## Reproduce the development comparison

To reproduce the historical **S** baseline, run the explicit S rollback command
above; to compare **X**, use the default service. Every result pins the actual
checkpoint, so the default change cannot silently relabel the previous S runs.

`capture_detector_dev` is an optional training-data collector, **not** a new
navigation adapter. Requires the already provisioned reference AI2-THOR build
and its own simulator environment. It downloads no licensed scene assets.

```bash
DISPLAY=:1 __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
  "$HOME/miniconda3/envs/ai2thor/bin/python" -m sparx_agency.tasks.mapping.scene_graph.serve.capture_detector_dev \
  --episodes "$HOME/datasets/objectnav/robothor" --output "$HOME/objnav_detector_dev/train_capture" \
  --detector-url http://127.0.0.1:18094

python -m sparx_agency.tasks.mapping.scene_graph.serve.compare_detectors \
  --manifest "$HOME/objnav_detector_dev/train_capture/manifest.json" \
  --url http://127.0.0.1:18095 --output "$HOME/objnav_detector_dev/yolo.json"
python -m sparx_agency.tasks.mapping.scene_graph.serve.compare_detectors \
  --manifest "$HOME/objnav_detector_dev/train_capture/manifest.json" \
  --url http://127.0.0.1:18096 --output "$HOME/objnav_detector_dev/llmdet.json"
```

The recorded dataset is small and explicitly developmental: first distinct
training starts, six fixed bearings, level/downward views; layouts 1/2 calibrate,
layouts 3/4 measure. Calibration maximizes **F0.5** on a declared 0.10–0.80 grid
after the **unchanged** shared suppression. Both current 0.35 and independently
selected thresholds are reported. Matching is score-ordered one-to-one IoU≥0.5;
visible-mask boxes <4 mask pixels are ignored; small means box area <32² in the
original 640×480 frame. This is **not AP**, an exhaustive benchmark, or held-out
ObjectNav performance. Occlusion can make visible-mask boxes differ from the
amodal boxes a detector predicts. Keep raw predictions, manifest and hashes.

See [measured results and limitations](DETECTOR_EVALUATION_20260915.md).

### Bounded real-action pipeline smoke

This infrastructure branch deliberately contains no RoboTHOR adapter. Use the
already provisioned adapter checkout via `PYTHONPATH`, leaving its files and
branch untouched. `ROOT` below is this shared detector checkout; `ADAPTER_ROOT`
is the existing RoboTHOR checkout. CPU services must already be running with the
adapter's vocabulary. Repeat for YOLO at its current 0.35 threshold.

```bash
ROOT="$PWD"
DISPLAY=:1 __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
PYTHONPATH="$ADAPTER_ROOT" "$HOME/miniconda3/envs/ai2thor/bin/python" \
  "$ROOT/sparx_agency/tasks/mapping/scene_graph/serve/smoke_detector_pipeline.py" \
  --episodes "$HOME/datasets/objectnav/robothor" --scene FloorPlan_Train3_1 \
  --url http://127.0.0.1:18096 --backend llmdet --confidence 0.60 --steps 12 \
  --output "$HOME/objnav_detector_dev/llmdet-smoke.json"
```

This runs the existing HeadlessObjNavAgent, scene graph, RPT*, weighted A* and
actual Unity actions. Room-language responses are an **explicit deterministic
fixture**, not a real LLM; this avoids letting a cold LLM take the rendering
GPU. Visibility metadata is read only after decisions for false-STOP scoring.
The method never receives instance masks/goal positions. These short checks
are plumbing/confirmation diagnostics, not navigation success-rate estimates.

## Lightweight validation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  sparx_agency/core/mapping/detection/tests sparx_agency/tasks/mapping/scene_graph/tests \
  sparx_agency/tasks/planning/objnav_benchmark_runtime/tests -q
.venv/bin/python -m sparx_agency.tasks.mapping.scene_graph.serve.detection_server --selftest
```

The IDE may flag the **intentional optional imports** in the model/provisioning/
simulator modules; use the appropriate interpreter, not a Torch install into
the core environment. Tests verify that importing/constructing either backend
does not import model packages.
