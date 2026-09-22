# Selectable YOLO / Grounding DINO / BLIP-2 perception

Grounding DINO is the language-conditioned box detector from the OSG paper,
**not DINOv2**. BLIP-2 answers visual questions; it does not supply boxes, depth,
calibrated probabilities or permission to move. This is a perception integration,
not an implementation or reproduction of OSG Navigator's scene-graph planner.

## One setting selects the mode

Edit **`backend`** in [configs/gibson_perception.json](configs/gibson_perception.json):

| Value | Loaded models | Use |
|---|---|---|
| `yolo_world` | YOLO-World X-v2 + its CLIP text encoder | Original YOLO-only path |
| `grounded_vlm` | Grounding DINO Base + BLIP-2 FLAN-T5 XL | No YOLO dependency at runtime |
| `hybrid` | All of the above | Independent YOLO and DINO proposals, then verification |

The example selects `hybrid`; the **service's no-config default remains YOLO**.
`grounding_dino` is also selectable for detector-only ablation, and `llmdet`
remains available. CLI flags override JSON. Unused model paths are not loaded or
required to exist. Unknown settings and explicitly blank paths fail. Relative
paths in JSON resolve against its directory; `~` expands to the current user's home.

## Setup and weights

Use a separate model environment, never install model dependencies into the
lightweight core environment. On this machine the dedicated environment is
`~/.venvs/objnav-detector`, reusing the existing navdp Torch installation read-only.
On another machine first supply a matching Torch/torchvision pair for its device.

```bash
# Run from the repository root; create this environment only if it is absent.
"$HOME/miniconda3/envs/navdp/bin/python" -m venv --system-site-packages "$HOME/.venvs/objnav-detector"
DETECT_PY="$HOME/.venvs/objnav-detector/bin/python"
"$DETECT_PY" -m pip install -r sparx_agency/tasks/mapping/scene_graph/serve/requirements-grounded-vlm.txt
CUDA_VISIBLE_DEVICES="" "$DETECT_PY" -m sparx_agency.tasks.mapping.scene_graph.serve.provision_grounded_vlm \
  --output "$HOME/models/objnav" --include-yolo
```

Provisioning pins immutable revisions and verifies downloaded bytes against the
publisher's hashes. Interrupted downloads resume only for that recorded revision;
unrelated existing directories are never overwritten. A pending/incomplete or
missing shard is rejected by inference. `source.json` records the provenance.

| Artifact | Selected release | Weight size, approximately | Terms |
|---|---|---:|---|
| Grounding DINO Base | `IDEA-Research/grounding-dino-base`, revision `12bdfa31` | 0.93 GB | Apache-2.0 |
| BLIP-2 FLAN-T5 XL | `Salesforce/blip2-flan-t5-xl`, revision `0eb0d3b4` | 15.77 GB, two shards | Publisher model card: MIT |
| YOLO-World X-v2 | Official Ultralytics v8.3.0 assets | See downloaded checkpoint | Ultralytics AGPL/commercial terms |
| CLIP ViT-B/32 | OpenAI checkpoint; pinned Ultralytics CLIP implementation | About 0.35 GB | MIT |

Weight size is **not peak memory**. The selected model versions are explicit
integration choices; the OSG paper does not identify these exact revisions.
YOLO and CLIP are optional provisioning with `--include-yolo`. All artifacts stay
under `~/models/objnav`, outside git. The example passes CLIP explicitly so that
hybrid inference never downloads it or installs CLIP implicitly.

## Start the service

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 YOLO_AUTOINSTALL=false \
  "$HOME/.venvs/objnav-detector/bin/python" -m sparx_agency.tasks.mapping.scene_graph.serve.detection_server \
  --config sparx_agency/tasks/mapping/scene_graph/serve/configs/gibson_perception.json
```

Add `--backend grounded_vlm` or `--backend yolo_world` to switch without editing
the file. Start only one mode per service/port. The example binds localhost:18100
and uses the exact ordered Gibson vocabulary; other datasets must supply their
own `--classes` from their adapter. Do not change another mission's service.

The original one-scene GUI's automatic service launcher remains YOLO-specific;
use these CLI-configured services and the evaluation CLIs for the new modes.

For a normal Gibson run, append to the existing dataset/episode command:

```bash
--detector-url http://127.0.0.1:18100 --detector-backend hybrid --detector-timeout-s 300
```

These flags work in `gibson.run`, `run_development`, `distinct_buildings`,
`compare_explorers` and `stair_diagnostic`. Campaign launchers forward the timeout
and freeze it with detector/config/source identity. Longer CPU campaigns can set
`--job-timeout-s`; this changes a wall-clock watchdog, not the action allowance.
Never resume a frozen campaign after changing the model, source or settings.

### Explicitly authorized shared GPU

The occupied-GPU refusal remains the default. On an operator-authorized device
with enough free VRAM, pass `--allow-shared-gpu` to both the detector service and
the Gibson campaign. The detector JSON also accepts the boolean
`allow_shared_gpu`; `--no-allow-shared-gpu` explicitly revokes it. Check all GPU
owners and headroom before launch; this opt-in never kills another process and
does not guarantee isolation. CUDA availability and the detector's 70% allocation
cap still apply. Health metadata and frozen evaluation configs record sharing.

Choose CUDA precision explicitly and keep it fixed for the entire campaign.
The initial CPU smoke identities are not GPU identities. Prefer a CPU-only room
LLM when perception and Habitat share the card. No automatic model or precision
fallback is added. GPU permission does not change navigation or scoring rules.

## Verification and tunable settings

- Both selected detectors see **every frame**. DINO can propose a stair that YOLO
  missed or called furniture. Same-label overlapping boxes keep the first
  source's score (YOLO first in hybrid); scores are never summed or averaged.
- BLIP-2 sees a full image marked with the candidate rectangle and an exact
  crop. Both answers must be exactly yes. No, uncertain and unrecognized output
  are withheld. Two views of the same image are **one observation**, not two
  confirmations. The original multi-view target gate still applies.
- `verify_labels: "*"` verifies all classes by default. A comma-separated list,
  such as `stairs,staircase,bed,couch,sofa`, explicitly allows other classes to
  pass without VLM checking; those rows are marked `not_requested`.
- `max_verifications` bounds boxes checked per frame (default 8); **overflow is
  withheld**, not silently accepted. Stairs are prioritized. This can reduce
  object/door recall and must be calibrated on real data.
- Conflicting overlapping confirmed stairs/furniture are quarantined. Real beds
  beside stairs are not blacklisted. The verifier does not relabel boxes or
  invent a localization from an answer.
- `conf` is the DINO emission floor, not a STOP threshold. `yolo_conf` can set a
  separate YOLO floor. Downstream filtering still uses original scores; no
  cross-model calibration or accuracy improvement is implied by matching values.
- HTTP responses contain raw proposals, sources, scores, verification answers,
  rejection reasons and accepted indices. A SHA-256 binds them to the submitted
  JPEG. Gibson records them under `method.perception.detector_evidence` in
  `steps.jsonl`; mutable answers never enter static `/health` identity.
- Once a composite service is loaded its vocabulary is immutable; restart it to
  change classes. Missing weights, model failures and malformed evidence fail
  explicitly; there is no YOLO fallback on VLM failure.

Depth and pose transforms, floor geometry, step limits, traversal, camera
ownership, map isolation and STOP confirmation are unchanged. This RGB service
does not accept geometric stair proposals directly; existing depth-only discovery
still runs independently. If both RGB detectors miss a stair, BLIP-2 cannot create
a box by itself. A later geometry-triggered visual-query interface remains separate.

## Validation and limits

On 2026-09-22 all requested snapshots were downloaded and verified. Real CPU
inference checked the saved Grounding DINO decoder values and BLIP-2 question
answering. All three modes also passed actual HTTP client/server smoke checks.
The smoke input is synthetic black RGB, **not a labeled accuracy evaluation**;
the verification budget/detection cap were reduced explicitly for this check.

Focused detector/HTTP/multi-floor regressions: **165 passed**; service self-test
also passed. Full detector/scene-graph/runtime regression run: 589 passed, one visualization
test failed. The same door-label rendering failure reproduces on an untouched
HEAD archive with the same interpreter. It was not masked or fixed by changing
perception. Artifacts are under `~/papers/2508.04678/` on this workstation.
Use `http-final-grounded_vlm.json`, `http-final-hybrid.json` and
`http-smoke-yolo_world.json` for final mode checks; earlier failed loader and
selective-verification logs are retained only as diagnostics.

The model environment inherits an unused Gradio dependency requiring Pillow<12;
the detector overlay uses Pillow 12.3.0. Gradio is not used, and the parent navdp
environment was not modified. Tests use the existing project `venv`; the machine's
documented `.venv` lacks pytest. The missing test dependency scikit-fmm was added
only to that test venv.

The initial CPU smoke checks did **not use** the RTX 4090: its graphics allocation exceeds the
service's conservative occupied-GPU gate. CUDA requires an idle/dedicated device
or the later explicit sharing opt-in above, and an explicit dtype. No automatic
precision fallback is provided.
No Jetson/control-loop rate, Gibson staircase recall or native navigation gain is
claimed. Validate labeled recorded stairs and hard negatives before trusting the
new mode, then run a frozen native multi-floor comparison.

