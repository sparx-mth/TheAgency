# 008 - Selectable grounded visual verification

**Branch:** `feat/objnav-grounded-vlm-nadav`
**Status:** done
**Roadmap item:** [Selectable perception](../ROADMAP.md)

## Goal

Provision official Grounding DINO Base and BLIP-2 FLAN-T5 XL weights and expose
three simple modes: unchanged YOLO only, Grounding DINO with BLIP-2 verification,
and YOLO plus Grounding DINO with BLIP-2 verification. Preserve the existing
RGB/pixel-box HTTP contract and observed-depth navigation safeguards.

## Why

Stairs mistaken for furniture can prevent discovery or create false goals.
Independent text-guided proposals and image-aware verification should be
testable without deleting YOLO or equating synthetic tests with accuracy.

## Steps

- [x] Confirm Grounding DINO, not DINOv2; inspect service, registry and consumers.
- [x] Add pinned, checksum-verified provisioning; download weights outside git.
- [x] Add lazy, offline Grounding DINO and BLIP-2 runtimes with explicit settings.
- [x] Compose selectable modes with conservative verification and raw diagnostics.
- [x] Wire service configuration, provenance, clients and Gibson selection.
- [x] Test mode isolation, contracts, uncertainty, duplicate evidence and failures.
- [x] Run real model/HTTP smoke checks without taking the rendering GPU.
- [x] Update package runbooks, changelog and lessons with measured limitations.

## Open questions

- Stair accuracy requires labeled Gibson recordings, absent from this machine;
  no improvement claim or automatic promotion of the experimental default.

## Notes

- 2026-09-22: User explicitly authorized weight downloads and confirmed Grounding
  DINO. Models selected: IDEA-Research/grounding-dino-base and
  Salesforce/blip2-flan-t5-xl. The paper does not identify its exact checkpoints.
- 2026-09-22: This machine has a 24 GiB RTX 4090, not the older 8 GiB example in
  CLAUDE.md. Its existing graphics allocation exceeds the service's conservative
  GPU gate. Use CPU for automatic validation; do not relax the gate or kill owners.
- 2026-09-22: Preserve user modifications to .gitignore and untracked recordings.
  Work is on a local feature branch; no commit, push or baseline merge is authorized.
- 2026-09-22: Grounding DINO Base's shared-head layout requires direct aliases to
  the saved decoder, unlike LLMDet's independent heads. Verified the six saved
  tensors and absence of meta parameters; see the new LESSONS.md entry.
- 2026-09-22: BLIP-2 now verifies all labels by default, not only stairs/furniture;
  otherwise an unexpected mislabel could bypass verification. Selective labels
  remain an explicit performance/recall trade-off. Budget overflow abstains.
- 2026-09-22: DINO, both BLIP-2 shards, X-v2 and CLIP downloaded and checked under
  `~/models/objnav`. The model environment is `~/.venvs/objnav-detector`. No GPU
  inference was run and no model weights were added to git.

## Result

Implemented three modes through one JSON `backend` setting: `yolo_world`,
`grounded_vlm`, `hybrid`. The no-config default remains YOLO. CLI launchers
forward registered modes and explicit HTTP timeouts. Existing depth geometry,
floor transitions and multi-view STOP safeguards are unchanged. The original
GUI auto-launcher is still YOLO-specific; new modes use the documented CLI.

Validation: **165 focused tests passed**; the broader relevant suites had
**589 passes and one door-label visualization failure**, reproduced on an
untouched HEAD archive. The service self-test passed. Real offline CPU DINO,
BLIP-2 and all three HTTP modes passed synthetic-frame smoke tests, with limited
verification/detection budgets recorded in their artifacts. These are not
staircase-accuracy or native navigation measurements.

Runbook: `sparx_agency/tasks/mapping/scene_graph/serve/GROUNDED_VLM.md`.
Configuration: `serve/configs/gibson_perception.json` beside that runbook.
Local logs: `~/papers/2508.04678/hybrid-final-tests.log`,
`baseline-viz-test.log`, `http-smoke-yolo_world.json`, and
`http-final-{grounded_vlm,hybrid}.json`. Earlier failed loader logs and the initial
selective-verification smokes are retained but superseded. See [CHANGELOG](../../../CHANGELOG.md)
and [LESSONS](../../../LESSONS.md). No commit or push was made.
