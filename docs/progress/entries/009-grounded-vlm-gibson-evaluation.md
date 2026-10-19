# 009 - Native grounded-only Gibson evaluation

**Branch:** `feat/objnav-grounded-vlm-nadav`
**Status:** in progress
**Roadmap item:** [Native grounded-only evaluation](../ROADMAP.md)

## Goal

Run the complete existing ObjectNav algorithm in three distinct Gibson buildings,
three episodes each, with Grounding DINO Base + BLIP-2 exclusively and a video
for every episode. Freeze settings before evaluation and retain all failures.

## Steps

- [x] Preserve current user changes and confirm the requested feature branch.
- [x] Validate user-downloaded Habitat trainval and ObjectNav v1.1 archives.
- [x] Provision an isolated Habitat runtime and dedicated room-language service.
- [x] Validate explicitly authorized GPU deployment without changing defaults.
- [x] Generate nine deterministic starts, preflight all services and freeze jobs.
- [ ] Execute all nine episodes without navigation/safety/scoring changes.
- [ ] Decode all episode videos and report per-episode/aggregate metrics.
- [ ] Record limitations, retained failures and final artifact paths.

## Constraints

- No YOLO, policy tuning, changed action allowance, easier replacement episodes,
  hidden retries, or scoring changes. Use the existing frontier/RPT pipeline and
  multi-story development/2 protocol, not a claim of published validation.
- Seed 17, three buildings, three episodes each; omit `--record-first` so every
  episode is recorded. Freeze all data/source/model/precision identities before
  the first reset. Keep failed attempts separate after any configuration change.
- No commit or push. Preserve prior user edits, including untracked recordings.

## Notes

- 2026-09-22: Initial readiness was blocked before any native episode; evidence
  remains under `runs/grounded_vlm_gibson_3x3_20260922T074704Z/`.
- 2026-09-22: The user supplied both licensed archives and explicitly authorized
  the RTX 4090 (24 GB) for this workload, optionally permitting a stronger room
  model. No unrelated GPU owner may be terminated. Record shared-GPU authorization
  explicitly; retain strict refusal as the default for other runs.
- 2026-09-22: Resumed evidence and provisioning logs are under
  `runs/grounded_vlm_gibson_3x3_20260922T084950Z/`. The annotations passed ZIP CRC
  and restricted semantic-map loading; all 25 training scene maps are present.
- 2026-09-22: Isolated Habitat 0.2.4 / Python 3.9 / NumPy 1.26.4 provisioned in
  `~/miniconda3/envs/objnav-habitat`; 125 focused regressions passed. A first
  native preflight exposed missing PyYAML; its failure is retained and the
  dependency is now declared. The second preflight passed real RGB-D rendering,
  grounded-only HTTP inference and video decoding in all three selected scenes.
- 2026-09-22: Full-budget Grounding DINO + BLIP-2 use explicit CUDA bfloat16;
  Qwen 2.5 3B stays on a dedicated CPU-only Ollama service at localhost:11435.
  No larger language model was needed. User's localhost:11434 service is untouched.
- 2026-09-22: Frozen jobs are Ranchester, Pomaria and Hanson, three episodes each,
  all nine recordings enabled. Allensville was rejected by the policy-independent
  multi-story generator (only one supported level), not by navigation outcomes.
  `campaign/locks/` and `execution-request.json` record source, assets, model
  identities, precision, services, seed, protocol and the six-hour per-building
  wall-clock watchdog. The 500-action allowance is unchanged. Batch launched at
  13:24 local time; Python source changes are forbidden until it finishes.

## Result

Pending native evaluation; no success or completeness claim yet.

