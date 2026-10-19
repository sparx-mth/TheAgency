# 014 - Office GPU visual navigation verification

**Branch:** `feat/objnav-habitat-gibson-nadav`
**Status:** cancelled by user; superseded by target-closing implementation (015)
**Roadmap item:** ObjectNav robustness (continues 013)

## Goal
Record two episodes each in Ranchester and Pomaria with full HUD, CUDA rendering
and detector inference, immediate completion metrics and an inspection opportunity.

## Steps
- [x] Inspect office GPU: RTX 4090, 24 GB; no resident compute workload at preflight.
- [x] Explicit CUDA 0 detector and local CLIP weights; verify health response.
- [x] Add evaluator-only DTG, room confidence and coordinate-based frontier HUD identity.
- [x] Add completion notification, video path, runtime/FPS and configurable inspection pause.
- [x] Complete regression validation (373 runtime tests before campaign execution).
- [x] Provision required Qwen 14B reasoning model (download authorized by user).
- [ ] Run all four recorded episodes and validate video frame counts and metrics.
- [ ] Report aggregate results and exact artifact paths.

## Notes
The active RPT* policy has no ZSON embedding stage or navigation torch tensors.
Navigation remains Python/NumPy/SciPy so this tests the updated algorithm rather
than an unvalidated GPU rewrite. YOLO-World and its CLIP text encoder use CUDA;
Habitat uses GPU rendering (OpenGL/EGL, not CUDA tensor rendering).
Record all decisions plus the terminal frame at 6 playback FPS; execution FPS is
actions / episode wall time. Inspection pauses are excluded from that denominator.
The selected frozen training-development manifest is from
`runs/fallback_gtstairs_yolo_20260928T130941Z/episodes.json`; no held-out claim.

## Result
Three episodes completed before the user's immediate termination request: Ranchester
000000/000001 and Pomaria 000000, all with SR=0 and SPL=0. Pomaria 000001 was
interrupted. The launcher, simulator, detector, encoder and dedicated LLM workers
were terminated; no batch restart was performed. Completed videos and partial
artifacts remain in `runs/office_gpu_visual_sanity_20261001`. This is not a
completed four-episode campaign or a measurement of the subsequent closing fix.
