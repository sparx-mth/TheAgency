# Roadmap

High-level direction only — no implementation detail here. Each line becomes a progress
entry under `entries/` when work actually starts on it.

## Now
- [ ] Native grounded-only Gibson evaluation: three buildings, three episodes each, all videos and failures retained → `entries/009-grounded-vlm-gibson-evaluation.md`
- [ ] ObjectNav exploration efficiency: no redundant spinning, no route-interrupting camera inspections, frontiers ranked by worth → `entries/010-objnav-exploration-efficiency.md`
- [ ] ObjectNav room-search loop: bounded in-room exploration, then re-classify → re-estimate → re-order → transit to the next room's nearest frontier → `entries/011-objnav-room-search-loop.md`
- [ ] ObjectNav robustness: a failed plan or model keeps the agent exploring (nearest frontier, never a spin), stairs from the simulator's ground truth with an explicit up/down decision, YOLO-only detector default → `entries/012-objnav-fallback-gt-stairs-yolo-default.md`
- [ ] The published Gibson ObjectNav validation benchmark (1,000 episodes, 5 scenes, SR/SPL/DTG) run to completion, detached and monitored → `entries/026-full-gibson-val-run-detached-monitor.md`
- [ ] ObjectNav target closing: an image-space pitch servo (the box in the frame drives LOOK_DOWN/LOOK_UP), a whole-box confident STOP, and the toilet fixes verified against couch/tv/bed → `entries/027-bbox-pitch-servo-and-closing-regression-check.md`
- [ ] The published Gibson val benchmark re-run with the toilet closing fixes, SAM off (A) then SAM on (B), and the two compared against each other and the 2026-10-08 baseline → `entries/028-gibson-val-rerun-closing-fixes-sam-off.md`
- [ ] The published Gibson val benchmark on the 8 GB laptop: `qwen2.5:3b-instruct` in both LLM roles, box depth (SAM off), lean records, resumable across a reboot, with the per-episode data a failure analysis needs → `entries/029-gibson-val-3b-laptop-run.md`
- [ ] ObjectNav on HM3D v1/v2: one shared navigation core (RPT*, room-search loop, fallback, YOLO default) behind two task modules, GibsonTask and HM3DTask → `entries/013-objnav-shared-core-hm3d-task.md`
- [ ] The published Gibson val benchmark re-run on the search and closing fixes of 2026-10-13/14 (frontier exploration, the LOOK pair, the STOP overshoot), compared per scene and per category against the 2026-10-12 run → `entries/030-gibson-val-rerun-search-closing-fixes.md`
- [x] ObjectNav exploration ablation (SR 0.665 / SPL 0.337 / DTG 1.75 m vs 0.792 / 0.464 / 0.94 m): goal-free planar FALCON + the same YOLO perception + the same closing, no LLM/RPT*/rooms, fifteen recorded episodes for review and then the 1,000-episode Gibson val split beside the full method's SR/SPL/DTG → `entries/032-falcon-yolo-exploration-baseline.md`
- [x] ObjectNav target closing for a television: the box's range read around a surface standing in front of the object, and a standoff ring that widens to reach a big object instead of releasing a correct lock → `entries/031-tv-closing-occluder-range-and-widened-standoff.md`
- [ ] FALCON BEV click-to-fly for Rooster/Sphera, end-to-end and reliable → `entries/001-falcon-rooster-clickfly.md`
- [ ] Move every Rooster node still running on the bare host into `robotican_dev`/`theagency:robotican` → `entries/002-rooster-full-containerize.md`
- [ ] Incoming updated FALCON/planning drop from the user — integrate and re-verify → `entries/003-falcon-planning-update.md`
- [ ] BEV click-to-fly: smooth, occupancy-aware navigation to the clicked point → `entries/004-occupancy-aware-navigation.md`
- [ ] YOLO-detected object label + position ("barrel") as a FALCON navigation goal → `entries/005-yolo-object-navigation.md`
- [ ] Calibrate/tune the FALCON->Rooster velocity controller against real drone behavior → `entries/007-rooster-velocity-controller.md`

## Next
- [ ] Track down the noisy/speckled occupancy map (vendor CDR corruption suspected, not confirmed as the full explanation)
- [ ] Resolve which battery-capacity config file Sphera actually reads (`rqs7-private-parameters` vs `rooster-private-parameters`, currently `10000.0` vs `1000.0`)
- [ ] Split today's Sphera/FALCON fixes off `create_devcontainer_daphna` onto their own `fix/` branch
- [ ] XTEND sibling Docker layer (`Dockerfile.xtend`) reusing `base_cuda`/`ros2_humble`/`perception`

## Later / ideas
- Long-tail migration of remaining DA3-consumer scripts/launch files to the model registry (explicitly deprioritized for now)

## Done
- [x] Selectable YOLO / Grounding DINO + BLIP-2 perception for Gibson, with auditable visual verification → `entries/008-grounded-vlm-perception.md`
- [x] GPU hardware-detection consolidation, model/engine registry, x86 dev container for Sphera/ROBOTICAN (commit `55f96e33`)
- [x] Rooster/Sphera takeoff, video streaming, depth `.npy` output, closed-loop altitude hold
- [x] Rooster frame capture → direct relay to the Jetson, orchestrated via mission_control.py → `entries/006-rooster-frame-jetson-relay.md`
