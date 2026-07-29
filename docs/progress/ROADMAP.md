# Roadmap

High-level direction only — no implementation detail here. Each line becomes a progress
entry under `entries/` when work actually starts on it.

## Now
- [ ] FALCON BEV click-to-fly for Rooster/Sphera, end-to-end and reliable → `entries/001-falcon-rooster-clickfly.md`
- [ ] Move every Rooster node still running on the bare host into `robotican_dev`/`theagency:robotican` → `entries/002-rooster-full-containerize.md`
- [ ] Incoming updated FALCON/planning drop from the user — integrate and re-verify → `entries/003-falcon-planning-update.md`
- [ ] BEV click-to-fly: smooth, occupancy-aware navigation to the clicked point → `entries/004-occupancy-aware-navigation.md`
- [ ] YOLO-detected object label + position ("barrel") as a FALCON navigation goal → `entries/005-yolo-object-navigation.md`

## Next
- [ ] Track down the noisy/speckled occupancy map (vendor CDR corruption suspected, not confirmed as the full explanation)
- [ ] Resolve which battery-capacity config file Sphera actually reads (`rqs7-private-parameters` vs `rooster-private-parameters`, currently `10000.0` vs `1000.0`)
- [ ] Split today's Sphera/FALCON fixes off `create_devcontainer_daphna` onto their own `fix/` branch
- [ ] XTEND sibling Docker layer (`Dockerfile.xtend`) reusing `base_cuda`/`ros2_humble`/`perception`

## Later / ideas
- Long-tail migration of remaining DA3-consumer scripts/launch files to the model registry (explicitly deprioritized for now)

## Done
- [x] GPU hardware-detection consolidation, model/engine registry, x86 dev container for Sphera/ROBOTICAN (commit `55f96e33`)
- [x] Rooster/Sphera takeoff, video streaming, depth `.npy` output, closed-loop altitude hold
- [x] Rooster frame capture → direct relay to the Jetson, orchestrated via mission_control.py → `entries/006-rooster-frame-jetson-relay.md`
