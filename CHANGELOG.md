# Changelog

All notable changes to this project are logged here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/) — one line per change, written for
future-you, not for a commit log.

## [Unreleased]
### Added
- Closed-loop PD altitude hold for the ROBOTICAN Rooster (`rooster_unit.py`), replacing
  the previous open-loop throttle constant that reliably drifted to floor or ceiling.
- `demo_mode_topic`/`demo_mode_request_topic` params on `nav_stack.launch`, letting
  `sphera_drone.launch` route Rooster's demo-mode handshake through `/R1/...` topics
  instead of the XTEND-shaped `/xtend/...` defaults.
- `rooster_demo_mode_manager.py` (new, `falcon_adapter` ROS1 node): a minimal Rooster
  equivalent of `xtend_drone_demo_manager.py` — echoes a requested demo mode back as
  the authoritative current mode, with no other side effects (deliberately does not
  auto-land on FINISH the way the XTEND version does).

### Changed
- `sphera_drone.launch` now overrides FALCON's navigation controller to `multi_axis`
  for Rooster (a genuinely holonomic platform), instead of the default `roll_assist`.
- Every Y-axis bound in `maps/sphera_jail.yaml` (`init_y`, `map_min_y`/`map_max_y`,
  `box_min_y`/`box_max_y`, `vbox_min_y`/`vbox_max_y`) and the Y-axis launch args
  documented in the `fly-rooster-sphera` skill (`bev_ymin`/`bev_ymax`/`goal_y`) —
  negated and min/max-swapped to match the corrected localization sign (see Fixed).
- `map_max_z` raised `4.0` → `5.0` and `box_max_z`/`vbox_max_z` raised `1.8` → `4.0`
  in `maps/sphera_jail.yaml`, so RViz shows real room geometry up to the actual
  ceiling instead of truncating at an artificially low height, while keeping a
  real (1.0m) margin between the map and box/vbox bounds (see Fixed for why the
  margin matters).
- `cam_min_depth` raised `0.1` → `0.45` in `sphera_drone.launch` (see Fixed).
- `mapping_sync`'s `freeze_on_turning_mode` set to `false` in `sphera_drone.launch`
  (see Fixed) — turning-smear protection is temporarily disabled pending a real fix
  to the rotation-supervisor bug it exposed.
- Stale saved RViz camera position in `maps/sphera_jail.rviz` (`Focal Point Y: 14.66`)
  updated to `Y: -14.66` to match the corrected localization sign.

### Fixed
- `waypoint_follower_node.py`: `_publish_twist_multi` was defined twice in the same
  class; the second (older) definition silently shadowed the first and didn't accept
  the `vz` keyword the caller passed, so every navigation tick threw a `TypeError`
  before ever calling `.publish()`. FALCON's internal state showed `nav=RUN` the whole
  time, but `/cmd_vel` was never actually emitted — the drone never moved toward a
  clicked BEV goal regardless of controller/topic configuration.
- `rooster_ground_truth_localization.py`: `position.y` was passed straight through
  from Sphera/Unreal telemetry while yaw was already negated for the left-handed →
  right-handed conversion, leaving position and rotation handedness inconsistent.
  Completed the conversion by negating `position.y` too (see LESSONS.md for how this
  was verified).
- Click-to-fly deadlock for Rooster: the default `roll_assist` controller's demo-mode
  confirmation handshake never resolves because nothing publishes to `/xtend/demo_mode`
  for this platform — fixed via the `multi_axis` controller switch above (Rooster is
  holonomic and doesn't need the turn-then-forward handshake `roll_assist` requires).
- Noisy/speckled voxel map: the bottom ~25% of every RGB frame showed a near-constant
  `~0.17-0.35m` depth reading (the camera rig/mount itself, visible in its own FOV,
  not real environment) that was below `cam_min_depth` and so got fused as a permanent
  phantom wall directly ahead of the drone on every frame — very likely the real cause
  of "boxed in, no A* route" seen the same day. Fixed by raising `cam_min_depth`
  (see Changed).
- `exploration_node` crashing (`voxel_mapping::ESDF::getDistance` → glog FATAL
  "Address out of range") introduced by setting `vbox_max_z` exactly equal to
  `map_max_z` — ESDF's neighbor-cell queries need a real margin near a boundary, not
  just `map ⊇ vbox` ordering. Fixed by raising `map_max_z` instead (see Changed).
- RViz appearing completely empty (no error, just nothing rendered): a stale saved
  camera focal point in `sphera_jail.rviz` from before the Y-axis fix pointed the
  camera at the mirror-image empty location where the room used to be under the old
  sign convention (see Changed).

### Known issues (not yet fixed)
- `waypoint_follower_node.py`'s rotation supervisor gets stuck permanently requesting
  `"turning"` mode while its navigation loop targets the default startup goal with no
  real flight dynamics ever confirming a turn is complete — this permanently froze
  `mapping_sync`'s rotation-freeze mechanism once `rooster_demo_mode_manager.py` made
  it actually engage. Worked around by disabling `freeze_on_turning_mode` for now
  (see Changed); the supervisor bug itself is unresolved.
- Possible left/right (lateral) mirroring in the built map: reported once during a
  BEV-driven flight (drone next to the real left wall in Sphera; map showed it next
  to the right wall). Forward/back and altitude are both confirmed correct via
  quantitative ground-truth tests, so this is a different bug from the Y-axis fix
  above — most likely in how the camera's local lateral axis projects into world-frame
  points, not the drone's own tracked pose. Not yet conclusively confirmed or
  root-caused; a physical landmark was added to the test hallway to check this
  properly next session.

<!--
Example of a real entry once you have one:

## [Unreleased]
### Fixed
- Hover z-axis drift at hover_z=560 traced to accumulated integral windup in the altitude
  controller, not the sim's ranger noise as first suspected. See LESSONS.md.
-->
