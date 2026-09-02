# 008 - New Sphera: domain 9 -> 6, and getting robotican_dev to actually talk to it

**Branch:** `feat/new_sphera_rooster_container_daphna`
**Status:** container test passing; Rooster flight still blocked (Sphera-side, not this repo)
**Date:** 2026-09-02

## Goal
Test the ROBOTICAN/Rooster backend container (`robotican_dev`, `theagency:robotican`) against
the new Sphera simulator build, on `ROS_DOMAIN_ID=6` (the new build's replacement for the old
`9`) - not `9`, and not `5` (that's XTEND/Jetson's domain, a different stack).

## What changed in this repo

### Domain 9 -> 6, everywhere it's live
Swept every **live operational** Rooster/Sphera reference to `ROS_DOMAIN_ID=9` to `6`:
`docker-compose.robotican.yml`, `docker-compose.detector.yml`, both `.devcontainer/*.json`,
`sparx_agency/tools/mission_control.py`, `sparx_agency/tools/rooster_turn_debug.py`, the
ROBOTICAN `run_*.sh` wrappers, `MISSION_CONTROL_INTEGRATION_PLAN.md`, `DOME_CAPTURE_README.md`,
`README.md`, `README_2.md`'s domain table, the FALCON bridge's `README.md`/`bridge.yaml`/
`run_bridge.sh`, `run_object_mission_sphera.sh`, and Demo No.5's launcher UI. Also patched the
(gitignored) `fly-rooster-sphera` skill runbook so the commands in it match.

Deliberately **left alone**: `LESSONS.md` and `docs/progress/entries/006-*.md` - dated
postmortems, not live config; rewriting `9` there would misrepresent what was actually true when
they were written.

### robotican_dev couldn't decode the new Sphera's own telemetry
`theagency:robotican` didn't exist as a built image yet (nothing in `docker images` matched),
so it got built from scratch via `docker/stage_vendor_msgs.sh` + `docker buildx bake`. First
build succeeded, but `robotican_dev` could see `/R1/sphera/state`/`/R1/sphera/set_state` in
`ros2 topic list` (DDS discovery doesn't need type info) and still fail to decode them:

```
The message type 'sphera_common_interfaces/msg/SpheraPawnState' is invalid
```

Root cause: `docker/stage_vendor_msgs.sh` only ever vendored `rooster_interfaces` (the old
`rqs_iai_ws`/`it`-container FCU backend's messages). `sphera_common_interfaces` - the package
that defines `SpheraPawnState`, carried by the new Sphera engine's own
`/R1/sphera/state`/`set_state` topics - was never wired in. `Dockerfile.robotican`'s own
comment said this was intentional ("it has no .msg source anywhere, only a prebuilt Foxy
binary"), which was true for the old Sphera build but isn't anymore: the new one ships real
`.msg`/`.srv` source at `~/sphera_ws/src/sphera_common_interfaces`.

Fixed by extending [`stage_vendor_msgs.sh`](../../../docker/stage_vendor_msgs.sh) to also stage
that package, and [`Dockerfile.robotican`](../../../docker/Dockerfile.robotican) to `COPY` and
`colcon build` it alongside `rooster_interfaces` in the same `/opt/rooster_ws` workspace (already
auto-sourced by `docker/entrypoint.sh`). Rebuilt - cache reused everything below the last layer,
so the rebuild took under a minute.

## Confirmed working
From inside `robotican_dev` itself, on `ROS_DOMAIN_ID=6`:
- `ros2 topic echo /R1/sphera/state --once` - decodes cleanly.
- `ros2 topic pub --once /R1/sphera/set_state ...` - publishes and round-trips (read back after
  writing, position stable).

## A host-machine fix that isn't in this diff
`~/rqs_iai_ws/src/cyclonedds.xml` pinned `NetworkInterfaceAddress` to `192.168.131.5`, which
isn't an interface on this host (`enp129s0` is `172.16.17.5`) - DDS couldn't bind at all until
this was corrected. This file lives outside the repo and isn't tracked here; if it ever reverts
or this runs on a different machine, re-check it against `ip -4 -o addr show` before assuming a
domain/repo issue.

## Not resolved - blocks actual flight, not a repo issue
`R1`'s own backend nodes (`fcu_driver`, `rooster_manager`, `video_handler`, all inside the
`sphera-backend:rooster` container - not `it`) crash on every start:

```
[R1.rooster_backend]: Param sitl.model not declared but accessed
```

Diffing the generated `~/.sphera/tmp/SpheraConfig/rooster_backend_Rooster_1.gen.yaml` against
the working `isr_backend_EVO_1.gen.yaml` shows the Rooster's config is missing an entire `sitl:`
block (`gcs_ip`/`gcs_port`/model) that the ISR (EVO) backend has. Placing the Rooster entity in
the world and recreating `R1` did not fix it - the file's `sphera.model` stays `""` regardless.
This looks like a missing SITL/FCU assignment for the Rooster specifically in this world's Sphera
scenario, not something fixable from the ROS/container/repo side. Needed before arm/takeoff can
be tested here.

## Bypassing `it` entirely for non-flight testing
Confirmed via `sphera_common_examples-main/docs` that `/R1/sphera/state` (read) and
`/R1/sphera/set_state` (write) are engine-level topics with no dependency on `it`,
`rqs_iai_ws`, or PX4/SITL - they work as long as the Sphera container itself is up. Useful for
container/connectivity tests like this one. Actual flight (arm/takeoff/land) still needs
`fcu_driver` alive one way or another (whether through `it`'s bridge stack or a direct MAVLink
client like the `sphera_mavlink_commands` example) - so the `sitl.model` gap above blocks both
paths equally.
