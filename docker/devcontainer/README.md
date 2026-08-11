# Dev container: PC + Sphera (ROBOTICAN Rooster)

A dev container for this project's own code -- perception, mapping, the
ROBOTICAN adapters. **Not** FALCON or the ROS1 bridge (they have their own
containers under `sparx_agency/tasks/planning/falcon/`), and **not** the
`launcher_ui_mapping.py` GUI (that stays a host-side control surface that
SSHes into the Jetson).

Built for **this machine** (host `pcn87652`, driver reporting CUDA 13.0,
`user1`/uid 1000) matching the ROS2 Humble build already run by Sphera's own
`drone_simulator` container. If you're building this on a different machine,
read the callouts below before you start -- several values are host-specific
on purpose (see `sparx_agency/robots/ROBOTICAN/DOME_CAPTURE_README.md` for
why).

## Before you build

1. **Stage the vendor message sources.** They live outside this repo, at
   `~/rqs_iai_ws/src/rooster_interfaces`, and a Docker build context can't
   reach outside `docker/devcontainer/`:

   ```bash
   ./docker/devcontainer/stage_vendor_msgs.sh
   ```

   This copies them into a gitignored `docker/devcontainer/vendor/` dir. Skip
   this and the build fails on the `COPY vendor/rooster_interfaces` line --
   that's intentional, not a bug.

2. **Check `nvidia-ctk` is registered** (it already is on this machine):

   ```bash
   docker info | grep -i nvidia
   ```

## Build

Either open the repo in VS Code and **"Reopen in Container"** (uses
`.devcontainer/devcontainer.json` directly), or build by hand:

```bash
docker build -f docker/devcontainer/Dockerfile -t theagency-dev:humble \
  --build-arg BASE_IMAGE=nvidia/cuda:13.0.0-cudnn-devel-ubuntu22.04 \
  docker/devcontainer
```

This takes a while the first time (ROS2 Humble + the vendor interfaces build
+ pip installs). If your GPU driver reports a different CUDA version than
13.0 (`nvidia-smi` at the top), pass a matching `BASE_IMAGE` -- see the
comment block at the top of the `Dockerfile` for how that default was chosen.

## Verify the basics (no Sphera needed)

```bash
docker run --rm --gpus all theagency-dev:humble \
  python3 -c "import rclpy, tensorrt, pycuda.driver, cv2, gi; print('ok')"

docker run --rm --gpus all theagency-dev:humble \
  python3 -m sparx_agency.tasks.common.hardware.detect
```

The second command should print your GPU name, `sm`, and a `target_tag` --
compare `sm` against `nvidia-smi --query-gpu=compute_cap --format=csv`.

## Verify against Sphera (the real gate)

This is the one unproven link in the plan: the container is Humble, Sphera's
FCU backend (`R1`/`it`) is Foxy. `sphera:drone_simulator` itself is already
Humble, so this should work for the topics that matter, but confirm it:

1. Start Sphera and the `it` container yourself (this repo doesn't start
   them -- see `.claude/skills/fly-rooster-sphera/SKILL.md` if you need the
   exact steps).
2. Open this repo in the dev container (VS Code "Reopen in Container", or
   `docker compose`/`docker run` with the flags in `devcontainer.json`).
3. From inside the container:

   ```bash
   ros2 topic list                              # expect /R1/* topics
   ros2 topic echo /R1/localization --once       # geometry_msgs/PoseStamped
   ros2 topic echo /R1/rooster_status --once     # std_msgs/String
   ```

   If `ros2 topic hz` hangs, plain `timeout` won't kill it --
   use `timeout -s KILL 10 ros2 topic hz /R1/localization`.

**If topics don't show up or `echo` hangs with no data:** this is the
`Humble ↔ Foxy` link failing. Rebuild with `--build-arg ROS_DISTRO=jazzy`
(matches the proven-working host config) and re-run this check before doing
anything else -- don't debug further up the stack until this passes.

**If `ros2` itself won't start or segfaults:** almost certainly the
CycloneDDS NIC. Three config files exist on this host and two are decoys --
see the comment in `devcontainer.json`'s `containerEnv` and
`DOME_CAPTURE_README.md:59-87`. Check which NIC `R1` actually picked:

```bash
docker logs R1 | grep -i "selected arbitrarily"
```

and make sure `~/rqs_iai_ws/src/cyclonedds.xml`'s `NetworkInterfaceAddress`
matches it -- `docker-compose.json`'s `CYCLONEDDS_URI` mount point tracks
that same file, so fixing it there fixes both the container and the host.

## Verify the model registry / depth

```bash
docker run --rm --gpus all \
  -v ~/depth_anything_ws:/home/user1/depth_anything_ws:ro \
  -e SPARX_MODEL_CACHE=/home/user1/.cache/sparx_agency/models \
  -v ~/.cache/sparx_agency:/home/user1/.cache/sparx_agency \
  theagency-dev:humble \
  python3 -m sparx_agency.tasks.common.model_registry.cli path \
    --model da3_metric_large --role depth_only --precision fp16 \
    --resolution 546x364 --no-download
```

Should print the existing engine's path with `origin=legacy` on stderr, no
network involved -- the same check `entrypoint.sh` runs automatically (as a
warning, not a blocker) every time the container starts.

## Known boundary: what stays outside this container

`sphera_common_interfaces` (used by `rooster_ground_truth_localization.py`
and the vendor `examples/`) has **no `.msg` source anywhere on this
machine** -- only a prebuilt Foxy binary inside Sphera's own image. That code
keeps running inside Sphera's `it` container via `docker exec` (which is why
`/var/run/docker.sock` is mounted in). Don't try to rebuild it here.

## Things worth knowing before you hit them

- **UDP port 5001 takes exactly one listener.** Running `ui.py`'s local
  preview while `rooster_frame_dir_publisher.py` is also capturing splits
  the stream and neither side gets a full feed -- easier to trip now that
  the container shares the host's network namespace.
- **Don't hardcode `DISPLAY=:0`.** It's been `:1` on this PC; the
  `devcontainer.json` passes it through from your session instead.
- **`~/rqs_iai_ws` on this host is a stale Foxy-era build.** The container
  builds its own vendor interfaces for Humble; the host `install/` tree is
  reference-only, read-only-mounted for its `cyclonedds.xml` and `.msg`
  sources.
- The nine `run_*.sh` wrappers under `sparx_agency/robots/ROBOTICAN/` still
  `exec .../venv/bin/python` -- a host-only Python 3.12 path. Running them
  *inside* this container needs the `SPARX_PYTHON` indirection described in
  the plan (not yet applied -- see the "long tail" migration task).
