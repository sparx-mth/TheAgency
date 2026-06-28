# ROS1 ↔ ROS2 Bridge (FALCON stack)

The ROS1 Noetic ↔ ROS2 Foxy bridge that connects the **ROS2 side** (the SJTU
Gazebo sim or the real XTEND drone) to the **ROS1 side** (the FALCON planner in
`../`). It is the *only* component that does ROS1↔ROS2 message passing — the
FALCON adapters do not, and must not, bridge themselves.

It uses `parameter_bridge` (not `dynamic_bridge`): an explicit, QoS-aware bridge
that bridges exactly the topics listed in [`bridge.yaml`](bridge.yaml), each with
its own QoS. This matters because Gazebo's camera and the drone publish
**best-effort**, and `dynamic_bridge` subscribes reliable by default — DDS
silently drops data across that mismatch, so pose works but depth stalls
mid-flight. `bridge.yaml` pins `reliability: best_effort` on the sensor streams
to fix that.

The bridged topics are exactly the ones the FALCON adapters consume/produce
(`/xtend/depth_frame_path`, `/xtend/rgb_frame_path`, `/xtend/localization`,
`/cmd_vel`, `/xtend/demo_mode`, …) — see `bridge.yaml` and the adapter nodes in
`../adapter/scripts/`. Depth/RGB are no longer raw Images: the drone writes each
frame to disk and bridges a tiny `std_msgs/String` "<path> <sec> <nsec>" the
consumers load, which removes the per-frame image serialization cost.

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Builds `ros1_bridge:noetic-foxy` (clones + compiles `ros1_bridge`) |
| `bridge.yaml` | Topic list + per-topic QoS (sensor streams = `best_effort`) |
| `entrypoint.sh` | wait for roscore → `rosparam load bridge.yaml` → `parameter_bridge` (restart loop) |
| `run_bridge.sh` | Convenience wrapper: builds the image if missing, mounts config, persists the log to the host |
| `verify_bridge.sh` | Checks the FALCON-critical topics are actually flowing (Hz check) |
| `fastdds_no_shm.xml` | Fast-DDS UDP-only profile (no shared memory) — used by `run_bridge.sh` for reliable cross-container discovery |
| `fastdds_localhost.xml`, `cyclonedds_localhost.xml` | Alternative localhost-only DDS profiles for the two RMWs |

> `ros1_bridge` needs ~4 GB free RAM to compile; the build caps parallelism via
> `MAX_JOBS` (default 2). Override on memory-constrained hosts:
> `docker build -t ros1_bridge:noetic-foxy --build-arg MAX_JOBS=1 .`

## Build & run

```bash
cd sparx_agency/tasks/planning/falcon/bridge
./run_bridge.sh          # builds ros1_bridge:noetic-foxy on first run, then starts it
```

`run_bridge.sh` runs `--net=host --ipc=host`, bind-mounts `bridge.yaml` (edit +
restart, no rebuild needed), and tees the bridge output to `bridge.log` on the
host. It needs a reachable **roscore** (the FALCON container's `roslaunch`
provides one; the bridge entrypoint waits for it).

## Launch order

The bridge tolerates ordering (it waits for roscore and restarts
`parameter_bridge` on failure), but the clean sequence is:

1. **FALCON** — `cd .. && ./run_falcon.sh <env>` (its `roslaunch` starts the ROS1 master).
2. **Bridge** — `./run_bridge.sh` (connects to that master, starts bridging).
3. **ROS2 sim or real drone** — start the SJTU Gazebo sim / power the XTEND.
4. **Verify** — `bash verify_bridge.sh` (expects the depth/pose topics flowing).

## Environment variables

| Variable | Default (in `run_bridge.sh`) | Notes |
|---|---|---|
| `ROS_DOMAIN_ID` | `5` | **Must match** the ROS2 sim/drone |
| `RMW_IMPLEMENTATION` | `rmw_fastrtps_cpp` | **Must match** the ROS2 side (Foxy default fastrtps; Humble/Jazzy default cyclonedds) |
| `ROS_MASTER_URI` | `http://localhost:11311` | ROS1 master |
| `BRIDGE_YAML` | `/bridge.yaml` | Path inside the container (bind-mounted) |
| `BAG_DIR` | _(unset)_ | Optional: host dir to mount read-only at `/bag` for rosbag playback |

## Adding a topic

Add an entry to `bridge.yaml` (topic / type / QoS — use `best_effort` for sensor
streams) and restart the bridge. No rebuild — the YAML is bind-mounted. Adding a
new *message type* the image was not built with requires re-running the Dockerfile
build (it generates conversion code only for message pairs present at build time).
