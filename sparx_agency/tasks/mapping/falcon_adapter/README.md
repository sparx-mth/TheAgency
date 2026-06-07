# `tasks/mapping/falcon_adapter`

ROS1 (rospy) adapter nodes that run **inside FALCON's Noetic container** (the
`falcon_adapter` catkin package). They bridge FALCON's topics to the ROS-free
algorithms in `sparx_agency.core`. This is separate from `tasks/mapping/ros2/`,
which is our own ROS2 stack — the nodes here are ROS1 only.

## Files

- **`bev_publisher_node.py`** — subscribes to FALCON's occupied/free voxel
  clouds, runs `core.mapping.bev.BevProjector`, publishes a latched
  `nav_msgs/OccupancyGrid`. Replaces the legacy `bev_publisher.py`.
  **All rosparams + a launch snippet are documented in the file footer.**
- **`cloud_utils.py`** — `PointCloud2 -> (N,3) float32` (ROS1 twin of
  `tasks/mapping/ros2/helpers.py`).

## Running inside the FALCON container

The scripts here are mounted into `/catkin_ws/src/falcon_adapter/scripts/` by
`falcon_docker/run_hospital.sh`. Three things are needed:

1. **Make `sparx_agency` importable.** `run_hospital.sh` does not mount the repo
   or set `PYTHONPATH` by default — add to the `docker run` args:
   ```bash
   --volume "${SPARX_REPO}:/opt/sparx_agency:ro" \
   --env PYTHONPATH="/opt:${PYTHONPATH}"          # parent of the package on the path
   ```
   (`SPARX_REPO=~/GIT/TheAgency/sparx_agency`)

2. **Mount the scripts.** Copy `bev_publisher_node.py` and `cloud_utils.py` into
   `falcon_docker/adapter/scripts/` and add **both** names to the `for f in …`
   mount loop — `cloud_utils.py` must be listed too, or the sibling import fails.

3. **Launch** (replaces the old bev block):
   ```xml
   <node pkg="falcon_adapter" type="bev_publisher_node.py" name="bev_publisher" output="screen">
     <param name="resolution" value="0.15"/>   <!-- match FALCON's voxel size -->
     <param name="z_peak"     value="1.00"/>   <!-- flight altitude -->
   </node>
   ```

## Gotchas

- **Keep the import path light.** `import sparx_agency.core.mapping.bev` executes
  every `__init__.py` up the chain. `core/common/utils.py` imports `torch`; if any
  `__init__.py` re-exports it, the node won't load in Noetic. Those files must stay
  torch/scipy/ROS-free (or import lazily).
- **Confirm the free cloud.** Verify FALCON actually publishes
  `/voxel_mapping/occupancy_grid_free` in your fork. The projector tolerates an
  empty free cloud, but door-protection and free evidence go inert without it.