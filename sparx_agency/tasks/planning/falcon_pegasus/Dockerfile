# ============================================================
# falcon-pegasus:noetic — the FALCON image plus this task's catkin package.
#
# Layered on top of the existing FALCON image rather than rebuilding it: that
# one already contains the compiled planner, the mapper, NLopt, Open3D and the
# quadrotor_msgs/trajectory message packages, and it takes the better part of an
# hour to build. All this adds is one rospy package, so the incremental
# catkin_make is short.
#
# Build:
#   cd sparx_agency/tasks/planning/falcon_pegasus
#   docker build -t falcon-pegasus:noetic .
#
# Or point it at a different FALCON image:
#   docker build --build-arg BASE_IMAGE=falcon-ros:noetic -t falcon-pegasus:noetic .
# ============================================================
ARG BASE_IMAGE=falcon-ros-custom:v1
FROM ${BASE_IMAGE}

ARG MAX_JOBS=4
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5

# Make FALCON's planning-failure log name the positions it could not connect.
# Upstream prints "No path to next viewpoint" and nothing else, which is not
# enough to tell a viewpoint outside the known space from an aircraft that is
# somewhere unexpected. Behaviour is unchanged; see the script.
COPY patches/log_plan_failure.sh /tmp/log_plan_failure.sh
RUN chmod +x /tmp/log_plan_failure.sh && /tmp/log_plan_failure.sh

# Stop exploration_node segfaulting on a cell with no reachable viewpoint. This
# one is a real crash, not a log improvement: it killed a healthy 85-second
# exploration mid-flight, silently. See the script.
COPY patches/fix_falcon_viewpoint_index.sh /tmp/fix_falcon_viewpoint_index.sh
RUN chmod +x /tmp/fix_falcon_viewpoint_index.sh && /tmp/fix_falcon_viewpoint_index.sh

# Stop exploration_node segfaulting when the aircraft leaves the exploration
# box -- which is routine here, because the box is inset from the building and
# its floor sits above the ground. Unchecked index into a std::vector; same
# shape as the viewpoint fix above, and the reason two real flights ended early.
COPY patches/fix_falcon_grid_cell_index.sh /tmp/fix_falcon_grid_cell_index.sh
RUN chmod +x /tmp/fix_falcon_grid_cell_index.sh && /tmp/fix_falcon_grid_cell_index.sh

# Occupancy-grid publish cadence vs map growth (measured on SJTU 2026-08-12):
# free/unknown clouds grow with explored volume; occupied stays 2 Hz.
COPY patches/fix_falcon_visgrid_cadence.sh /tmp/fix_falcon_visgrid_cadence.sh
RUN chmod +x /tmp/fix_falcon_visgrid_cadence.sh && /tmp/fix_falcon_visgrid_cadence.sh

# Tell A* how big the drone is. Upstream routes a 0.7 m aircraft through a 0.3 m
# gap because it tests one voxel per node, and the B-spline optimiser then
# cannot thread the route it was handed. Inert unless the launch file sets
# /astar/inflate_radius, so this image still reproduces a baseline run.
# Stop the mapper aborting on a depth ray that steps one voxel off the edge of
# the map. glog's CHECK is abort(), so this killed a flight outright.
COPY patches/fix_falcon_raycast_out_of_map.sh /tmp/fix_falcon_raycast_out_of_map.sh
RUN chmod +x /tmp/fix_falcon_raycast_out_of_map.sh && /tmp/fix_falcon_raycast_out_of_map.sh

# Run the vendored LKH TSP solver in a forked child, with a greedy fallback.
# It is one-shot C being called thousands of times in one process, its global
# state does not survive that, and its segfault was the single biggest reason
# no flight finished -- invisible from the aircraft, because traj_server
# outlives the planner and keeps republishing the last trajectory.
COPY patches/isolate_lkh_tsp_solver.sh /tmp/isolate_lkh_tsp_solver.sh
RUN chmod +x /tmp/isolate_lkh_tsp_solver.sh && /tmp/isolate_lkh_tsp_solver.sh

COPY patches/inflate_astar_by_airframe.sh /tmp/inflate_astar_by_airframe.sh
RUN chmod +x /tmp/inflate_astar_by_airframe.sh && /tmp/inflate_astar_by_airframe.sh

# The catkin package: two rospy nodes, a launch file, and the run configs.
# runs/ lands INSIDE the package so `$(find falcon_pegasus)/runs/<run>.yaml`
# resolves in the launch file. run_falcon_pegasus.sh bind-mounts the host copies
# over both, so editing a run config or a node needs no rebuild.
COPY adapter/ /catkin_ws/src/falcon_pegasus/
COPY runs/ /catkin_ws/src/falcon_pegasus/runs/

# Incremental: every other package in the workspace is already built and
# unchanged, so this only configures and installs the new one.
WORKDIR /catkin_ws
RUN /bin/bash -c "\
    source /opt/ros/noetic/setup.bash && \
    catkin_make -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -j${MAX_JOBS} \
    "

# The nodes import sparx_agency.tasks.planning.falcon_pegasus.* (the wire
# protocol and the frame renderer) from the repo, which is mounted read-only at
# /opt/sparx_agency by the run script.
ENV PYTHONPATH=/opt
