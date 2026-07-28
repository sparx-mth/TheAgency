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
