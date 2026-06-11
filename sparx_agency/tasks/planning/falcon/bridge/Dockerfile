# ros_bridge_docker/Dockerfile
# ============================================================
# ROS1 Noetic ↔ ROS2 Foxy bridge — built for Jetson AGX (arm64)
# also works on x86_64. Ubuntu 20.04 is the only OS where
# Noetic + Foxy coexist via apt.
#
# Lessons baked in (from the FALCON Docker port):
#   - ubuntu:20.04 base IS multiarch (amd64/arm64/...) ✓
#   - Both ROS apt repos serve arm64 builds for focal ✓
#   - apt-key is deprecated → use signed-by= on both repos
#   - colcon + vcstool via apt (python3-colcon-common-extensions)
#       is more reliable on arm64 than pip
#   - ros1_bridge needs 4 GB free RAM per ROS docs; cap parallelism
#       with MAKEFLAGS to avoid OOM on Jetson
#   - --no-install-recommends throughout
#   - Entrypoint COPY'd LAST so iterating on it doesn't invalidate
#       the 20–40 min bridge build layer
#   - bridge.yaml is NOT loaded by `dynamic_bridge --bridge-all-topics`
#       (only `parameter_bridge` reads it). It's left in your folder
#       as a future-use spec; remove if you don't plan to switch modes.
#
# Build:
#   docker build -t ros1_bridge:noetic-foxy .
#   # On Jetson Xavier or anything memory-constrained, override:
#   #   docker build -t ros1_bridge:noetic-foxy --build-arg MAX_JOBS=1 .
#
# Build time: ~20–40 min on AGX Orin, longer on Xavier.
# ============================================================
FROM ubuntu:20.04

ARG MAX_JOBS=2
# MAX_JOBS caps the inner make -j during the bridge build.
#   AGX Orin (12c/32GB): 2-4 safe. AGX Xavier (8c/32GB): 2.
#   Nano / 8GB Jetson: 1.
# ros1_bridge generates hundreds of message-conversion translation
# units; peak per-process memory is ~1.5-2 GB. -j2 ≈ ~4 GB peak.

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV ROS1_DISTRO=noetic
ENV ROS2_DISTRO=foxy

# ── Base system tools ─────────────────────────────────────────
# curl, gnupg2: needed once to fetch the ROS apt keyring
# ca-certificates: TLS for the curl above
# build-essential + cmake + git: required to build the bridge
# locales: en_US.UTF-8 for the build
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg2 lsb-release \
        build-essential cmake git \
        locales \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# ── ROS1 + ROS2 apt repos (modern signed-by, no apt-key) ──────
# Both repos share the same signing key (rosdistro/ros.asc).
# arch=$(dpkg --print-architecture) auto-resolves to arm64 on Jetson,
# amd64 on x86_64. No platform-specific changes needed.
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
      | gpg --dearmor -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu focal main" \
        > /etc/apt/sources.list.d/ros1.list \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu focal main" \
        > /etc/apt/sources.list.d/ros2.list

# ── ROS1 Noetic + ROS2 Foxy + bridge build deps ───────────────
# Message packages installed in MATCHING pairs on both sides.
# ros1_bridge auto-discovers any pair available at build time and
# generates conversion code for it. Adding a message type later
# (e.g. ros-noetic-quadrotor-msgs ↔ ros-foxy-quadrotor-msgs)
# requires re-running the colcon build step.
#
# Why each:
#   ros-{noetic,foxy}-ros-base
#       core ROS runtime
#   ros-{noetic,foxy}-tf2-ros
#       tf/tf2 bridging (FALCON uses tf heavily)
#   ros-{noetic,foxy}-geometry-msgs
#       Pose, Twist, PoseStamped, Point, etc.
#   ros-{noetic,foxy}-sensor-msgs
#       Image, CameraInfo, PointCloud2
#   ros-{noetic,foxy}-nav-msgs
#       Path, OccupancyGrid, Odometry (FALCON BEV + waypoints)
#   ros-{noetic,foxy}-std-msgs
#       Bool, Empty, Int8 (used by sensor_gate /freeze etc.)
#   ros-{noetic,foxy}-visualization-msgs
#       Marker, MarkerArray (FALCON publishes many of these from
#       odom_visualization and exploration_node frontier viz)
#
# ROS2-side RMW: both fastrtps and cyclonedds installed so you can
# pick at runtime via RMW_IMPLEMENTATION env var. Must match the
# RMW used by the host's ROS2 sim:
#   - Foxy default         = rmw_fastrtps_cpp
#   - Humble/Jazzy default = rmw_cyclonedds_cpp
# run_bridge.sh currently sets rmw_fastrtps_cpp; flip it if
# your host is Humble/Jazzy with default cyclonedds.
#
# python3-colcon-common-extensions: builds the bridge.
# python3-vcstool: not strictly needed but handy if you extend the ws.
# python3-rosdep: standard ROS dep resolver.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-noetic-ros-base \
        ros-noetic-tf2-ros \
        ros-noetic-geometry-msgs \
        ros-noetic-sensor-msgs \
        ros-noetic-nav-msgs \
        ros-noetic-std-msgs \
        ros-noetic-visualization-msgs \
        ros-foxy-ros-base \
        ros-foxy-rmw-fastrtps-cpp \
        ros-foxy-rmw-cyclonedds-cpp \
        ros-foxy-tf2-ros \
        ros-foxy-geometry-msgs \
        ros-foxy-sensor-msgs \
        ros-foxy-nav-msgs \
        ros-foxy-std-msgs \
        ros-foxy-visualization-msgs \
        python3-rosdep \
        python3-colcon-common-extensions \
        python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

# ── Clone ros1_bridge (foxy branch) ───────────────────────────
RUN mkdir -p /bridge_ws/src \
    && git clone --depth 1 -b foxy \
        https://github.com/ros2/ros1_bridge.git /bridge_ws/src/ros1_bridge

# ── Build the bridge (memory-capped) ──────────────────────────
# MAKEFLAGS=-j${MAX_JOBS} is the official mitigation from the ROS
# docs ("ros1_bridge requires 4Gb of free RAM to compile"). It
# limits the underlying make's parallelism. Belt + suspenders with
# CMAKE_BUILD_PARALLEL_LEVEL for cmake's --build step.
#
# --packages-select  build only ros1_bridge (the only thing in the ws)
# --symlink-install  reuse source paths for resources
# --cmake-force-configure  standard recommendation for cross-ROS builds
# --event-handlers console_direct+  verbose, lets you see progress
#
# The final `test -f ...dynamic_bridge` is a fail-loud check: if
# the bridge didn't actually produce a binary, the Docker build
# fails here instead of producing a silently-broken image.
RUN /bin/bash -c "\
    set -e && \
    source /opt/ros/noetic/setup.bash && \
    source /opt/ros/foxy/setup.bash && \
    cd /bridge_ws && \
    MAKEFLAGS='-j${MAX_JOBS}' \
    CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS} \
    colcon build --symlink-install \
        --packages-select ros1_bridge \
        --cmake-force-configure \
        --event-handlers console_direct+ \
    " \
    && test -f /bridge_ws/install/ros1_bridge/lib/ros1_bridge/dynamic_bridge \
    || (echo '[ERROR] dynamic_bridge binary not found — build failed silently' \
        && ls -la /bridge_ws/install/ros1_bridge/lib/ros1_bridge/ 2>/dev/null \
        && exit 1)

# ── Entrypoint LAST for cache efficiency ──────────────────────
# Changes to entrypoint.sh only invalidate the last 2 layers, not
# the ~30-min bridge build above.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]