# ============================================================
# FALCON Docker Image  (ROS1 Noetic + Adapter)
#
# Base image is selected via BASE_IMAGE build-arg:
#   x86_64 (WITH_SIM=1, default):  osrf/ros:noetic-desktop-full
#   Jetson AGX (WITH_SIM=0):       ros:noetic-perception  (multiarch arm64)
#
# WITH_SIM=1: builds Open3D + mesh_render + sim packages.
# WITH_SIM=0: skips Open3D; CATKIN_IGNORE simulator packages
#             AND rviz_plugins AND multi_map_server.
# ============================================================
ARG BASE_IMAGE=osrf/ros:noetic-desktop-full
FROM ${BASE_IMAGE}

ARG WITH_SIM=1
ARG MAX_JOBS=4

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC  LANG=C.UTF-8  LC_ALL=C.UTF-8
ENV ROS_DISTRO=noetic
# Modern CMake (3.27+) errors on FALCON's cmake_minimum_required(2.8.3).
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5

# Log visibility: FALCON uses glog (LOG(INFO)/LOG(WARNING)), which by
# default writes INFO/WARNING to /tmp log files and sends only
# ERROR/FATAL to stderr -- so routine logs are invisible on the console.
# Route everything glog produces to stderr so roslaunch shows it on
# screen. Use GLOG_alsologtostderr=1 instead if you also want glog to
# keep writing its own /tmp log files.
ENV GLOG_logtostderr=1
ENV GLOG_colorlogtostderr=1

# ── 1. Common system tools ────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget git gnupg2 lsb-release ca-certificates \
        pkg-config \
        python3-tk python3-matplotlib \
        locales \
    && if [ "$WITH_SIM" = "1" ]; then \
         apt-get install -y --no-install-recommends \
             python3-pip python3-dev sudo docker.io ; \
       fi \
    && locale-gen en_US.UTF-8 && rm -rf /var/lib/apt/lists/*

RUN rosdep update

# ── 2. FALCON-required apt deps + perception/geometry ROS pkgs ─────
# C++ libs:
#   libgoogle-glog-dev — exploration_manager, fast_planner, voxel_mapping
#   libdw-dev          — voxel_mapping/CMakeLists.txt:50 has a literal
#                         `-ldw` linker flag. -ldw resolves libdw from
#                         elfutils, used by the vendored backward-cpp
#                         stack-trace lib (falcon_planner/exploration_utils/
#                         include/backward/backward.hpp). Without it,
#                         voxel_mapping fails to link.
#   libdwarf-dev       — companion debug-info lib; was in FALCON's
#                         original install list. Small (~2 MB).
#   libarmadillo-dev   — multi_map_server, odom_visualization, pose_utils
#   libeigen3-dev      — 10 active packages
#   libboost-dev       — header-only boost (bind, functional/hash)
#
# ROS packages (always installed; idempotent if base image has them):
#   cv_bridge, pcl_conversions, pcl_ros, laser_geometry, image_transport,
#   tf2_eigen, tf2_geometry_msgs, tf_conversions, eigen_conversions
#
# RViz + viz plugins (NEW): always installed so we can run RViz inside
# the container on Jetson too (the perception base image doesn't have it;
# the desktop-full base does — apt is idempotent so this is harmless
# either way).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgoogle-glog-dev \
        libdw-dev \
        libdwarf-dev \
        libarmadillo-dev \
        libeigen3-dev \
        libboost-dev \
        ros-noetic-cv-bridge \
        ros-noetic-pcl-conversions \
        ros-noetic-pcl-ros \
        ros-noetic-laser-geometry \
        ros-noetic-image-transport \
        ros-noetic-tf2-eigen \
        ros-noetic-tf2-geometry-msgs \
        ros-noetic-tf-conversions \
        ros-noetic-eigen-conversions \
        ros-noetic-rviz \
        ros-noetic-rviz-imu-plugin \
        ros-noetic-visualization-tutorials \
    && rm -rf /var/lib/apt/lists/*

# ── 2b. Open3D-only apt deps (WITH_SIM=1 only) ────────────────
RUN if [ "$WITH_SIM" = "1" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
            libpng-dev libjpeg-dev libssl-dev xorg-dev \
        && rm -rf /var/lib/apt/lists/* ; \
    fi

# ── 3. CMake ≥ 3.26 from Kitware APT (multiarch) ──────────────
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
        | gpg --dearmor - > /usr/share/keyrings/kitware-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main" \
        > /etc/apt/sources.list.d/kitware.list \
    && apt-get update && apt-get install -y --no-install-recommends cmake \
    && rm -rf /var/lib/apt/lists/* && cmake --version

# ── 4. NLopt 2.7.1 (used by falcon_planner/trajectory) ────────
RUN cd /tmp && git clone --depth 1 --branch v2.7.1 https://github.com/stevengj/nlopt.git \
    && cd nlopt && mkdir build && cd build \
    && cmake -DCMAKE_BUILD_TYPE=Release .. \
    && make -j${MAX_JOBS} && make install \
    && ldconfig && rm -rf /tmp/nlopt

# ── 5. Open3D 0.18.0 (CPU-only) — only when WITH_SIM=1 ────────
ENV OPEN3D_INSTALL_PREFIX=/opt/open3d
ENV OPEN3D_SOURCE=/opt/Open3D
RUN if [ "$WITH_SIM" = "1" ]; then \
        echo "==> Building Open3D 0.18.0 (CPU, no ISPC, no GUI)" && \
        git clone --depth 1 --branch v0.18.0 \
            https://github.com/isl-org/Open3D.git ${OPEN3D_SOURCE} && \
        cd ${OPEN3D_SOURCE} && mkdir build && cd build && \
        cmake -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_INSTALL_PREFIX=${OPEN3D_INSTALL_PREFIX} \
              -DBUILD_CUDA_MODULE=OFF \
              -DBUILD_ISPC_MODULE=OFF \
              -DBUILD_GUI=OFF \
              -DBUILD_TENSORFLOW_OPS=OFF \
              -DBUILD_PYTORCH_OPS=OFF \
              -DBUILD_PYTHON_MODULE=OFF \
              -DBUILD_UNIT_TESTS=OFF \
              -DBUILD_BENCHMARKS=OFF .. && \
        make -j${MAX_JOBS} && make install && ldconfig ; \
    else \
        echo "==> WITH_SIM=0  Skipping Open3D" ; \
    fi

# ── 6. Clone FALCON ───────────────────────────────────────────
RUN mkdir -p /catkin_ws/src
WORKDIR /catkin_ws/src
RUN git clone --branch ros1-noetic \
        https://github.com/HKUST-Aerial-Robotics/FALCON.git

# ── 6a. Patch FALCON CMakeLists.txt for new CMake (3.27+) ─────
RUN sed -i \
      's|multi_map_server_messages_cpp|multi_map_server_generate_messages_cpp|g' \
      /catkin_ws/src/FALCON/uav_simulator/utils/multi_map_server/CMakeLists.txt

# Patch open3d_resource_path only when mesh_render will be built.
RUN if [ "$WITH_SIM" = "1" ]; then \
        find /catkin_ws/src/FALCON -name "mesh_render.yaml" \
          | xargs --no-run-if-empty sed -i \
            "s|open3d_resource_path:.*|open3d_resource_path: ${OPEN3D_SOURCE}/build/bin/resources|g" \
          || true ; \
    fi

# ── 6a-bis. Patch FALCON exploration_utils/system_info.cpp ────
# printSystemInfo() crashes the exploration_node on Jetson because
# nvidia-smi --query-gpu fails (NVML error) and std::stol receives a
# non-numeric string. Without this patch, exploration_node aborts at
# startup with std::invalid_argument and no voxels are ever produced.
COPY patches/fix_falcon_system_info.sh /tmp/fix_falcon_system_info.sh
RUN chmod +x /tmp/fix_falcon_system_info.sh && /tmp/fix_falcon_system_info.sh

# ── 6a-ter. Patch FALCON hierarchical_grid.cpp cost CHECK ─────
# UniformGrid::calculateCostMatrixSingleThread() guards every cost-matrix
# edge with a glog FATAL CHECK_GT(cost, 1e-4). When the drone sits ~on a
# grid-cell free center, the A*/connectivity cost is a tiny-but-nonzero
# value below the threshold, so the node aborts (SIGABRT / exit -6) at
# RUNTIME during planning. This floors every such cost so the check
# can't fire while still keeping a zero-weight edge out of the solver.
COPY patches/fix_falcon_cost_check.sh /tmp/fix_falcon_cost_check.sh
RUN chmod +x /tmp/fix_falcon_cost_check.sh && /tmp/fix_falcon_cost_check.sh

# ── 6a-quater. Patch FALCON depthToPointcloud buffer overflow ─
COPY patches/fix_falcon_depth_overflow.sh /tmp/fix_falcon_depth_overflow.sh
RUN chmod +x /tmp/fix_falcon_depth_overflow.sh && /tmp/fix_falcon_depth_overflow.sh

# ── 6a-quinquies. Silence per-callback FSM glog spam ──────────
# exploration_fsm.cpp logs two LOG(INFO) lines on EVERY callback
# (~90-130 Hz). With GLOG_logtostderr=1 (set above) they bury every
# other log on the console. Demote just those two to VLOG(99) so they
# vanish by default; all other FALCON logs are untouched. Restore with
# GLOG_v=99. See patches/fix_falcon_fsm_logspam.sh.
COPY patches/fix_falcon_fsm_logspam.sh /tmp/fix_falcon_fsm_logspam.sh
RUN chmod +x /tmp/fix_falcon_fsm_logspam.sh && /tmp/fix_falcon_fsm_logspam.sh

# ── 6b. Skip CUDA-only + (Jetson) simulator-side packages ─────
COPY patches/ignore_cuda_pkgs.sh /tmp/ignore_cuda_pkgs.sh
RUN chmod +x /tmp/ignore_cuda_pkgs.sh \
    && WITH_SIM=${WITH_SIM} /tmp/ignore_cuda_pkgs.sh /catkin_ws/src

# ── 7. Add adapter package + map config ───────────────────────
COPY adapter/ /catkin_ws/src/falcon_adapter/
COPY maps/hospital.yaml /catkin_ws/src/FALCON/falcon_planner/exploration_manager/config/map/hospital.yaml

# ── 8. Build everything ───────────────────────────────────────
WORKDIR /catkin_ws
RUN /bin/bash -c "\
    source /opt/ros/noetic/setup.bash && \
    catkin_make \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DOpen3D_DIR=${OPEN3D_INSTALL_PREFIX}/lib/cmake/Open3D \
        -j${MAX_JOBS} \
    "

# ── 8b. Patch FALCON SOP timeout (1s → 10s) and rebuild ───────
COPY patches/fix_falcon_sop.sh /fix_falcon_sop.sh
RUN bash /fix_falcon_sop.sh

# ── 9. Shell + entrypoint ─────────────────────────────────────
RUN echo "source /opt/ros/noetic/setup.bash"               >> /root/.bashrc \
    && echo "source /catkin_ws/devel/setup.bash"           >> /root/.bashrc \
    && echo "export ROS_MASTER_URI=http://localhost:11311" >> /root/.bashrc

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]