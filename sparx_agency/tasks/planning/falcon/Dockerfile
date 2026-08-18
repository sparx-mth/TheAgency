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
# python3-requests/python3-pil (apt, not pip — WITH_SIM=0/Jetson never
# installs pip, see below): core/planning/vlas/navdp/client.py lazily imports
# requests + Pillow inside reset()/pointgoal_step() so importing the
# module itself stays numpy-only for the Python-3.8 adapter (see that
# file's docstring) -- but nothing previously installed either package
# into the container at all, so the first real call crashed with
# ModuleNotFoundError. Unconditional (not just WITH_SIM=1) since
# combination_planner_node.py's NavDP path isn't simulator-only.
RUN apt-get update && apt-get install -y --no-install-recommends \
        wget git gnupg2 lsb-release ca-certificates \
        pkg-config \
        python3-tk python3-matplotlib \
        python3-requests python3-pil \
        locales \
    && if [ "$WITH_SIM" = "1" ]; then \
         apt-get install -y --no-install-recommends \
             python3-pip python3-dev sudo docker.io ; \
       fi \
    && locale-gen en_US.UTF-8 && rm -rf /var/lib/apt/lists/*

# --include-eol-distros is required, not optional: ROS Noetic went end-of-life,
# so a plain `rosdep update` now prints "Skip end-of-life distro noetic" and
# exits 1, which fails the build at this line. Found 2026-08-18 attempting the
# first from-scratch rebuild in months.
RUN rosdep update --include-eol-distros

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
# ============== ROBOTICAN FIX ==============
# libc++-dev/libc++abi-dev added below for the BUILD_GUI=ON change further
# down (also ROBOTICAN FIX): Open3D's BUILD_GUI=ON path links the
# precompiled Filament library, which requires an LLVM libc++/libc++abi
# (>=7) even though the rest of the build uses GCC/libstdc++ --
# 3rdparty/find_dependencies.cmake fails configure without them
# ("Cannot find matching libc++ and libc++abi libraries").
# Scope: this whole block is WITH_SIM=1 only, so Jetson (WITH_SIM=0,
# falcon-jetson) never runs it -- XTEND/Jetson is unaffected either way.
# ============================================
RUN if [ "$WITH_SIM" = "1" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
            libpng-dev libjpeg-dev libssl-dev xorg-dev \
            libc++-dev libc++abi-dev \
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
# ============== ROBOTICAN FIX ==============
# BUILD_GUI=ON (was OFF) is required even though rendering is headless:
# FALCON's mesh_render node uses Open3D's normal GUI-enabled classes
# (visualization::gui::Application, visualization::rendering::Open3DScene/
# FilamentRenderer) and switches them to offscreen mode at RUNTIME via
# EngineInstance::EnableHeadless() -- this is a different mechanism from
# Open3D's own ENABLE_HEADLESS_RENDERING cmake option, which actually
# forces BUILD_GUI OFF and builds an incompatible software-rendering path.
# With BUILD_GUI=OFF, mesh_render_node fails to LINK (undefined references
# to Application::GetInstance/Initialize/RenderToImage etc.) because those
# symbols live in visualization/gui, which is only compiled when
# BUILD_GUI=ON. On x86_64 this fetches a precompiled Filament binary
# rather than building it from source (BUILD_FILAMENT_FROM_SOURCE
# defaults OFF off ARM), so the extra cost is moderate, not a full
# Filament compile.
# Scope: WITH_SIM=1 only (Jetson/falcon-jetson uses WITH_SIM=0 and skips
# this whole Open3D step -- "WITH_SIM=0  Skipping Open3D" below -- so
# XTEND/Jetson is completely unaffected by this change).
# ============================================
ENV OPEN3D_INSTALL_PREFIX=/opt/open3d
ENV OPEN3D_SOURCE=/opt/Open3D
RUN if [ "$WITH_SIM" = "1" ]; then \
        echo "==> Building Open3D 0.18.0 (CPU, no ISPC, GUI on for headless mesh_render)" && \
        git clone --depth 1 --branch v0.18.0 \
            https://github.com/isl-org/Open3D.git ${OPEN3D_SOURCE} && \
        cd ${OPEN3D_SOURCE} && mkdir build && cd build && \
        cmake -DCMAKE_BUILD_TYPE=Release \
              -DCMAKE_INSTALL_PREFIX=${OPEN3D_INSTALL_PREFIX} \
              -DBUILD_CUDA_MODULE=OFF \
              -DBUILD_ISPC_MODULE=OFF \
              -DBUILD_GUI=ON \
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

# ── 6-sparx. Apply SPARX control/crash patches ported from falcon_sjtu ─
# Four upstream fixes proven live on the SJTU deployment, ported here so
# Rooster/Sphera gets the same crash-safety and tracking-quality floor.
# Applied as clean git diffs against the fresh clone above, BEFORE any of
# the sed-based fix_falcon_*.sh scripts below touch the same files.
#   - falcon_hgrid_clamp.patch: clamps UniformGrid grid-cell lookups into
#     the box (crashes on every takeoff below the box floor) AND floors
#     the A*/BFS cost-matrix CHECK_GT(cost>1e-4) -- this SUPERSEDES
#     fix_falcon_cost_check.sh, which is therefore removed below.
#   - falcon_slow_traj_rescale.patch: fixes the FSM's <0.5m/s trajectory
#     rescale, which both DIVIDES BY ZERO (inf/NaN ratio) on a
#     non-turning leg and compresses every plan 4-13x for an aircraft
#     configured to cruise under 0.5 m/s (Rooster's own
#     explore_fixed_vx=0.2/vy=0.1 are both under that trigger) -- the
#     non-finite ratio aborts exploration_node from inside a
#     visualization call and wipes the (non-decaying) voxel map.
#   - falcon_replan_from_pose.patch: falls back to the measured pose once
#     a replan's predicted start drifts >1.5m from odometry, instead of
#     compounding from a "ghost" start point on the previous trajectory
#     -- directly a control-tracking fix (this is exactly the class of
#     lag the follower/velocity_servo layer cannot correct on its own).
#   - falcon_raycast_out_of_map.patch: skips a TSDF voxel address one
#     step past the map edge instead of writing/CHECK-aborting on it.
# See sparx_agency/tasks/planning/falcon_sjtu/patches/README.md for the
# full derivation/measurements behind each fix.
COPY patches/ /tmp/sparx_patches/
# Order matters: falcon_deadend_guard introduces FrontierFinder::addBlockedRegion
# and blocked_regions_, which blocked_region_ttl/widen then extend. The four
# original ports stay first so their context is unchanged.
RUN cd /catkin_ws/src/FALCON && \
    for p in falcon_hgrid_clamp falcon_slow_traj_rescale falcon_replan_from_pose \
             falcon_raycast_out_of_map falcon_deadend_guard falcon_blocked_region_ttl \
             falcon_blocked_region_widen falcon_deadend_looping \
             falcon_ccl_slice_height falcon_publish_fail_blacklist; do \
        echo "[sparx-patch] Applying ${p}.patch" && \
        { git apply "/tmp/sparx_patches/${p}.patch" \
          || git apply --3way "/tmp/sparx_patches/${p}.patch" \
          || { echo "[sparx-patch] ERROR: ${p}.patch failed to apply" >&2; exit 1; }; }; \
    done && \
    grep -q "positionToGridCellId" falcon_planner/exploration_preprocessing/src/hierarchical_grid.cpp && \
    grep -q "slow_traj_target_vel" falcon_planner/exploration_manager/src/exploration_fsm.cpp && \
    grep -q "replan_from_pose_drift" falcon_planner/exploration_manager/src/exploration_fsm.cpp && \
    grep -q "isInMap(voxel_idx)" falcon_planner/voxel_mapping/src/tsdf.cpp && \
    grep -q "addBlockedRegion" falcon_planner/exploration_preprocessing/src/frontier_finder.cpp && \
    grep -q "never enter LKH" falcon_planner/exploration_manager/src/exploration_manager.cpp && \
    echo "[sparx-patch] OK: all 10 patches applied and verified"

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

# ── 6a-ter. hierarchical_grid.cpp cost CHECK: superseded ──────
# UniformGrid::calculateCostMatrixSingleThread() guards every cost-matrix
# edge with a glog FATAL CHECK_GT(cost, 1e-4); a grid-cell-center A* cost
# can legitimately be tiny-but-nonzero and abort the node at RUNTIME.
# Formerly fixed here by fix_falcon_cost_check.sh (floors the cost). Now
# subsumed by falcon_hgrid_clamp.patch (applied in step 6-sparx above),
# which carries the same cost floors PLUS the grid-cell-index box clamp
# fix_falcon_cost_check.sh never addressed. Do not re-add
# fix_falcon_cost_check.sh here -- its sed edits target the exact lines
# falcon_hgrid_clamp.patch's context expects unmodified, so running both
# leaves the tree unpatchable.

# Occupancy-grid publish cadence vs map growth: free/unknown clouds grow with
# explored volume and scale publish+parse cost with mapping progress.
COPY patches/fix_falcon_visgrid_cadence.sh /tmp/fix_falcon_visgrid_cadence.sh
RUN chmod +x /tmp/fix_falcon_visgrid_cadence.sh && /tmp/fix_falcon_visgrid_cadence.sh

# ── 6a-quater. Patch FALCON depthToPointcloud buffer overflow ─
COPY patches/fix_falcon_depth_overflow.sh /tmp/fix_falcon_depth_overflow.sh
RUN chmod +x /tmp/fix_falcon_depth_overflow.sh && /tmp/fix_falcon_depth_overflow.sh

# ── 6a-quater-bis. Make the map resolution settable, and visible ─
# Three bugs in the same twenty lines of map_server.cpp: the fine
# resolution is read from a misspelled key and so cannot be set at
# all; the resolution is chosen from the exploration box's VOLUME,
# so shrinking the box can multiply memory by eight; and the grid's
# size is only logged behind `verbose`, which ships false. The voxel
# grid is allocated in full on the first tick, so this is the number
# that decides whether a large area starts at all.
COPY patches/fix_falcon_map_resolution.sh /tmp/fix_falcon_map_resolution.sh
RUN chmod +x /tmp/fix_falcon_map_resolution.sh && /tmp/fix_falcon_map_resolution.sh

# ── 6a-quinquies. Silence per-callback FSM glog spam ──────────
# exploration_fsm.cpp logs two LOG(INFO) lines on EVERY callback
# (~90-130 Hz). With GLOG_logtostderr=1 (set above) they bury every
# other log on the console. Demote just those two to VLOG(99) so they
# vanish by default; all other FALCON logs are untouched. Restore with
# GLOG_v=99. See patches/fix_falcon_fsm_logspam.sh.
COPY patches/fix_falcon_fsm_logspam.sh /tmp/fix_falcon_fsm_logspam.sh
RUN chmod +x /tmp/fix_falcon_fsm_logspam.sh && /tmp/fix_falcon_fsm_logspam.sh

# ── 6a-septies. Stop FALCON finishing while the world is unexamined ─
# ============== ROBOTICAN FIX ==============
# Measured live 2026-08-18 (sphera_jail): 372s into an otherwise healthy run --
# tracking error 0.02-0.10m, zero collisions, voxel count still climbing -- the
# frontier set collapsed to ONE cluster with TWELVE retired behind it and the
# FSM quit with "Finish exploration: No frontier detected".
#
# Three causes, all in frontier_finder.cpp: UNKNOWN treated as an occluder (so a
# frontier is invisible for being a frontier), an absolute visibility bar that
# bites hardest on the small leftover clusters, and grantFinishAmnesty() being
# checked only BEFORE categorisation, which misses the case where the last
# cluster goes dormant during it.
#
# This is an anchor-based script rather than a .patch on purpose: falcon_sjtu's
# falcon_vp_audit / falcon_visib_unknown_tolerance / falcon_open_visib_bar all
# collide with our own frontier_finder.cpp ports in step 6-sparx and cannot be
# applied in any order. See patches/README.md.
# ============================================
COPY patches/fix_falcon_frontier_visibility.sh /tmp/fix_falcon_frontier_visibility.sh
RUN chmod +x /tmp/fix_falcon_frontier_visibility.sh && /tmp/fix_falcon_frontier_visibility.sh

# ── 6a-octies. Make "exploration finished" hard to reach, because it is terminal ─
# ============== ROBOTICAN FIX ==============
# The FSM finishes on the FIRST empty frontier verdict and traj_server shuts down
# on the resulting replan, so the decision cannot be walked back. Measured
# 2026-08-18 with 6a-septies already in place: a run that had flown 90.2m in
# 137.8s quit that way, and 43s later the frontier finder was offering four
# recoverable clusters to a state machine that had already stopped. Require the
# verdict to hold, in both consecutive checks and wall-clock seconds.
# ============================================
COPY patches/fix_falcon_finish_grace.sh /tmp/fix_falcon_finish_grace.sh
RUN chmod +x /tmp/fix_falcon_finish_grace.sh && /tmp/fix_falcon_finish_grace.sh

# ── 6a-decies. And let a finished exploration START AGAIN ────────────────
# ============== ROBOTICAN FIX ==============
# The grace window above makes FINISH hard to REACH; it is still impossible to
# LEAVE, and entering it publishes replan type 2, which ENDS the traj_server
# process. Measured live 2026-08-18: a 610 s run tracked for its first 106 s,
# finished once, and then held station for the remaining 560 s while its own
# frontier finder -- which keeps running in FINISH -- went on finding work.
# Re-opens FINISH when frontiers reappear, and stops traj_server exiting.
# ============================================
COPY patches/fix_falcon_finish_reopen.sh /tmp/fix_falcon_finish_reopen.sh
RUN chmod +x /tmp/fix_falcon_finish_reopen.sh && /tmp/fix_falcon_finish_reopen.sh

# ── 6a-nonies. Never fly (or hand downstream) a zero-duration trajectory ─
# ============== ROBOTICAN FIX ==============
# planTrajToView() logs "Time lower bound not satified" and then uses the
# trajectory anyway. When the optimiser returns duration 0.00, everything
# downstream divides by it (dt_yaw = duration_/seg_num) and the yaw stage spins.
# Measured 2026-08-18: that log line was the LAST output of the mission --
# exploration_node then pinned a core at 94% and grew to 35 GB RSS at ~90 MB/s
# with no further logging, and had to be killed to protect the host. FAIL is
# already handled by the FSM, which simply replans.
# ============================================
COPY patches/fix_falcon_degenerate_traj.sh /tmp/fix_falcon_degenerate_traj.sh
RUN chmod +x /tmp/fix_falcon_degenerate_traj.sh && /tmp/fix_falcon_degenerate_traj.sh

# ── 6a-sexies. Patch so3_disturbance_generator missing gencfg ─
# ============== ROBOTICAN FIX ==============
# CMake configure fails (WITH_SIM=1 only): CMakeLists.txt add_dependencies()
# on ${PROJECT_NAME}_gencfg but never calls
# generate_dynamic_reconfigure_options() to create it -- an upstream
# rosbuild->catkin port gap. See patches/fix_falcon_so3_gencfg.sh.
# Scope: so3_disturbance_generator is CATKIN_IGNORE'd on Jetson
# (WITH_SIM=0, see ignore_cuda_pkgs.sh below), so patching its
# CMakeLists.txt here is a no-op for the Jetson/XTEND build either way --
# the file is patched but never compiled there.
# ============================================
COPY patches/fix_falcon_so3_gencfg.sh /tmp/fix_falcon_so3_gencfg.sh
RUN chmod +x /tmp/fix_falcon_so3_gencfg.sh && /tmp/fix_falcon_so3_gencfg.sh

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