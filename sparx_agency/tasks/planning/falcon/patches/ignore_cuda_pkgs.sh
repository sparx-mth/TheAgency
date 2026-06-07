#!/bin/bash
# ============================================================
# ignore_cuda_pkgs.sh
#
# Seeds CATKIN_IGNORE on packages we don't want built, then
# cascades CATKIN_IGNORE to any package whose package.xml lists
# a dep on something already ignored.
#
# SEED LIST is environment-dependent:
#
#   WITH_SIM=1 (default — preserves the original behaviour):
#     Only the two hard-CUDA packages are seeded:
#       pointcloud_render
#       map_render
#
#   WITH_SIM=0 (Jetson / real-drone-only):
#     Above plus everything simulator-side, rviz_plugins, and
#     multi_map_server:
#       mesh_render               (needs Open3D — skipped)
#       so3_quadrotor_simulator   (Gazebo replacement, unused)
#       so3_control               (controller for above)
#       so3_disturbance_generator
#       poscmd_2_odom
#       waypoint_generator        (sim helper)
#       rviz_plugins              (RViz visual plugin, unused;
#                                  ros:noetic-perception has no
#                                  rviz dev headers anyway)
#       multi_map_server          (only consumed by rviz_plugins;
#                                  has a CMake-3.27+ incompatible
#                                  add_dependencies bug upstream.
#                                  Nothing else in the active set
#                                  uses it.)
#
#     uav_simulator/utils/{quadrotor_msgs, odom_visualization,
#     uav_utils, pose_utils, cmake_utils} stay active — they are
#     build-time deps of the planner or runtime nodes referenced
#     in our launches.
#
# Usage:
#   ./ignore_cuda_pkgs.sh /catkin_ws/src
#   WITH_SIM=0 ./ignore_cuda_pkgs.sh /catkin_ws/src   # Jetson
# ============================================================
set -e

WS_SRC="${1:-/catkin_ws/src}"
WITH_SIM="${WITH_SIM:-1}"

# Always-ignored: the hard-CUDA simulator renderers.
SEED_DIRS=(
    "${WS_SRC}/FALCON/uav_simulator/camera_sensing/pointcloud_render"
    "${WS_SRC}/FALCON/uav_simulator/map_render"
)

# WITH_SIM=0 extends the seed with everything we don't need
# off the real-drone code path.
if [ "${WITH_SIM}" = "0" ]; then
    echo "==> WITH_SIM=0  Extending seed list with sim packages + rviz_plugins + multi_map_server."
    SEED_DIRS+=(
        "${WS_SRC}/FALCON/uav_simulator/camera_sensing/mesh_render"
        "${WS_SRC}/FALCON/uav_simulator/so3_quadrotor_simulator"
        "${WS_SRC}/FALCON/uav_simulator/so3_control"
        "${WS_SRC}/FALCON/uav_simulator/so3_disturbance_generator"
        "${WS_SRC}/FALCON/uav_simulator/poscmd_2_odom"
        "${WS_SRC}/FALCON/uav_simulator/utils/waypoint_generator"
        "${WS_SRC}/FALCON/uav_simulator/utils/rviz_plugins"
        "${WS_SRC}/FALCON/uav_simulator/utils/multi_map_server"
    )
else
    echo "==> WITH_SIM=1  Only CUDA packages will be seeded."
fi

for d in "${SEED_DIRS[@]}"; do
    if [ -d "$d" ]; then
        touch "$d/CATKIN_IGNORE"
        echo "==> Seed ignored: $(basename "$d")"
    fi
done

# Cascade: keep ignoring packages whose package.xml lists a
# dependency on an already-ignored package, until quiescence.
ITER=0
while : ; do
    ITER=$((ITER + 1))
    CHANGED=0

    # Build current set of ignored package names (from package.xml).
    IGNORED_NAMES=""
    while IFS= read -r ig_marker; do
        ig_dir=$(dirname "$ig_marker")
        if [ -f "$ig_dir/package.xml" ]; then
            n=$(grep -oP '<name>\K[^<]+' "$ig_dir/package.xml" | head -1)
            IGNORED_NAMES="$IGNORED_NAMES $n"
        fi
    done < <(find "$WS_SRC" -name CATKIN_IGNORE)

    # Walk every active (not-yet-ignored) package and check its deps.
    while IFS= read -r pkg_xml; do
        pkg_dir=$(dirname "$pkg_xml")
        [ -f "$pkg_dir/CATKIN_IGNORE" ] && continue

        for ig_name in $IGNORED_NAMES; do
            if grep -qE "<(build_depend|depend|exec_depend|run_depend)>${ig_name}</" "$pkg_xml"; then
                pkg_name=$(grep -oP '<name>\K[^<]+' "$pkg_xml" | head -1)
                echo "==> Iter ${ITER}: ignoring '${pkg_name}' (depends on '${ig_name}')"
                touch "$pkg_dir/CATKIN_IGNORE"
                CHANGED=1
                break
            fi
        done
    done < <(find "$WS_SRC" -name package.xml)

    [ "$CHANGED" = "0" ] && break
done

echo ""
echo "===== Final ignored packages ====="
find "$WS_SRC" -name CATKIN_IGNORE | while read -r m; do
    d=$(dirname "$m")
    if [ -f "$d/package.xml" ]; then
        grep -oP '<name>\K[^<]+' "$d/package.xml" | head -1
    fi
done | sort
echo "=================================="