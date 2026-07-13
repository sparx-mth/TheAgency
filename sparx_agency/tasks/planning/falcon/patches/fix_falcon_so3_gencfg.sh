#!/bin/bash
# ============== ROBOTICAN FIX ==============
# New file. Patches uav_simulator/so3_disturbance_generator, which is
# CATKIN_IGNORE'd on Jetson (WITH_SIM=0, see ignore_cuda_pkgs.sh) --
# this patch is a no-op for the Jetson/XTEND build (file patched, never
# compiled there). Needed for x86 (WITH_SIM=1) builds, where this
# package DOES compile and previously failed CMake configure.
# ============================================
# ============================================================
# fix_falcon_so3_gencfg.sh
#
# Fixes a CMake CONFIGURE-time failure in FALCON's ros1-noetic branch:
#
#   CMake Error at FALCON/uav_simulator/so3_disturbance_generator/
#   CMakeLists.txt:70 (add_dependencies):
#     The dependency target "so3_disturbance_generator_gencfg" of target
#     "so3_disturbance_generator" does not exist.
#
# Root cause:
#   so3_disturbance_generator/CMakeLists.txt depends on dynamic_reconfigure
#   and add_dependencies()'s on the auto-generated ${PROJECT_NAME}_gencfg
#   target, but never calls generate_dynamic_reconfigure_options() to
#   actually create that target -- the upstream rosbuild->catkin port
#   (cfg/disturbance_ui.cfg already uses parameter_generator_catkin)
#   dropped the macro call. This is a WITH_SIM=1-only package (part of
#   uav_simulator), so it isn't skipped by ignore_cuda_pkgs.sh.
#
# Fix:
#   Insert generate_dynamic_reconfigure_options(cfg/disturbance_ui.cfg)
#   before catkin_package(), matching the standard dynamic_reconfigure
#   ordering (http://wiki.ros.org/dynamic_reconfigure/Tutorials).
#
# Idempotent + self-verifying. Run AFTER cloning FALCON, BEFORE
# catkin_make.
# ============================================================
set -euo pipefail

CM="/catkin_ws/src/FALCON/uav_simulator/so3_disturbance_generator/CMakeLists.txt"
if [ ! -f "${CM}" ]; then
  echo "[fix_so3_gencfg] ERROR: ${CM} not found" >&2
  exit 1
fi
echo "[fix_so3_gencfg] Patching ${CM}"

if grep -q "generate_dynamic_reconfigure_options" "${CM}"; then
  echo "[fix_so3_gencfg] Already patched, skipping."
  exit 0
fi
cp "${CM}" "${CM}.orig.bak"

sed -i \
  's|^catkin_package()|generate_dynamic_reconfigure_options(cfg/disturbance_ui.cfg)\ncatkin_package()|' \
  "${CM}"

if ! grep -q "generate_dynamic_reconfigure_options(cfg/disturbance_ui.cfg)" "${CM}"; then
  echo "[fix_so3_gencfg] ERROR: patch did not apply; reverting." >&2
  mv "${CM}.orig.bak" "${CM}"
  exit 1
fi
echo "[fix_so3_gencfg] OK: so3_disturbance_generator_gencfg target will now be generated."
