#!/usr/bin/env bash
# Resolve the external SJTU simulator checkout, or fail loudly.
#
# The worlds, the drone model and libplugin_drone.so live in their own 207 MB
# repository. Nothing here vendors, copies or hardcodes a path to it: one
# environment variable, $SJTU_PROJECT_DIR, points at it, and everything else in
# this package is derived from that. Two checkouts on one machine, a checkout on
# a different disk, or a CI box with no simulator at all are then all the same
# case, and the last one fails with a sentence instead of a stack trace.
#
# Source it (`source setup/env.sh`) to get the exports; run it directly to print
# what it resolved and check the checkout is complete.
#
# Exports:
#   SJTU_PROJECT_DIR      the external repo root (validated, absolute)
#   SJTU_DRONE_DIR        <root>/sjtu_drone            -- packages + Dockerfile
#   SJTU_DESCRIPTION_DIR  the sjtu_drone_description package (URDF, plugin, playground)
#   SJTU_HOSPITAL_DIR     the aws-robomaker-hospital-world checkout, if present
#   SJTU_CONTAINER_WS     where the repo is mounted inside the container
#   SJTU_IMAGE            the prebuilt docker image (base, Fast DDS only)
#   SJTU_CYCLONE_IMAGE    the same image plus CycloneDDS, built from setup/Dockerfile
#   SJTU_IMAGE_EXPLICIT   true when the caller pinned SJTU_IMAGE themselves
#   SJTU_CYCLONEDDS_URI   the CycloneDDS profile baked into SJTU_CYCLONE_IMAGE
#   ROS_DOMAIN_ID         defaulted, never overridden if already set

# `return` at the top level of a sourced script returns from it; in an executed
# script it fails, and the `|| exit` takes over. One line that behaves correctly
# both ways -- getting this wrong closes an operator's interactive shell over a
# missing variable, which is a memorable way to learn it.
_sjtu_die() {
  echo "[sjtu/env] ERROR: $*" >&2
}

if [[ -z "${SJTU_PROJECT_DIR:-}" ]]; then
  _sjtu_die "SJTU_PROJECT_DIR is not set.

  It must point at the external SJTU simulator repository -- the one holding
  sjtu_drone/ (the ROS 2 packages and the Gazebo plugin) and the world repos
  beside it. It is a separate git repository and is deliberately not part of
  TheAgency, which is why no path to it is written down anywhere in this tree.

      export SJTU_PROJECT_DIR=/path/to/sjtu_project

  Then re-run this script to check the checkout."
  return 1 2>/dev/null || exit 1
fi

if [[ ! -d "${SJTU_PROJECT_DIR}" ]]; then
  _sjtu_die "SJTU_PROJECT_DIR points at '${SJTU_PROJECT_DIR}', which is not a directory."
  return 1 2>/dev/null || exit 1
fi

# Resolve to an absolute path once, so every derived path and every `docker -v`
# is absolute no matter where the caller was standing.
SJTU_PROJECT_DIR="$(cd "${SJTU_PROJECT_DIR}" && pwd)"
export SJTU_PROJECT_DIR

export SJTU_DRONE_DIR="${SJTU_PROJECT_DIR}/sjtu_drone"
export SJTU_DESCRIPTION_DIR="${SJTU_DRONE_DIR}/sjtu_drone_description"
export SJTU_HOSPITAL_DIR="${SJTU_PROJECT_DIR}/aws-robomaker-hospital-world"

if [[ ! -d "${SJTU_DESCRIPTION_DIR}" ]]; then
  _sjtu_die "'${SJTU_PROJECT_DIR}' does not look like the SJTU simulator repo:
  expected ${SJTU_DESCRIPTION_DIR} to exist. Check that SJTU_PROJECT_DIR points at
  the repository ROOT, not at the sjtu_drone/ package inside it."
  return 1 2>/dev/null || exit 1
fi

# Where the repo is bind-mounted inside the container. Mirrors the external
# repo's own run.sh convention on purpose: colcon bakes absolute paths into
# install/, so building under a second mount point would silently invalidate a
# workspace built under the first.
export SJTU_CONTAINER_WS="/root/$(basename "${SJTU_PROJECT_DIR}")"

# Prebuilt from the external repo's Dockerfile: ROS 2 Humble + Gazebo Classic 11.
# Recorded before defaulting: a caller who pinned SJTU_IMAGE has made a choice,
# and bringup_world.sh must not quietly substitute a different image for it.
if [[ -n "${SJTU_IMAGE:-}" ]]; then
  export SJTU_IMAGE_EXPLICIT="true"
else
  export SJTU_IMAGE_EXPLICIT="false"
fi
export SJTU_IMAGE="${SJTU_IMAGE:-sjtu_drone_nadav:humble_ros2}"

# The same image with ros-humble-rmw-cyclonedds-cpp baked in -- built here, from
# setup/Dockerfile, because the ROS 1 bridge runs Foxy and Humble's default Fast
# DDS 2.6 is not wire-compatible with Foxy's 2.1. Without it the simulator and
# FALCON share a domain and exchange nothing. Absence is not fatal: bring-up
# falls back to Fast DDS, which is fine for anything that stays inside ROS 2.
export SJTU_CYCLONE_IMAGE="${SJTU_CYCLONE_IMAGE:-sjtu_drone_sparx:humble}"
export SJTU_CYCLONEDDS_URI="${SJTU_CYCLONEDDS_URI:-file:///etc/cyclonedds/no_shm.xml}"

# Every participant must agree, and a mismatch drops all traffic silently -- it
# looks exactly like a simulator that never started. 20 is what the external
# repo uses.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-20}"

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "SJTU_PROJECT_DIR     ${SJTU_PROJECT_DIR}"
  echo "SJTU_DRONE_DIR       ${SJTU_DRONE_DIR}"
  echo "SJTU_DESCRIPTION_DIR ${SJTU_DESCRIPTION_DIR}"
  if [[ -d "${SJTU_HOSPITAL_DIR}" ]]; then
    echo "SJTU_HOSPITAL_DIR    ${SJTU_HOSPITAL_DIR}"
  else
    echo "SJTU_HOSPITAL_DIR    ${SJTU_HOSPITAL_DIR}   (absent -- 'hospital' will not resolve)"
  fi
  echo "SJTU_CONTAINER_WS    ${SJTU_CONTAINER_WS}"
  echo "SJTU_IMAGE           ${SJTU_IMAGE}$( [[ "${SJTU_IMAGE_EXPLICIT}" == "true" ]] && echo '   (pinned by the caller)' )"
  if docker image inspect "${SJTU_CYCLONE_IMAGE}" >/dev/null 2>&1; then
    echo "SJTU_CYCLONE_IMAGE   ${SJTU_CYCLONE_IMAGE}   (present -- bring-up defaults to CycloneDDS)"
  else
    echo "SJTU_CYCLONE_IMAGE   ${SJTU_CYCLONE_IMAGE}   (absent -- bring-up falls back to Fast DDS, no ROS 1 bridge)"
  fi
  echo "ROS_DOMAIN_ID        ${ROS_DOMAIN_ID}"
  if ! docker image inspect "${SJTU_IMAGE}" >/dev/null 2>&1; then
    echo >&2
    echo "[sjtu/env] WARNING: docker image '${SJTU_IMAGE}' is not present." >&2
    echo "           Build it with: docker build -t ${SJTU_IMAGE} ${SJTU_DRONE_DIR}" >&2
  fi
  if ! docker image inspect "${SJTU_CYCLONE_IMAGE}" >/dev/null 2>&1; then
    echo >&2
    echo "[sjtu/env] NOTE: '${SJTU_CYCLONE_IMAGE}' is not present. It is a thin layer" >&2
    echo "           over ${SJTU_IMAGE} adding CycloneDDS, and it takes ~10 s to build:" >&2
    echo "             docker build -t ${SJTU_CYCLONE_IMAGE} $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" >&2
    echo "           Without it the simulator speaks Fast DDS 2.6, which the Foxy ROS 1" >&2
    echo "           bridge cannot decode -- FALCON would see an empty topic list." >&2
  fi
fi
