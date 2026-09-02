#!/bin/bash
# ============================================================
# stage_vendor_msgs.sh — copy the ROBOTICAN vendor message packages into the
# Docker build context so Dockerfile.robotican can build them for Humble.
#
# Two vendor packages, from two separate workspaces outside the repo:
#   - rooster_interfaces at ~/rqs_iai_ws/src/rooster_interfaces/ (the old
#     rqs7/`it`-container FCU backend's messages, delivered by the vendor)
#   - sphera_common_interfaces at ~/sphera_ws/src/sphera_common_interfaces/
#     (the new Sphera engine-level messages -- SpheraPawnState and friends,
#     what /R1/sphera/state and /R1/sphera/set_state carry. As of the new
#     Sphera build this ships real .msg/.srv source, not just a prebuilt
#     Foxy binary inside Sphera's own image -- earlier versions of this
#     script and Dockerfile.robotican predate that and left it out)
#
# A Docker build context can't reach outside docker/, so this script copies
# both into a gitignored vendor/ dir right before the build. Vendor code
# never enters git; the image stays self-contained.
#
# Run this once (or whenever either vendor tree changes) before building
# Dockerfile.robotican.
# ============================================================
set -euo pipefail

ROOSTER_SRC="${1:-$HOME/rqs_iai_ws/src/rooster_interfaces}"
SPHERA_SRC="${2:-$HOME/sphera_ws/src/sphera_common_interfaces}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="$HERE/vendor"

stage() {
  local src="$1" dest="$2" label="$3"
  if [ ! -d "$src" ]; then
    echo "error: $label source not found at $src" >&2
    echo "  (pass the path explicitly: $0 <rooster_interfaces_path> <sphera_common_interfaces_path>)" >&2
    exit 1
  fi
  rm -rf "$dest"
  mkdir -p "$(dirname "$dest")"
  cp -r "$src" "$dest"
  local count
  count=$(find "$dest" -name '*.msg' -o -name '*.srv' | wc -l)
  echo "staged $count .msg/.srv files from $src -> $dest"
}

stage "$ROOSTER_SRC" "$VENDOR_DIR/rooster_interfaces" "rooster_interfaces"
stage "$SPHERA_SRC" "$VENDOR_DIR/sphera_common_interfaces" "sphera_common_interfaces"
