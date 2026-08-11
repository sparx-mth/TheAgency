#!/bin/bash
# ============================================================
# stage_vendor_msgs.sh — copy the ROBOTICAN vendor message packages into the
# Docker build context so the image can build them for Humble.
#
# The .msg/.srv sources only exist outside the repo, at
# ~/rqs_iai_ws/src/rooster_interfaces/ (delivered by the vendor, not this
# repo). A Docker build context can't reach outside docker/devcontainer/, so
# this script copies them into a gitignored vendor/ dir right before the
# build. Vendor code never enters git; the image stays self-contained.
#
# Run this once (or whenever the vendor tree changes) before `docker build`.
# ============================================================
set -euo pipefail

SRC="${1:-$HOME/rqs_iai_ws/src/rooster_interfaces}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/vendor/rooster_interfaces"

if [ ! -d "$SRC" ]; then
  echo "error: vendor source not found at $SRC" >&2
  echo "  (pass the path explicitly: $0 /path/to/rooster_interfaces)" >&2
  exit 1
fi

rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
cp -r "$SRC" "$DEST"

count=$(find "$DEST" -name '*.msg' -o -name '*.srv' | wc -l)
echo "staged $count .msg/.srv files from $SRC -> $DEST"
