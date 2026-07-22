#!/usr/bin/env bash
# run_webcam_test.sh -- start the webcam drone-RGB mock + the target-lock mission at once.
#
# Launches webcam_frame_publisher (webcam -> /tmp/xtend_frames) in the background,
# then run_webcam_target_lock reading that folder. All arguments are forwarded to
# the mission runner, e.g.:
#     ./run_webcam_test.sh --target person --detector yolo
#     ./run_webcam_test.sh --target cup --detector color --color red --lock-mode detector
#
# The background publisher is killed when the mission window closes (or on Ctrl+C).
set -euo pipefail

# Repo root = four levels up from this script (tasks/planning/object_approach_webcam/).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
# Python: an explicit $PYTHON wins; else the shell's ACTIVE venv (so a venv that has
# ultralytics/torch is used for --detector yolo); else this repo's .venv.
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PY="$VIRTUAL_ENV/bin/python"
else
  PY="$REPO/.venv/bin/python"
fi
FRAMES="${FRAMES_DIR:-/tmp/xtend_frames}"
MOD="sparx_agency.tasks.planning.object_approach_webcam"

cd "$REPO"
echo "[webcam-test] publisher -> $FRAMES   (python: $PY)"
"$PY" -m "$MOD.webcam_frame_publisher" --out "$FRAMES" &
PUB_PID=$!
trap 'kill "$PUB_PID" 2>/dev/null || true' EXIT INT TERM

sleep 1.5   # let the first frames land before the mission starts polling
"$PY" -m "$MOD.run_webcam_target_lock" --images "$FRAMES" "$@"
