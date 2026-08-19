#!/usr/bin/env bash
# ============================================================
# record_run.sh — fly + RECORD an InternVLA-N1 run in the SJTU hospital.
#
#   ./record_run.sh                         # hospital, the exploration order, until Ctrl-C
#   ./record_run.sh hospital "go to the pharmacy"   # any world / instruction
#   RECORD_SECONDS=120 ./record_run.sh      # fly 2 minutes, then tear down + report
#
# Thin wrapper over run_sjtu_n1.sh with RECORD=1: it produces
#   • an MP4  — drone camera (left) + N1's route top-down (right) + System-1/2 FPS
#   • a rosbag — every relevant topic, for replay/analysis
# and prints the measured S1/S2 FPS at the end (next to the optimizer's numbers).
#
# The default order is the one asked for; override as arg 2 or $INSTRUCTION.
# ============================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORLD="${1:-hospital}"
INSTRUCTION="${2:-Explore the entire hospital, enter all the rooms, reach every area at least once}"

exec env \
    RECORD=1 \
    RECORD_OUTPUT="${RECORD_OUTPUT:-${SJTU_N1_LOG_DIR:-/tmp/sjtu_n1}/hospital_run.mp4}" \
    "${SCRIPT_DIR}/run_sjtu_n1.sh" "${WORLD}" "${INSTRUCTION}"

