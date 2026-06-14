#!/usr/bin/env bash
# setup_venv.sh — create or update the project venv for this device.
#
# Usage:
#   bash scripts/setup_venv.sh                  # auto-detect by username + hostname
#   bash scripts/setup_venv.sh <device_name>    # override (e.g. jetson / office / laptop)
#
# To add a new device: add a case entry below and create the matching
# requirements/devices/<name>.txt file.
#
# Known devices:
#   user    @ user-agx1   → jetson_agx1_15w     (Jetson AGX Orin)
#   user1   @ PCN87653    → office_pc            (office PC)
#   daphnaa @ *           → laptop_home_daphna   (home laptop)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_DIR}/venv"
DEVICE_REQ_DIR="${REPO_DIR}/requirements/devices"
ACTIVE_REQ="${REPO_DIR}/requirements/active.txt"

# --- device detection: user@hostname, overridable ---
DEVICE="${1:-$(whoami)@$(hostname)}"

case "${DEVICE}" in
    user@user-agx1|user@agx1|user@jetson*|user-agx1|agx1|jetson*)
        DEVICE_FILE="${DEVICE_REQ_DIR}/jetson_agx1_15w.txt"
        VENV_OPTS="--system-site-packages"
        ;;
    daphnaa@*|*daphna*|*laptop*)
        DEVICE_FILE="${DEVICE_REQ_DIR}/laptop_home_daphna.txt"
        VENV_OPTS=""
        ;;
    user1@PCN87653|user1@*|PCN87653|*office*)
        DEVICE_FILE="${DEVICE_REQ_DIR}/office_pc.txt"
        VENV_OPTS=""
        ;;
    *)
        echo "[setup_venv] Unknown device: '${DEVICE}'"
        echo "[setup_venv] Available device files:"
        ls "${DEVICE_REQ_DIR}/"
        echo ""
        echo "Usage: bash scripts/setup_venv.sh <device_name>"
        echo "       e.g.: bash scripts/setup_venv.sh jetson"
        exit 1
        ;;
esac

echo "[setup_venv] Device:   ${DEVICE}"
echo "[setup_venv] Req file: ${DEVICE_FILE}"
echo "[setup_venv] Venv:     ${VENV_DIR}"
echo ""

# --- copy to active ---
cp "${DEVICE_FILE}" "${ACTIVE_REQ}"
echo "[setup_venv] Copied requirements -> ${ACTIVE_REQ}"

# --- create venv if missing ---
if [ ! -d "${VENV_DIR}" ]; then
    echo "[setup_venv] Creating venv ..."
    # shellcheck disable=SC2086
    python3 -m venv ${VENV_OPTS} "${VENV_DIR}"
fi

# --- install ---
echo "[setup_venv] Upgrading pip ..."
"${VENV_DIR}/bin/pip" install --upgrade pip --quiet

echo "[setup_venv] Installing requirements ..."
"${VENV_DIR}/bin/pip" install -r "${ACTIVE_REQ}"

echo ""
echo "[setup_venv] Done."
echo "[setup_venv] Activate with:  source ${VENV_DIR}/bin/activate"