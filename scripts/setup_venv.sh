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
#   user    @ user-agx1   → jetson_agx1_15w     (Jetson AGX Orin — depth/bridge)
#   user    @ user-agx2   → jetson_agx2_ollama  (Jetson AGX Orin — Ollama LLM, ROS2 Jazzy/py3.12)
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
        ROS_DISTRO="humble"
        ROS_PYTHON_VER="3.10"
        HAS_CUDA_LD="yes"
        ;;
    user@user-agx2|user-agx2|agx2)
        DEVICE_FILE="${DEVICE_REQ_DIR}/jetson_agx2_ollama.txt"
        VENV_OPTS="--system-site-packages"
        ROS_DISTRO="jazzy"
        ROS_PYTHON_VER="3.12"
        HAS_CUDA_LD="yes"
        ;;
    daphnaa@*|*daphna*|*laptop*)
        DEVICE_FILE="${DEVICE_REQ_DIR}/laptop_home_daphna.txt"
        VENV_OPTS=""
        ROS_DISTRO="humble"
        ROS_PYTHON_VER="3.10"
        HAS_CUDA_LD="no"
        ;;
    user1@PCN87653|user1@*|PCN87653|*office*)
        DEVICE_FILE="${DEVICE_REQ_DIR}/office_pc.txt"
        VENV_OPTS=""
        ROS_DISTRO="jazzy"
        ROS_PYTHON_VER="3.12"
        HAS_CUDA_LD="no"
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

# --- create venv if missing or broken ---
if [ ! -f "${VENV_DIR}/bin/pip" ]; then
    [ -d "${VENV_DIR}" ] && echo "[setup_venv] Removing broken venv ..." && rm -rf "${VENV_DIR}"
    echo "[setup_venv] Creating venv ..."
    # shellcheck disable=SC2086
    python3 -m venv ${VENV_OPTS} "${VENV_DIR}"
fi

# --- install ---
echo "[setup_venv] Upgrading pip ..."
"${VENV_DIR}/bin/pip" install --upgrade pip --quiet

echo "[setup_venv] Installing requirements ..."
"${VENV_DIR}/bin/pip" install -r "${ACTIVE_REQ}"

# --- add repo root to sys.path via .pth file so 'import sparx_agency' works ---
PYTHON_VER=$("${VENV_DIR}/bin/python3" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "${REPO_DIR}" > "${VENV_DIR}/lib/python${PYTHON_VER}/site-packages/theagency.pth"
echo "[setup_venv] Added ${REPO_DIR} to venv sys.path (theagency.pth)"

# --- write agency() shell function to ~/.bashrc ---
echo "[setup_venv] Writing agency() function to ~/.bashrc ..."
export _SETUP_REPO_DIR="${REPO_DIR}"
export _SETUP_ROS_DISTRO="${ROS_DISTRO}"
export _SETUP_ROS_PYTHON_VER="${ROS_PYTHON_VER}"
export _SETUP_HAS_CUDA_LD="${HAS_CUDA_LD}"

python3 - << 'PYEOF'
import os, re, pathlib

repo    = os.environ['_SETUP_REPO_DIR']
distro  = os.environ['_SETUP_ROS_DISTRO']
pyver   = os.environ['_SETUP_ROS_PYTHON_VER']
cuda_ld = os.environ['_SETUP_HAS_CUDA_LD'] == 'yes'

ros_base = f"/opt/ros/{distro}"
lines = [
    "agency() {",
    f"    cd {repo}",
    f"    source {ros_base}/setup.bash",
    f"    source {repo}/venv/bin/activate",
    "    export ROS_DOMAIN_ID=5",
    "    export PYTHONUNBUFFERED=1",
]
if cuda_ld:
    lines.append(
        f"    export LD_LIBRARY_PATH={ros_base}/opt/rviz_ogre_vendor/lib:"
        f"{ros_base}/lib/aarch64-linux-gnu:{ros_base}/lib:"
        "/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
    )
lines.append(
    f"    export PYTHONPATH={ros_base}/lib/python{pyver}/site-packages:"
    f"{ros_base}/local/lib/python{pyver}/dist-packages:"
    f"{repo}:{repo}/sparx_agency:${{PYTHONPATH}}"
)
lines.append("}")
func_text = "\n".join(lines)

bashrc = pathlib.Path.home() / '.bashrc'
text = bashrc.read_text(encoding='utf-8') if bashrc.exists() else ''
text = re.sub(r'\nagency\(\) \{.*?\n\}', '', text, flags=re.DOTALL)
text = text.rstrip('\n') + '\n\n' + func_text + '\n'
bashrc.write_text(text, encoding='utf-8')
print(f'[setup_venv] agency() written to ~/.bashrc  (ros={distro}, domain=5)')
PYEOF

echo ""
echo "[setup_venv] Done."
echo "[setup_venv] Activate with:  source ${VENV_DIR}/bin/activate"
echo "[setup_venv] Then run:       agency"