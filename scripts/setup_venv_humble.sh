#!/usr/bin/env bash
# setup_venv_humble.sh — create or update the project venv for the Humble devcontainer profile.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_DIR}/venv"
DEVICE_REQ_DIR="${REPO_DIR}/requirements/devices"
ACTIVE_REQ="${REPO_DIR}/requirements/active.txt"
DEVICE="${1:-office_pc}"
DEVICE_FILE="${DEVICE_REQ_DIR}/${DEVICE}.txt"

if [[ "${DEVICE}" != "office_pc" ]]; then
    echo "[setup_venv_humble] This helper is intended for the Humble devcontainer profile."
    echo "[setup_venv_humble] Supported device: office_pc"
    exit 1
fi

if [[ ! -f "${DEVICE_FILE}" ]]; then
    echo "[setup_venv_humble] ERROR: requirements file not found -> ${DEVICE_FILE}" >&2
    exit 1
fi

echo "[setup_venv_humble] Device:   ${DEVICE}"
echo "[setup_venv_humble] Req file: ${DEVICE_FILE}"
echo "[setup_venv_humble] Venv:     ${VENV_DIR}"
echo ""

mkdir -p "$(dirname "${ACTIVE_REQ}")"
install -m 0644 "${DEVICE_FILE}" "${ACTIVE_REQ}"
echo "[setup_venv_humble] Copied requirements -> ${ACTIVE_REQ}"
#
if [ ! -f "${VENV_DIR}/bin/pip" ]; then
   [ -d "${VENV_DIR}" ] && echo "[setup_venv_humble] Removing broken venv ..." && sudo find "$VENV_DIR" -mindepth 1 -delete 2>/dev/null || true
#    echo "[setup_venv_humble] Creating venv ..."
#    python3 -m venv "${VENV_DIR}"
fi

sudo chown -R vscode:vscode "$VENV_DIR"
sudo chmod -R 755 "$VENV_DIR"
sudo mkdir -p /home/vscode/.cache/pip
sudo chown -R vscode:vscode /home/vscode/.cache
sudo chmod -R 755 /home/vscode/.cache


if [ ! -d "$VENV_DIR/lib" ]; then
    echo "Creating clean Python virtual environment..."
    python3 -m venv "$VENV_DIR" --copies
fi

echo "[setup_venv_humble] Upgrading pip ..."
"${VENV_DIR}/bin/pip" install --upgrade pip --quiet

echo "[setup_venv_humble] Installing requirements ..."
"${VENV_DIR}/bin/pip" install -r "${ACTIVE_REQ}"

PYTHON_VER=$("${VENV_DIR}/bin/python3" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "${REPO_DIR}" > "${VENV_DIR}/lib/python${PYTHON_VER}/site-packages/theagency.pth"
echo "[setup_venv_humble] Added ${REPO_DIR} to venv sys.path (theagency.pth)"

cat >> "${HOME}/.bashrc" <<'EOF'

# TheAgency Humble helper
agency() {
    cd /workspace
    source /opt/ros/humble/setup.bash
    source /workspace/venv/bin/activate
    export ROS_DOMAIN_ID=5
    export PYTHONUNBUFFERED=1
    export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:/workspace:/workspace/sparx_agency:${PYTHONPATH}
}
EOF

echo "[setup_venv_humble] Added agency() helper to ~/.bashrc"
echo ""
echo "[setup_venv_humble] Done."
echo "[setup_venv_humble] Activate with:  source ${VENV_DIR}/bin/activate"
echo "[setup_venv_humble] Then run:       agency"
