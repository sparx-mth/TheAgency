#!/usr/bin/env bash
# Start the FlowNav image-goal TensorRT host server (loopback) for FALCON.
#
# Mirrors the manual NavDP server start: run this on the GPU host, in the
# flownav_trt env, BEFORE launching FALCON with vla:=flownav. The FALCON adapter
# container reaches it over --network host + 127.0.0.1:<port>.
#
# Env overrides:
#   PY          python interpreter for the flownav_trt env (default: python3)
#   PORT        loopback port (default: 8889)
#   TAG         engine target tag (default: auto-detected for this GPU)
#   ENGINE_DIR  engine directory (default: .../flownav/engines/<TAG>)
# Extra args are forwarded to the server (e.g. --num-steps 4 --context-size 3).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# serve -> flownav -> vlas -> planning -> tasks -> sparx_agency -> repo root (6 up)
REPO_ROOT="$(cd "${HERE}/../../../../../.." && pwd)"  # dir containing sparx_agency/
PY="${PY:-python3}"
PORT="${PORT:-8889}"
TAG="${TAG:-$(PYTHONPATH="${REPO_ROOT}" "${PY}" -c \
  'from sparx_agency.tasks.planning.vlas.common.hardware.detect import detect; print(detect().target_tag)')}"
ENGINE_DIR="${ENGINE_DIR:-${REPO_ROOT}/sparx_agency/tasks/planning/vlas/flownav/trt/engines/${TAG}}"

echo "[flownav-server] py=${PY} port=${PORT} engine-dir=${ENGINE_DIR}"
PYTHONPATH="${REPO_ROOT}" exec "${PY}" \
  -m sparx_agency.tasks.planning.vlas.flownav.serve.flownav_trt_server \
  --engine-dir "${ENGINE_DIR}" --port "${PORT}" "$@"
