#!/usr/bin/env bash
# vllm_orin_nx_acceptance_test.sh
#
# Acceptance smoke test for running our actual production VLM (Qwen3-VL-4B
# AWQ via vLLM) on a Jetson Orin NX. Written to double as BOTH:
#
#   1. The pre-ship "install-and-prove" step we ask Robotican to run before
#      the drone leaves their facility -- proves Docker + the NVIDIA
#      container runtime work AND that the model we actually use fits in
#      whatever RAM variant (8GB/16GB) they shipped, instead of a generic
#      "GPU visible" check that wouldn't have caught either problem.
#   2. Our own re-run once the unit physically arrives.
#
# There is no official NVIDIA/jetson-containers catalog of which VLM configs
# are validated on Orin NX specifically (checked both jetson-containers' own
# vllm package docs and Jetson AI Lab's benchmark pages -- neither covers
# Orin NX) -- so this has to be established empirically, on the real device,
# which is exactly what this script does.
#
# If this OOMs: that is a real finding, not a bug in the script. The
# documented fallback (from this repo's own README) is to retry with
# --gpu-mem-util 0.4 --max-model-len 512, and if that still doesn't fit, a
# smaller/differently-quantized model is the next step -- chosen empirically
# once real numbers are in hand, not guessed at now.
#
# Usage:
#   ./vllm_orin_nx_acceptance_test.sh --model-dir ~/my_models/qwen3-vl-4b \
#       --image /path/to/test.jpg
#
# Requires: docker (with the NVIDIA container runtime), curl, jq, base64.
# tegrastats (Jetson-only) is used opportunistically for a memory snapshot;
# its absence does not fail the test.

set -u

IMAGE_NAME="vllm_qwen3_vl_4b_instruct_aws_4bit:latest"
MODEL_DIR="${HOME}/my_models/qwen3-vl-4b"
CONTAINER_NAME="vllm_orin_nx_acceptance_test"
PORT=8080
GPU_MEM_UTIL="0.5"
MAX_MODEL_LEN="2048"
HEALTH_TIMEOUT_SEC=180
TEST_IMAGE=""
TEST_PROMPT="Describe in a short list JUST the objects in the image."
KEEP_RUNNING=0

usage() {
  echo "Usage: $0 --image /path/to/test.jpg [options]"
  echo "  --model-dir DIR       Host dir mounted to /app/model (default: ${MODEL_DIR})"
  echo "  --image-name NAME     Docker image to run (default: ${IMAGE_NAME})"
  echo "  --port PORT           vLLM serve port (default: ${PORT})"
  echo "  --gpu-mem-util FLOAT  --gpu-memory-utilization passed to vllm serve (default: ${GPU_MEM_UTIL})"
  echo "  --max-model-len N     --max-model-len passed to vllm serve (default: ${MAX_MODEL_LEN})"
  echo "  --health-timeout-sec N  How long to wait for the server to come up (default: ${HEALTH_TIMEOUT_SEC})"
  echo "  --image PATH          REQUIRED: local test image to send in the inference call"
  echo "  --keep-running        Don't stop/remove the container when done"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --image-name) IMAGE_NAME="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --gpu-mem-util) GPU_MEM_UTIL="$2"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
    --health-timeout-sec) HEALTH_TIMEOUT_SEC="$2"; shift 2 ;;
    --image) TEST_IMAGE="$2"; shift 2 ;;
    --keep-running) KEEP_RUNNING=1; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown argument: $1"; usage ;;
  esac
done

if [[ -z "${TEST_IMAGE}" ]]; then
  echo "[vllm-acceptance] FAIL: --image is required (a local test image to send in the inference call)."
  usage
fi
if [[ ! -f "${TEST_IMAGE}" ]]; then
  echo "[vllm-acceptance] FAIL: --image path does not exist: ${TEST_IMAGE}"
  exit 1
fi
if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "[vllm-acceptance] FAIL: --model-dir does not exist: ${MODEL_DIR}"
  echo "  (this must already hold the Qwen3-VL-4B AWQ weights)"
  exit 1
fi
for bin in docker curl jq base64; do
  if ! command -v "${bin}" >/dev/null 2>&1; then
    echo "[vllm-acceptance] FAIL: required tool not found: ${bin}"
    exit 1
  fi
done

echo "[vllm-acceptance] image_name      : ${IMAGE_NAME}"
echo "[vllm-acceptance] model_dir       : ${MODEL_DIR}"
echo "[vllm-acceptance] port            : ${PORT}"
echo "[vllm-acceptance] gpu_mem_util    : ${GPU_MEM_UTIL}"
echo "[vllm-acceptance] max_model_len   : ${MAX_MODEL_LEN}"

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1

echo "[vllm-acceptance] starting container..."
docker run -d --name "${CONTAINER_NAME}" \
  --runtime nvidia \
  --network host \
  -v "${MODEL_DIR}:/app/model" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  "${IMAGE_NAME}" \
  vllm serve /app/model \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --dtype float16 \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --enforce-eager \
  > /dev/null

if [[ $? -ne 0 ]]; then
  echo "[vllm-acceptance] FAIL: docker run failed to start the container."
  exit 1
fi

echo "[vllm-acceptance] waiting up to ${HEALTH_TIMEOUT_SEC}s for the server to come up..."
deadline=$(( $(date +%s) + HEALTH_TIMEOUT_SEC ))
server_up=0
while [[ $(date +%s) -lt ${deadline} ]]; do
  if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "200"; then
    server_up=1
    break
  fi
  if ! docker ps --filter "name=^${CONTAINER_NAME}$" --format '{{.Names}}' | grep -q "${CONTAINER_NAME}"; then
    echo "[vllm-acceptance] FAIL: container exited before the server came up. Logs:"
    docker logs "${CONTAINER_NAME}" 2>&1 | tail -n 60
    exit 1
  fi
  sleep 3
done

if [[ "${server_up}" -ne 1 ]]; then
  echo "[vllm-acceptance] FAIL: server did not respond within ${HEALTH_TIMEOUT_SEC}s. Logs:"
  docker logs "${CONTAINER_NAME}" 2>&1 | tail -n 60
  [[ "${KEEP_RUNNING}" -eq 0 ]] && docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1
  exit 1
fi
echo "[vllm-acceptance] server is up."

echo "[vllm-acceptance] taking a tegrastats memory snapshot (Jetson-only, best-effort)..."
if command -v tegrastats >/dev/null 2>&1; then
  timeout 2 tegrastats --interval 1000 2>/dev/null | head -n 1
else
  echo "[vllm-acceptance] tegrastats not found (not running on a Jetson, or not on PATH) -- skipping."
fi

echo "[vllm-acceptance] sending one real inference request..."
IMAGE_B64=$(base64 -w0 "${TEST_IMAGE}" 2>/dev/null || base64 "${TEST_IMAGE}")
EXT="${TEST_IMAGE##*.}"
PAYLOAD=$(jq -n \
  --arg prompt "${TEST_PROMPT}" \
  --arg data_url "data:image/${EXT};base64,${IMAGE_B64}" \
  '{
    model: "/app/model",
    messages: [{
      role: "user",
      content: [
        {type: "text", text: $prompt},
        {type: "image_url", image_url: {url: $data_url}}
      ]
    }],
    max_tokens: 64
  }')

t0=$(date +%s.%N)
RESPONSE=$(curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H "Content-Type: application/json" -d "${PAYLOAD}")
t1=$(date +%s.%N)
elapsed=$(echo "${t1} - ${t0}" | bc 2>/dev/null || awk "BEGIN{print ${t1}-${t0}}")

CONTENT=$(echo "${RESPONSE}" | jq -r '.choices[0].message.content // empty' 2>/dev/null)

echo ""
echo "[vllm-acceptance] ---- result ----"
echo "[vllm-acceptance] latency_sec : ${elapsed}"
if [[ -n "${CONTENT}" ]]; then
  echo "[vllm-acceptance] response    : ${CONTENT}"
else
  echo "[vllm-acceptance] response    : <empty/invalid -- raw response below>"
  echo "${RESPONSE}"
fi

if [[ "${KEEP_RUNNING}" -eq 0 ]]; then
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1
else
  echo "[vllm-acceptance] --keep-running set: container ${CONTAINER_NAME} left running on port ${PORT}."
fi

if [[ -n "${CONTENT}" ]]; then
  echo "[vllm-acceptance] PASS"
  echo "[vllm-acceptance] REMINDER: record which Orin NX RAM variant (8GB/16GB) this ran on --"
  echo "[vllm-acceptance]           that number is what everything downstream of this test depends on."
  exit 0
else
  echo "[vllm-acceptance] FAIL: server responded but produced no usable content."
  echo "[vllm-acceptance] If this looks like an OOM, retry with:"
  echo "[vllm-acceptance]   --gpu-mem-util 0.4 --max-model-len 512"
  echo "[vllm-acceptance] and if that still fails, this model does not fit on this device -- report the"
  echo "[vllm-acceptance] finding rather than continuing to guess at flags."
  exit 1
fi
