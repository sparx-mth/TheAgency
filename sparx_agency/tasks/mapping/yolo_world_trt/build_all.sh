#!/usr/bin/env bash
# Export + build (+ optionally benchmark) all four YOLO-World variants end to end.
#
# Runs the whole pipeline for s/m/l/x with ONE baked prompt set and ONE imgsz, so
# the four engines are directly comparable. Export needs ultralytics+torch+onnx;
# build + benchmark need tensorrt+pycuda ON THE TARGET (DLA only exists on Jetson).
#
# Usage:
#   export PROMPTS="refrigerator chair door person"     # the baked mission vocabulary
#   export WEIGHTS_DIR=/path/to/yolo_world_weights       # holds yolov8{s,m,l,x}-worldv2.pt
#   export IMAGES=/path/to/bench_frames                  # optional: enables benchmark
#   ./build_all.sh
#
# Env overrides: VARIANTS (default "s m l x"), PYTHON (default python3),
#   ONNX_DIR, IMGSZ (else config default), PRECISION (fp16|int8), DLA (auto|on|off).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"     # -> repo root (TheAgency)
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
MOD="sparx_agency.tasks.mapping.yolo_world_trt"
VARIANTS="${VARIANTS:-s m l x}"
ONNX_DIR="${ONNX_DIR:-$HERE/engines/onnx}"
PROMPTS="${PROMPTS:?set PROMPTS to the baked class list, e.g. 'refrigerator chair'}"
WEIGHTS_DIR="${WEIGHTS_DIR:?set WEIGHTS_DIR to the folder holding the .pt checkpoints}"

imgsz_flag=(); [[ -n "${IMGSZ:-}" ]] && imgsz_flag=(--imgsz "$IMGSZ")
prec_flag=();  [[ -n "${PRECISION:-}" ]] && prec_flag=(--precision "$PRECISION")
dla_flag=()
case "${DLA:-auto}" in on) dla_flag=(--dla);; off) dla_flag=(--no-dla);; esac

echo "== YOLO-World TRT build_all =="
echo "   variants=$VARIANTS  prompts=[$PROMPTS]  weights_dir=$WEIGHTS_DIR"
echo "   onnx_dir=$ONNX_DIR"

for v in $VARIANTS; do
  pt="$WEIGHTS_DIR/yolov8${v}-worldv2.pt"
  [[ -f "$pt" ]] || pt="$WEIGHTS_DIR/yolov8${v}-world.pt"
  echo "--- [$v] export ($pt) ---"
  "$PYTHON" -m "$MOD.export_onnx" --weights "$pt" --variant "$v" \
      --prompts $PROMPTS "${imgsz_flag[@]}" --out-dir "$ONNX_DIR"
done

engines=()
for v in $VARIANTS; do
  echo "--- [$v] build engine ---"
  "$PYTHON" -m "$MOD.build_engine" --onnx "$ONNX_DIR/yolo_world_${v}.onnx" \
      --variant "$v" "${prec_flag[@]}" "${dla_flag[@]}"
done

# Discover the engines the builder wrote (target_tag dir is a sibling of onnx/).
TARGET_DIR="$("$PYTHON" -c "from $MOD.hardware import detect; print(detect().target_tag)")"
ENG_DIR="$HERE/engines/$TARGET_DIR"
echo "== engines in: $ENG_DIR =="
ls -1 "$ENG_DIR"/*.engine 2>/dev/null || true

if [[ -n "${IMAGES:-}" ]]; then
  echo "--- benchmark ($IMAGES) ---"
  eng_flags=()
  for v in $VARIANTS; do
    e=$(ls -1 "$ENG_DIR"/yolo_world_${v}.*.engine 2>/dev/null | head -n1 || true)
    [[ -n "$e" ]] && eng_flags+=(--engine "${v}:${e}")
  done
  "$PYTHON" -m "$MOD.benchmark" --images "$IMAGES" "${eng_flags[@]}" \
      --out /tmp/yolo_world_trt_compare
fi
echo "== done =="
