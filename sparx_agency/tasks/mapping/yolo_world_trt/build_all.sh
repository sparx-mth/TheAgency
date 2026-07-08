#!/usr/bin/env bash
# Export + build (+ optionally benchmark) all four open-set YOLO-World splits.
#
# For each variant s/m/l/x it exports the backbone+head ONNX (open-set: text is a
# runtime input, NOT baked), builds the backbone(DLA) + head(GPU) engines, and --
# if IMAGES is set -- benchmarks them. Export needs ultralytics+torch+onnx; build +
# benchmark need tensorrt+pycuda ON THE TARGET (DLA only exists on Jetson).
#
# Usage:
#   export WEIGHTS_DIR=/path/to/yolo_world_weights   # holds yolov8{s,m,l,x}-worldv2.pt
#   export IMAGES=/path/to/bench_frames              # optional: enables benchmark
#   ./build_all.sh [variants...]
#
#   ./build_all.sh            # all four:  s m l x
#   ./build_all.sh s          # just the small model (start here)
#   ./build_all.sh s m        # small + medium
#
# Variants may also come from the VARIANTS env var; positional args win over it.
# Env overrides: PYTHON (default python3), ONNX_DIR, IMGSZ (else config default),
#   PRECISION (fp16|int8), DLA (auto|on|off), N_MAX (head dynamic-N ceiling),
#   NUM_PROMPTS (benchmark prompt count).
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  # Print the top comment block (skip the shebang; stop at the first code line).
  awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"; exit 0
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
MOD="sparx_agency.tasks.mapping.yolo_world_trt"
ONNX_DIR="${ONNX_DIR:-$HERE/engines/onnx}"

# Variants: positional args win, else the VARIANTS env var, else all four.
if [[ $# -gt 0 ]]; then
  VARIANTS="$*"
else
  VARIANTS="${VARIANTS:-s m l x}"
fi
for v in $VARIANTS; do
  case "$v" in
    s|m|l|x) ;;
    *) echo "error: unknown variant '$v' (allowed: s m l x)" >&2; exit 2;;
  esac
done

WEIGHTS_DIR="${WEIGHTS_DIR:?set WEIGHTS_DIR to the folder holding the .pt checkpoints}"

imgsz_flag=(); [[ -n "${IMGSZ:-}" ]] && imgsz_flag=(--imgsz "$IMGSZ")
nmax_flag=();  [[ -n "${N_MAX:-}" ]] && nmax_flag=(--n-max "$N_MAX")
prec_flag=();  [[ -n "${PRECISION:-}" ]] && prec_flag=(--precision "$PRECISION")
dla_flag=()
case "${DLA:-auto}" in on) dla_flag=(--dla);; off) dla_flag=(--no-dla);; esac

echo "== YOLO-World open-set TRT build_all =="
echo "   variants=$VARIANTS  weights_dir=$WEIGHTS_DIR  onnx_dir=$ONNX_DIR"

for v in $VARIANTS; do
  pt="$WEIGHTS_DIR/yolov8${v}-worldv2.pt"
  [[ -f "$pt" ]] || pt="$WEIGHTS_DIR/yolov8${v}-world.pt"
  echo "--- [$v] export backbone+head ($pt) ---"
  "$PYTHON" -m "$MOD.export_onnx" --weights "$pt" --variant "$v" \
      "${imgsz_flag[@]}" "${nmax_flag[@]}" --out-dir "$ONNX_DIR"
done

for v in $VARIANTS; do
  echo "--- [$v] build backbone(DLA) + head(GPU) ---"
  "$PYTHON" -m "$MOD.build_engine" --onnx-dir "$ONNX_DIR" --variant "$v" \
      --role both "${prec_flag[@]}" "${dla_flag[@]}"
done

TARGET_DIR="$("$PYTHON" -c "from $MOD.hardware import detect; print(detect().target_tag)")"
ENG_DIR="$HERE/engines/$TARGET_DIR"
echo "== engines in: $ENG_DIR =="
ls -1 "$ENG_DIR"/*.engine 2>/dev/null || true

if [[ -n "${IMAGES:-}" ]]; then
  echo "--- benchmark ($IMAGES) ---"
  pair_flags=()
  for v in $VARIANTS; do
    b=$(ls -1 "$ENG_DIR"/yolo_world_${v}.backbone.*.engine 2>/dev/null | head -n1 || true)
    h=$(ls -1 "$ENG_DIR"/yolo_world_${v}.head.*.engine 2>/dev/null | head -n1 || true)
    [[ -n "$b" && -n "$h" ]] && pair_flags+=(--pair "${v}:${b},${h}")
  done
  "$PYTHON" -m "$MOD.benchmark" --images "$IMAGES" --num-prompts "${NUM_PROMPTS:-4}" \
      "${pair_flags[@]}" --out /tmp/yolo_world_trt_compare
fi
echo "== done =="
