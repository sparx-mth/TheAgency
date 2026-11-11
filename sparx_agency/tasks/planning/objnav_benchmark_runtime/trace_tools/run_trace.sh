#!/usr/bin/env bash
# The unchanged method with a trace, 3B room labels and node oracle. Services: Ollama (qwen2.5:3b-instruct) and the detector.
# VERSION=v1 uses the 20 episodes in episodes_lm.json; set EPISODES_PER_SCENE=N to take the first N of each scene instead.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../../../../.." && pwd)}"
HAB_PY="${HAB_PY:-$HOME/miniconda3/envs/objnav-habitat/bin/python}"
VERSION="${VERSION:-v1}"
export LLM_MODEL=qwen2.5:3b-instruct LLM_REASONING_MODEL=qwen2.5:3b-instruct LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:11434}"
export PYTHONPATH="$REPO" HM3D_EPISODES_DIR="${HM3D_EPISODES_DIR:-$HOME/datasets/objectnav/hm3d/$VERSION/val}" HM3D_SCENES_DIR="${HM3D_SCENES_DIR:-$HOME/datasets/scene_datasets}"
DETECTOR_URL="${DETECTOR_URL:-http://127.0.0.1:18192}"
OUT="${OUT:-$HOME/trace_out}"
cd "$REPO"
run_scene() {
  TRACE_OUT="$OUT/$scene" "$HAB_PY" -u "$HERE/trace_run.py" --version "$VERSION" --split val --seed 17 --gpu-device 0 \
    --detector-url "$DETECTOR_URL" --detector-backend "${DETECTOR_BACKEND:-yolo_world}" --explorer frontier --lean --allow-shared-gpu "$@" \
    --scenes "$scene" "${selection[@]}" --output "$OUT/$scene/run" >> "$OUT/$scene/run.log" 2>&1
}
for scene in ${SCENES:-00853-5cdEh9F2hJL 00890-6s7QHgap2fW}; do
  if [ -n "${EPISODES_PER_SCENE:-}" ]; then selection=(--episodes-per-scene "$EPISODES_PER_SCENE")
  else mapfile -t ids < <(grep -o "$scene/[0-9]*" "$HERE/episodes_lm.json"); selection=(--episode-ids "${ids[@]}"); fi
  mkdir -p "$OUT/$scene"
  if ! run_scene --validate-starts && grep -q "reproduce the publisher's own geodesic" "$OUT/$scene/run.log"; then
    echo "start validation refused this scene; rerun without it" > "$OUT/$scene/validation_skipped.txt"
    run_scene || echo "$scene failed" >> "$OUT/failed.txt"
  fi
done
echo done > "$OUT/done.txt"
