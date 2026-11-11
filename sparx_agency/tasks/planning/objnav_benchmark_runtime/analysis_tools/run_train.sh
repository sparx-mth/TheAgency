#!/usr/bin/env bash
# Phase A: the unchanged method on seeded single-floor HM3D v1 TRAIN episodes. CPU LLM (Ollama 11435), CPU detector (18192); Habitat is the only GPU user.
set -uo pipefail
REPO="$HOME/GIT/TheAgency"; HERE="$REPO/sparx_agency/tasks/planning/objnav_benchmark_runtime/trace_tools"
HAB_PY="$HOME/miniconda3/envs/objnav-habitat/bin/python"; OUT="${OUT:-$HOME/trace_train_a}"; PER="${PER:-5}"
export LLM_MODEL=qwen2.5:3b-instruct LLM_REASONING_MODEL=qwen2.5:3b-instruct LLM_BASE_URL=http://127.0.0.1:11435
export PYTHONPATH="$REPO" HM3D_EPISODES_DIR="$HOME/datasets_v01/objectnav/hm3d/v1/train" HM3D_SCENES_DIR="$HOME/datasets_v01/scene_datasets"
cd "$REPO"
for scene in ${SCENES:-$(python3 -c "import json;print(' '.join(json.load(open('$HOME/tc_sweep/train_episodes.json'))))")}; do
  mapfile -t ids < <(python3 -c "import json;print('\n'.join(json.load(open('$HOME/tc_sweep/train_episodes.json'))['$scene'][:$PER]))")
  mkdir -p "$OUT/$scene"
  TRACE_OUT="$OUT/$scene" nice -n 5 "$HAB_PY" -u "$HERE/trace_run.py" --version v1 --split train --seed 17 --gpu-device 0 \
    --detector-url http://127.0.0.1:18192 --detector-backend yolo_world --explorer frontier --lean --allow-shared-gpu \
    --scenes "$scene" --episodes-per-scene 50000 --episode-ids "${ids[@]}" --output "$OUT/$scene/run" >> "$OUT/$scene/run.log" 2>&1 || echo "$scene failed" >> "$OUT/failed.txt"
done
echo done > "$OUT/done.txt"
