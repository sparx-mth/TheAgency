#!/usr/bin/env bash
# Run each warm-up configuration over the same HM3D v1 start-floor episodes, one scene at a time.
#
#   run_compare.sh                       # every config in configs/ (about 120 episodes each)
#   run_compare.sh baseline scan360      # only these configs
#   OUT=runs/my_compare run_compare.sh   # resume: finished (config, scene) directories are skipped
#   PER_SCENE=10 run_compare.sh          # a short test: the first 10 listed episodes of each scene
#
# Services come from hm3d/run_benchmark.sh (HAB_PY, DETECT_PY, HM3D_EPISODES_DIR, HM3D_SCENES_DIR, OLLAMA_MODE ...).
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$HERE/../../hm3d/run_benchmark.sh"
OUT="${OUT:-runs/warmup_compare_$(date -u +%Y%m%d)}"
EPISODES="$HERE/episodes_hm3d_v1_start_floor.txt"

# A resident model takes VRAM, so the next launch's GPU plan would move YOLO to the CPU: every launch starts empty.
unload_models() { ollama ps 2>/dev/null | awk 'NR > 1 {print $1}' | xargs -r -n1 ollama stop || true; }

CONFIGS=("$@")
[[ ${#CONFIGS[@]} -gt 0 ]] || { for f in "$HERE"/configs/*.json; do CONFIGS+=("$(basename "$f" .json)"); done; }

for config in "${CONFIGS[@]}"; do
  while read -r scene; do
    target="$OUT/$config/$scene"
    [[ -f "$target/summary.json" ]] && { echo "skip $config/$scene (finished)"; continue; }
    mapfile -t ids < <(grep "^$scene/" "$EPISODES" | head -n "${PER_SCENE:-40}")
    unload_models
    sleep 5
    "$LAUNCHER" --version v1 --scene "$scene" --output "$target" \
      --policy-config "$HERE/configs/$config.json" --episode-ids "${ids[@]}"
  done < <(cut -d/ -f1 "$EPISODES" | uniq)
done
