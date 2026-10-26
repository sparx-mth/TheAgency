#!/usr/bin/env bash
# Run configs over a slice of the listed episodes, with a start deadline and a choice of loop order.
#
#   run_queue.sh [--first N] [--count N] [--tag SUFFIX] [--by-scene] [--deadline HH:MM] config...
#
#   --first / --count  episodes FIRST..FIRST+COUNT-1 of each scene's list (defaults 1 and 10)
#   --tag              appended to each scene directory, so a later slice joins a config's earlier results
#   --by-scene         scene outermost, configs inside it: a cut-off leaves every config with the same scenes
#   --deadline         no launch starts after this time (today, or tomorrow if already past); a launch under way finishes
#
# OUT names the output root (default runs/warmup_compare_<date>). A launch runs only the slice's episodes that no
# directory of that (config, scene, tag) holds yet, into a fresh directory: a crash costs the unfinished episodes, never
# a resume (the frozen GPU plan records free memory, so a resumed run refuses itself).
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$HERE/../../hm3d/run_benchmark.sh"
OUT="${OUT:-runs/warmup_compare_$(date -u +%Y%m%d)}"
EPISODES="$HERE/episodes_hm3d_v1_start_floor.txt"
FIRST=1 COUNT=10 TAG="" BY_SCENE=0 DEADLINE_EPOCH=""
CONFIGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --first) FIRST="$2"; shift 2 ;;
    --count) COUNT="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --by-scene) BY_SCENE=1; shift ;;
    --deadline)
      DEADLINE_EPOCH="$(date -d "$2" +%s)"
      [[ "$DEADLINE_EPOCH" -gt "$(date +%s)" ]] || DEADLINE_EPOCH=$((DEADLINE_EPOCH + 86400))
      shift 2 ;;
    *) CONFIGS+=("$1"); shift ;;
  esac
done
[[ ${#CONFIGS[@]} -gt 0 ]] || { echo "no configs given" >&2; exit 2; }

# A resident model takes VRAM, so the next launch's GPU plan would move YOLO to the CPU: every launch starts empty.
unload_models() { ollama ps 2>/dev/null | awk 'NR > 1 {print $1}' | xargs -r -n1 ollama stop || true; }

# The ids given that no directory of this target (any suffix) has recorded in an episodes.jsonl yet.
missing_ids() {
  local target="$1" recorded id; shift
  recorded="$(cat "$target"*/episodes.jsonl 2>/dev/null | grep -o '"episode_id": *"[^"]*"' | sed 's/.*"\([^"]*\)"$/\1/' || true)"
  for id in "$@"; do grep -qxF "$id" <<< "$recorded" || echo "$id"; done
}

launch() {
  local config="$1" scene="$2" target="$OUT/$1/$2$TAG" attempt dir
  local -a ids remaining
  if [[ -n "$DEADLINE_EPOCH" && "$(date +%s)" -ge "$DEADLINE_EPOCH" ]]; then
    echo "deadline passed: not starting $config/$scene$TAG"; return
  fi
  mapfile -t ids < <(grep "^$scene/" "$EPISODES" | tail -n +"$FIRST" | head -n "$COUNT")
  for attempt in 1 2 3; do
    mapfile -t remaining < <(missing_ids "$target" "${ids[@]}")
    [[ ${#remaining[@]} -gt 0 ]] || { echo "skip $config/$scene$TAG (finished)"; return; }
    dir="$target"
    [[ -e "$target/episodes.jsonl" || -e "$target/run.json" ]] && dir="${target}_r$(date +%H%M%S)"
    unload_models
    sleep 5
    "$LAUNCHER" --version v1 --scene "$scene" --output "$dir" \
      --policy-config "$HERE/configs/$config.json" --episode-ids "${remaining[@]}" && break
    echo "launch failed (attempt $attempt of 3): $config/$scene$TAG"; sleep 20
  done || true
}

mapfile -t SCENES < <(cut -d/ -f1 "$EPISODES" | uniq)
if [[ "$BY_SCENE" == 1 ]]; then
  for scene in "${SCENES[@]}"; do for config in "${CONFIGS[@]}"; do launch "$config" "$scene"; done; done
else
  for config in "${CONFIGS[@]}"; do for scene in "${SCENES[@]}"; do launch "$config" "$scene"; done; done
fi
