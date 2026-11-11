#!/usr/bin/env bash
# The local-exploration comparison (experiment 2), one arm per command; `without` is shared with the warm-up study:
#   mp3d/explore_compare.sh explore_viewpoint --dry-run
#   mp3d/explore_compare.sh explore_viewpoint                # the explore visit by viewpoints (replays the answers)
#   mp3d/explore_compare.sh explore_viewpoint_transparent    # ... with MapEx's transparent unknown (no survival prior)
#   mp3d/explore_compare.sh report --explain
# Every option of look_study.sh applies (--jobs N, --resume, the environment).
exec "$(dirname -- "${BASH_SOURCE[0]}")/look_study.sh" explore "$@"
