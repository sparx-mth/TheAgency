#!/usr/bin/env bash
# The warm-up comparison (experiment 1), one arm per command:
#   mp3d/warmup_compare.sh without --dry-run          # the plan and the checks, nothing started
#   mp3d/warmup_compare.sh without                    # the ten-turn spin; records every LLM answer
#   mp3d/warmup_compare.sh p1_b25                     # one vantage point, budget 25 (replays the answers)
#   mp3d/warmup_compare.sh p3_b25 | p1_b15 | p3_b40 | p0_b15
#   mp3d/warmup_compare.sh report --explain           # the table, and why each changed episode changed
# Every option of look_study.sh applies (--jobs N, --resume, the environment).
exec "$(dirname -- "${BASH_SOURCE[0]}")/look_study.sh" warmup "$@"
