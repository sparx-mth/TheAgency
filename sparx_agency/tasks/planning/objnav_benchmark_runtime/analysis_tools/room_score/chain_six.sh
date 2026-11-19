#!/usr/bin/env bash
# After the episode-target prompts finish, ask all six objects for both prompts (2B).
until grep -q SCORE_DONE "$HOME/tc_sweep/tools/score_target_full.log" 2>/dev/null; do sleep 30; done
nice -n 10 /usr/bin/python3 -W ignore "$HOME/tc_sweep/tools/score_frames_six.py" "$HOME/models/objnav/qwen3vl2b_dl" 2b > "$HOME/tc_sweep/tools/score_six_2b.log" 2>&1
