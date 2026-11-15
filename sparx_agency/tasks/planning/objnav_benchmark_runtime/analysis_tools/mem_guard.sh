#!/usr/bin/env bash
# Stops the (resumable) VLM job when available memory gets low, so the laptop does not freeze.
while true; do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  if [ "$avail" -lt 1800 ]; then
    pgrep -f '[v]lm_box_check.py' | xargs -r kill
    echo "$(date +%T) STOPPED job, avail=${avail}MB"
  fi
  sleep 3
done
