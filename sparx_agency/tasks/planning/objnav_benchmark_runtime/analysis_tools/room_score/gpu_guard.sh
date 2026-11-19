#!/usr/bin/env bash
# Stops the (resumable) frame-scoring job if available memory is low or the GPU is hot.
while true; do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1); temp=${temp:-0}
  if [ "$avail" -lt 1800 ] || [ "$temp" -ge 83 ]; then
    if pgrep -f '[s]core_frames_' >/dev/null; then pgrep -f '[s]core_frames_' | xargs -r kill; echo "$(date +%T) STOPPED job: avail=${avail}MB gpu=${temp}C"; fi
  fi
  sleep 3
done
