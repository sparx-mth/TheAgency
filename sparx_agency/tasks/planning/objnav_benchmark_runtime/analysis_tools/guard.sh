#!/usr/bin/env bash
# Pauses the Habitat run (SIGSTOP) while the laptop is under pressure, resumes it when calm. Never kills anything.
paused=0
while true; do
  avail=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  load=$(awk '{print int($1)}' /proc/loadavg)
  temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  temp=${temp:-0}
  if [ "$paused" = 0 ] && { [ "$avail" -lt 1500 ] || [ "$load" -ge 14 ] || [ "$temp" -ge 80 ]; }; then
    pgrep -f '[t]race_run.py' | xargs -r kill -STOP; paused=1; echo "$(date +%T) PAUSE avail=${avail}MB load=$load gpu=${temp}C"
  elif [ "$paused" = 1 ] && [ "$avail" -ge 3000 ] && [ "$load" -le 8 ] && [ "$temp" -le 70 ]; then
    pgrep -f '[t]race_run.py' | xargs -r kill -CONT; paused=0; echo "$(date +%T) RESUME avail=${avail}MB load=$load gpu=${temp}C"
  fi
  sleep 5
done
