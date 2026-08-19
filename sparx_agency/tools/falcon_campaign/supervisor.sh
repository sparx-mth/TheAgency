#!/usr/bin/env bash
# supervisor.sh -- the outer loop that must not die.
#
# Runs campaign cycles back to back, forever. Deliberately dumb: all the
# intelligence lives in campaign.py and in the agent that reads the run folders.
# This script's only jobs are to keep flying, to never spin hot, and to refuse
# to fly code that does not compile.
#
# Two sentinels, both under runs/:
#   PAUSE  finish the current cycle, then wait (touch it before editing code).
#          SELF-EXPIRING: past CAMPAIGN_PAUSE_MAX_AGE_S (default 30 min) it is
#          treated as forgotten and removed, because a pause protects minutes of
#          work and one left behind cost this campaign 13.5 hours of flying.
#   STOP   exit after the current cycle (never expires -- stopping is intent)
#
# Start it:
#   nohup bash sparx_agency/tools/falcon_campaign/supervisor.sh \
#       > runs/supervisor.log 2>&1 & disown
#
# Survive a reboot: see install_reboot_hook() at the bottom, or run
#   bash sparx_agency/tools/falcon_campaign/supervisor.sh --install-reboot-hook

set -uo pipefail

REPO="${SPARX_REPO:-/home/user1/GIT/TheAgency}"
RUNS="$REPO/runs"
PY="${CAMPAIGN_PYTHON:-python3}"
# The window comes from config.py, which sets it to what the battery actually
# supports; a hardcoded 600 here quietly overrode that and every run flew ~170 s
# on a dying pack, which is recorded as flight and is not.
DURATION="${CAMPAIGN_DURATION:-$(cd "$REPO" && PYTHONPATH="$REPO" "$PY" -c 'from sparx_agency.tools.falcon_campaign import config; print(config.FLIGHT_SECONDS)' 2>/dev/null || echo 430)}"
# A pause protects an edit or a manual flight -- both minutes of work. Left
# behind it silently idles the whole campaign: one forgotten sentinel cost 13.5
# hours of flying while this loop logged "holding" every 30s and did exactly what
# it was told. Past this age the pause is treated as forgotten, not as intent.
PAUSE_MAX_AGE_S="${CAMPAIGN_PAUSE_MAX_AGE_S:-1800}"

# Modules that must compile before any flight. A syntax error in one of these
# would otherwise burn a whole 10-minute cycle to discover.
GUARDED_MODULES=(
  "sparx_agency/tools/falcon_campaign/config.py"
  "sparx_agency/tools/falcon_campaign/bringup.py"
  "sparx_agency/tools/falcon_campaign/campaign.py"
  "sparx_agency/tools/falcon_campaign/recorder.py"
  "sparx_agency/tools/falcon_campaign/analyze.py"
  "sparx_agency/robots/ROBOTICAN/adapters/rooster_twist_control_adapter.py"
  "sparx_agency/robots/ROBOTICAN/adapters/rooster_command_unit.py"
  "sparx_agency/robots/ROBOTICAN/helpers/rooster_unit.py"
  "sparx_agency/robots/ROBOTICAN/rooster_ground_truth_localization.py"
  "sparx_agency/core/control/axis_velocity_servo.py"
)

say() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

install_reboot_hook() {
  local line="@reboot sleep 120 && cd $REPO && nohup bash sparx_agency/tools/falcon_campaign/supervisor.sh >> $RUNS/supervisor.log 2>&1 &"
  if crontab -l 2>/dev/null | grep -Fq "falcon_campaign/supervisor.sh"; then
    say "reboot hook already installed"
    return 0
  fi
  { crontab -l 2>/dev/null; echo "$line"; } | crontab - \
    && say "reboot hook installed" \
    || say "could not install reboot hook (crontab unavailable)"
}

if [[ "${1:-}" == "--install-reboot-hook" ]]; then
  mkdir -p "$RUNS"; install_reboot_hook; exit 0
fi

mkdir -p "$RUNS"
cd "$REPO" || { say "FATAL: $REPO is gone"; exit 1; }

# One supervisor only. A second would fly the same drone from two processes,
# which is the failure mode this whole stack has been bitten by most often.
LOCK="$RUNS/supervisor.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  say "another supervisor already holds $LOCK -- exiting"
  exit 0
fi
echo $$ >&9

say "supervisor up (pid $$, duration ${DURATION}s, repo $REPO)"
install_reboot_hook

CYCLE=0
CONSECUTIVE_FAILURES=0

while true; do
  if [[ -f "$RUNS/STOP" ]]; then
    say "STOP sentinel present -- exiting"
    exit 0
  fi

  if [[ -f "$RUNS/PAUSE" ]]; then
    pause_age=$(( $(date +%s) - $(stat -c %Y "$RUNS/PAUSE" 2>/dev/null || date +%s) ))
    if (( pause_age > PAUSE_MAX_AGE_S )); then
      say "PAUSE has been held ${pause_age}s (> ${PAUSE_MAX_AGE_S}s) -- treating it as"
      say "forgotten and RESUMING. Touch it again if the hold was still wanted."
      rm -f "$RUNS/PAUSE"
    else
      # Only narrate the wait occasionally; this used to emit 1600 identical lines.
      if (( pause_age % 300 < 31 )); then
        say "PAUSE sentinel present ${pause_age}s -- holding (auto-resume at ${PAUSE_MAX_AGE_S}s)"
      fi
      sleep 30
      continue
    fi
  fi

  bad=""
  for m in "${GUARDED_MODULES[@]}"; do
    if [[ -f "$m" ]] && ! "$PY" -m py_compile "$m" 2>/dev/null; then
      bad="$m"
      break
    fi
  done
  # Launch files too: roslaunch rejects the whole file on one XML slip (a `--`
  # inside an XML comment is illegal, and that alone cost one cycle), and it
  # fails minutes into a bring-up rather than up front.
  if [[ -z "$bad" ]]; then
    bad=$("$PY" - <<'EOF'
import glob, xml.etree.ElementTree as ET
for f in sorted(glob.glob("sparx_agency/tasks/planning/falcon/adapter/launch/*.launch")):
    try:
        ET.parse(f)
    except ET.ParseError as exc:
        print("%s (%s)" % (f, exc))
        break
EOF
)
  fi
  # An arg the entry launch file never declared is accepted on the command line
  # and then silently dropped -- roslaunch does not complain. That cost three
  # cycles once; catching it here costs nothing.
  if [[ -z "$bad" ]]; then
    bad=$(PYTHONPATH="$REPO" "$PY" - <<'EOF'
import re
from sparx_agency.tools.falcon_campaign import config as C
entry = ("sparx_agency/tasks/planning/falcon/adapter/launch/sphera_drone.launch")
declared = set(re.findall(r'<arg\s+name="([^"]+)"', open(entry).read()))
passed = set(re.findall(r'([A-Za-z_][A-Za-z0-9_]*):=', C.adapter_launch_cmd()))
missing = sorted(passed - declared)
if missing:
    print("%s does not declare: %s" % (entry, ", ".join(missing)))
EOF
)
  fi
  if [[ -n "$bad" ]]; then
    say "REFUSING TO FLY: $bad is invalid. Waiting for it to be fixed."
    sleep 60
    continue
  fi

  CYCLE=$((CYCLE + 1))
  say "--- cycle $CYCLE starting ---"
  if PYTHONPATH="$REPO" "$PY" -m sparx_agency.tools.falcon_campaign.campaign \
       --duration "$DURATION"; then
    say "--- cycle $CYCLE completed ---"
    CONSECUTIVE_FAILURES=0
  else
    CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
    say "--- cycle $CYCLE ended unhappily (streak $CONSECUTIVE_FAILURES) ---"
  fi

  # Back off on a losing streak so a systematically broken stack does not burn
  # hours of Sphera restarts, but never back off so far that the loop looks dead.
  if (( CONSECUTIVE_FAILURES >= 3 )); then
    backoff=$(( 60 * CONSECUTIVE_FAILURES ))
    (( backoff > 600 )) && backoff=600
    say "backing off ${backoff}s after $CONSECUTIVE_FAILURES failures"
    sleep "$backoff"
  else
    sleep 20
  fi
done
