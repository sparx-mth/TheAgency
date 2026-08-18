#!/usr/bin/env bash
# supervisor.sh -- the outer loop that must not die.
#
# Runs campaign cycles back to back, forever. Deliberately dumb: all the
# intelligence lives in campaign.py and in the agent that reads the run folders.
# This script's only jobs are to keep flying, to never spin hot, and to refuse
# to fly code that does not compile.
#
# Two sentinels, both under runs/:
#   PAUSE  finish the current cycle, then wait (touch it before editing code)
#   STOP   exit after the current cycle
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
DURATION="${CAMPAIGN_DURATION:-600}"

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
    say "PAUSE sentinel present -- holding (remove $RUNS/PAUSE to resume)"
    sleep 30
    continue
  fi

  bad=""
  for m in "${GUARDED_MODULES[@]}"; do
    if [[ -f "$m" ]] && ! "$PY" -m py_compile "$m" 2>/dev/null; then
      bad="$m"
      break
    fi
  done
  if [[ -n "$bad" ]]; then
    say "REFUSING TO FLY: $bad does not compile. Waiting for it to be fixed."
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
