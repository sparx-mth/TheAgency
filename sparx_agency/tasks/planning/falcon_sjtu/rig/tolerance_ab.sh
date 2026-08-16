#!/usr/bin/env bash
# ============================================================
# tolerance_ab.sh — A/B the unknown-visibility tolerance on the warehouse.
#
#   ./tolerance_ab.sh [legs_per_arm] [cap_s]
#
# Why an A/B rather than a campaign. `visib_unknown_tolerance` widens what
# countVisibleCells is willing to call a line of sight, so the honest worry is
# that it sends the aircraft at frontiers it cannot really see. One warehouse
# leg cannot answer that: this world's capsize rate is roughly one leg in ten
# even unpatched, and its contact counts have been measured at 213, 216 and 345
# on three consecutive legs of three different configurations. A single FATAL
# proves nothing in that noise.
#
# The warehouse is the right arena anyway. It is the cheap world -- a leg runs
# in 100-460 s against the hospital's 1700-2400 s -- and it is where the
# hazard lives, because its ClutteringC crates are hollow shells whose interior
# cavity can never be observed and therefore never stops being a frontier.
#
# Both arms fly the SAME binary. Only the rosparam changes, which is exactly
# what the parameter was added for: 0 restores upstream's "any UNKNOWN blocks
# the line of sight" and needs no rebuild to fall back to.
# ============================================================
set -uo pipefail

LEGS="${1:-3}"
CAP_S="${2:-600}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LAUNCH="${PKG_DIR}/adapter/launch/exploration.launch"
BASE="${TOLERANCE_AB_BASE:-/tmp/falcon_sjtu/tolerance_ab}"
mkdir -p "${BASE}"

export DISPLAY="${DISPLAY:-:1}"
export SJTU_PROJECT_DIR="${SJTU_PROJECT_DIR:-${HOME}/GIT/sjtu_project}"

say() { echo "[ab $(date +%H:%M:%S)] $*"; }

set_tolerance() {
    python3 - "$LAUNCH" "$1" <<'PY'
import re, sys
path, value = sys.argv[1], sys.argv[2]
src = open(path).read()
pattern = r'(<arg name="frontier_visib_unknown_tolerance" default=")[0-9]+(")'
new, n = re.subn(pattern, lambda m: m.group(1) + value + m.group(2), src)
if n != 1:
    raise SystemExit("expected 1 tolerance arg, found %d" % n)
open(path, "w").write(new)
PY
    # A malformed launch file is refused by roslaunch and the container exits at
    # startup, which reads as an infrastructure failure rather than a bad edit.
    python3 -c "import xml.dom.minidom,sys; xml.dom.minidom.parse('$LAUNCH')" || exit 1
}

RESULTS="${BASE}/results.tsv"
: > "${RESULTS}"

for arm in 2 0; do
    set_tolerance "${arm}"
    say "arm tolerance=${arm}: ${LEGS} warehouse leg(s), cap ${CAP_S}s"
    for i in $(seq 1 "${LEGS}"); do
        run_dir="${BASE}/tol${arm}_leg${i}"
        rm -rf "${run_dir}"
        WORLD=no_roof_small_warehouse FINISH_MIN_COVERAGE_M3=300 \
            bash "${SCRIPT_DIR}/campaign_run.sh" warehouse "${run_dir}" "${CAP_S}" \
            > "${run_dir}.console" 2>&1
        verdict="$(python3 - "${run_dir}/verdict.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("NO_VERDICT\t0\t0\t0"); raise SystemExit
print("%s\t%.2f\t%d\t%d" % (d.get("verdict", "?"), d.get("coverage_m3", 0),
                            d.get("contacts", 0), d.get("planner_respawns", 0)))
PY
)"
        printf 'tol%s\tleg%d\t%s\n' "${arm}" "${i}" "${verdict}" >> "${RESULTS}"
        say "  tol=${arm} leg ${i}: ${verdict}"
    done
done

say "==== A/B summary (verdict, coverage m3, contacts, respawns) ===="
column -t "${RESULTS}" 2>/dev/null || cat "${RESULTS}"
say "finished per arm:"
for arm in 2 0; do
    fin=$(grep -c "^tol${arm}.*FINISHED" "${RESULTS}" 2>/dev/null || echo 0)
    tot=$(grep -c "^tol${arm}" "${RESULTS}" 2>/dev/null || echo 0)
    fat=$(grep -c "^tol${arm}.*FATAL" "${RESULTS}" 2>/dev/null || echo 0)
    say "  tolerance=${arm}: ${fin}/${tot} finished, ${fat} fatal"
done
