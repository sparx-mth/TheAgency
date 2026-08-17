#!/usr/bin/env bash
# ============================================================
# world_ab.sh — A/B one map-config scalar, same binary, both arms.
#
#   ./world_ab.sh <map> <yaml_key> <treatment> <control> [legs] [cap_s]
#
#   ./world_ab.sh warehouse box_max_z 2.7 2.4 3 900
#
# Why this exists. Nothing in this stack can be judged from one leg. The
# hospital has produced 1-in-6 and 4-in-6 finishes on IDENTICAL code; the
# warehouse's contact counts ran 213, 216 and 345 across three consecutive legs
# of three different configurations, and its capsize rate is roughly one leg in
# ten even unpatched. A single before/after is indistinguishable from that
# noise, and this package has twice published a result that later evaporated.
#
# So every configuration change gets flown against its own control, in the same
# session, on the same image, alternating nothing but the one value under test.
# The script edits the map config in place, flies the arm, restores it, and
# flies the control -- so a crash mid-campaign leaves the repository holding
# the TREATMENT value, which is the safe direction to fail (it is the value the
# operator was testing and it is visible in `git diff`).
#
# Runs headless: campaign_run.sh defaults RVIZ=0 and FOLLOW=0.
# ============================================================
set -uo pipefail

MAP="${1:?map name (config/<map>.yaml)}"
KEY="${2:?yaml key, e.g. box_max_z}"
TREATMENT="${3:?treatment value}"
CONTROL="${4:?control value}"
LEGS="${5:-3}"
CAP_S="${6:-900}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CFG="${PKG_DIR}/config/${MAP}.yaml"
[ -f "${CFG}" ] || { echo "no such map config: ${CFG}"; exit 1; }

BASE="${WORLD_AB_BASE:-/tmp/falcon_sjtu/world_ab_${MAP}_${KEY}}"
mkdir -p "${BASE}"
RESULTS="${BASE}/results.tsv"
: > "${RESULTS}"

export DISPLAY="${DISPLAY:-:1}"
export SJTU_PROJECT_DIR="${SJTU_PROJECT_DIR:-${HOME}/GIT/sjtu_project}"
# The Gazebo world file and the FALCON map config are named differently for the
# warehouse, which campaign_run.sh resolves through WORLD=.
case "${MAP}" in
    warehouse) WORLD_NAME="no_roof_small_warehouse" ;;
    *)         WORLD_NAME="${MAP}" ;;
esac
# What this world affords, so a trivially small map cannot pass as a finish.
case "${MAP}" in
    warehouse) FLOOR=300 ;;
    hospital)  FLOOR=600 ;;
    *)         FLOOR=0 ;;
esac

say() { echo "[ab $(date +%H:%M:%S)] $*"; }

set_key() {
    python3 - "${CFG}" "${KEY}" "$1" <<'PY'
import re, sys
path, key, value = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()
# Only the assignment line, never the prose above it: these configs carry long
# comment blocks that mention the same key and its previous values.
pattern = r'^(\s*%s:\s*)(-?[0-9.]+)(\s*)$' % re.escape(key)
new, n = re.subn(pattern, lambda m: m.group(1) + value + m.group(3), src, flags=re.M)
if n != 1:
    raise SystemExit("expected exactly 1 assignment of %s, found %d" % (key, n))
open(path, "w").write(new)
PY
}

for arm in "${TREATMENT}" "${CONTROL}"; do
    set_key "${arm}" || { say "FAILED to set ${KEY}=${arm}"; exit 1; }
    label=$([ "${arm}" = "${TREATMENT}" ] && echo treat || echo ctrl)
    say "arm ${label}: ${KEY}=${arm}, ${LEGS} ${MAP} leg(s), cap ${CAP_S}s"
    for i in $(seq 1 "${LEGS}"); do
        run_dir="${BASE}/${label}_${arm}_leg${i}"
        rm -rf "${run_dir}"
        WORLD="${WORLD_NAME}" FINISH_MIN_COVERAGE_M3="${FLOOR}" \
            bash "${SCRIPT_DIR}/campaign_run.sh" "${MAP}" "${run_dir}" "${CAP_S}" \
            > "${run_dir}.console" 2>&1
        row="$(python3 - "${run_dir}/verdict.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("NO_VERDICT\t0\t0\t0\t0\t0"); raise SystemExit
print("%s\t%.2f\t%d\t%d\t%d\t%d" % (d.get("verdict", "?"), d.get("coverage_m3", 0),
                                    d.get("contacts", 0), d.get("planner_respawns", 0),
                                    d.get("elapsed_s", 0), d.get("contact_objects", 0)))
PY
)"
        printf '%s\t%s\tleg%d\t%s\n' "${label}" "${arm}" "${i}" "${row}" >> "${RESULTS}"
        say "  ${label} ${KEY}=${arm} leg ${i}: ${row}"
    done
done

# Leave the treatment in place: it is what the operator was testing, and it is
# the value `git diff` will show.
set_key "${TREATMENT}"

say "==== ${MAP} ${KEY}: ${TREATMENT} (treat) vs ${CONTROL} (ctrl) ===="
say "arm value leg verdict coverage contacts respawns elapsed"
column -t "${RESULTS}" 2>/dev/null || cat "${RESULTS}"
python3 - "${RESULTS}" <<'PY'
import collections, sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
for arm in ("treat", "ctrl"):
    r = [x for x in rows if x[0] == arm]
    if not r:
        continue
    fin = sum(1 for x in r if "FINISHED" in x[3])
    fatal = sum(1 for x in r if "FATAL" in x[3])
    con = [int(x[5]) for x in r]
    obj = [int(x[8]) if len(x) > 8 else 0 for x in r]
    cov = [float(x[4]) for x in r]
    el = [int(x[7]) for x in r]
    print("[ab] %-5s %s=%-5s  finished %d/%d  fatal %d  OBJECTS %s (mean %.1f)  "
          "reports %s  coverage mean %.1f m3  elapsed mean %.0f s"
          % (arm, "value", r[0][1], fin, len(r), fatal, "/".join(map(str, obj)),
             sum(obj) / len(obj), "/".join(map(str, con)),
             sum(cov) / len(cov), sum(el) / len(el)))
print("[ab] OBJECTS TOUCHED and finish rate are the safety verdict, not raw\n"
      "     reports: Gazebo emits one per contact point per physics step, so a\n"
      "     single IV stand grazed repeatedly has read as 186 reports on 1 object.\n"
      "[ab] also: coverage is NOT "
      "comparable across arms when the tested key changes the box volume.")
PY
