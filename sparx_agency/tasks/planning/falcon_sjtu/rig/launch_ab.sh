#!/usr/bin/env bash
# ============================================================
# launch_ab.sh — A/B one exploration.launch <arg> default, same binary.
#
#   ./launch_ab.sh <map> <launch_arg> <treatment> <control> [legs] [cap_s]
#
#   ./launch_ab.sh warehouse frontier_viewpoint_sampling_3d true false 3 900
#
# The sibling of world_ab.sh, which does the same job for a map-config scalar.
# Use this one when the knob under test is a launch argument -- a rosparam the
# planner or the follower reads -- rather than world geometry.
#
# Why both arms, every time. Nothing in this stack can be judged from one leg.
# The hospital has produced 1-in-6 and 4-in-6 finishes on IDENTICAL code; the
# warehouse's contact counts ran 937, 34 and 251 across three legs of ONE
# configuration, and its capsize rate is roughly one leg in ten. A single
# before/after is indistinguishable from that noise, and this package has more
# than once published a result that later evaporated.
#
# Runs headless: campaign_run.sh defaults RVIZ=0 and FOLLOW=0.
# ============================================================
set -uo pipefail

MAP="${1:?map name (config/<map>.yaml)}"
KEY="${2:?launch arg name, e.g. frontier_viewpoint_sampling_3d}"
TREATMENT="${3:?treatment value}"
CONTROL="${4:?control value}"
LEGS="${5:-3}"
CAP_S="${6:-900}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LAUNCH="${PKG_DIR}/adapter/launch/exploration.launch"
[ -f "${LAUNCH}" ] || { echo "no launch file: ${LAUNCH}"; exit 1; }

BASE="${LAUNCH_AB_BASE:-/tmp/falcon_sjtu/launch_ab_${MAP}_${KEY}}"
mkdir -p "${BASE}"
RESULTS="${BASE}/results.tsv"
: > "${RESULTS}"

export DISPLAY="${DISPLAY:-:1}"
export SJTU_PROJECT_DIR="${SJTU_PROJECT_DIR:-${HOME}/GIT/sjtu_project}"
case "${MAP}" in
    warehouse) WORLD_NAME="no_roof_small_warehouse"; FLOOR=300 ;;
    hospital)  WORLD_NAME="hospital";                FLOOR=600 ;;
    *)         WORLD_NAME="${MAP}";                  FLOOR=0 ;;
esac

SETTER="${BASE}/set_arg.py"
cat > "${SETTER}" <<'SETTER_EOF'
import re, sys, xml.dom.minidom
path, key, value = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()
pattern = r'(<arg name="%s" default=")[^"]*(")' % re.escape(key)
new, n = re.subn(pattern, lambda m: m.group(1) + value + m.group(2), src)
if n != 1:
    raise SystemExit("expected exactly 1 <arg> named %s, found %d" % (key, n))
open(path, "w").write(new)
# roslaunch refuses a malformed file and the container then exits at startup,
# which reads as an infrastructure failure rather than a bad edit.
xml.dom.minidom.parse(path)
SETTER_EOF

say() { echo "[ab $(date +%H:%M:%S)] $*"; }

READER="${BASE}/read_verdict.py"
cat > "${READER}" <<'READER_EOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    print("NO_VERDICT\t0\t0\t0\t0\t0"); raise SystemExit
print("%s\t%.2f\t%d\t%d\t%d\t%d" % (d.get("verdict", "?"), d.get("coverage_m3", 0),
                                    d.get("contacts", 0), d.get("planner_respawns", 0),
                                    d.get("elapsed_s", 0), d.get("contact_objects", 0)))
READER_EOF

for arm in "${TREATMENT}" "${CONTROL}"; do
    python3 "${SETTER}" "${LAUNCH}" "${KEY}" "${arm}" || { say "FAILED to set ${KEY}=${arm}"; exit 1; }
    label=$([ "${arm}" = "${TREATMENT}" ] && echo treat || echo ctrl)
    say "arm ${label}: ${KEY}=${arm}, ${LEGS} ${MAP} leg(s), cap ${CAP_S}s"
    for i in $(seq 1 "${LEGS}"); do
        run_dir="${BASE}/${label}_${arm}_leg${i}"
        rm -rf "${run_dir}"
        WORLD="${WORLD_NAME}" FINISH_MIN_COVERAGE_M3="${FLOOR}" \
            bash "${SCRIPT_DIR}/campaign_run.sh" "${MAP}" "${run_dir}" "${CAP_S}" \
            > "${run_dir}.console" 2>&1
        row="$(python3 "${READER}" "${run_dir}/verdict.json")"
        printf '%s\t%s\tleg%d\t%s\n' "${label}" "${arm}" "${i}" "${row}" >> "${RESULTS}"
        say "  ${label} ${KEY}=${arm} leg ${i}: ${row}"
    done
done

# Leave the treatment in place: it is what the operator was testing, and it is
# the value `git diff` will show.
python3 "${SETTER}" "${LAUNCH}" "${KEY}" "${TREATMENT}"

say "==== ${MAP} launch ${KEY}: ${TREATMENT} (treat) vs ${CONTROL} (ctrl) ===="
say "arm value leg verdict coverage contacts respawns elapsed"
column -t "${RESULTS}" 2>/dev/null || cat "${RESULTS}"

SUMMARY="${BASE}/summarise.py"
cat > "${SUMMARY}" <<'SUMMARY_EOF'
import sys
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
    print("[ab] %-5s %s=%-6s finished %d/%d  fatal %d  OBJECTS %s (mean %.1f)  "
          "reports %s  coverage mean %.1f m3  elapsed mean %.0f s"
          % (arm, "value", r[0][1], fin, len(r), fatal, "/".join(map(str, obj)),
             sum(obj) / len(obj), "/".join(map(str, con)),
             sum(cov) / len(cov), sum(el) / len(el)))
print("[ab] OBJECTS TOUCHED and finish rate are the safety verdict, not raw\n"
      "     reports: Gazebo emits one per contact point per physics step, so a\n"
      "     single IV stand grazed repeatedly has read as 186 reports on 1 object.\n"
      "[ab] also: watch elapsed too, "
      "since a sampling change can cost planner CPU without showing in coverage.")
SUMMARY_EOF
python3 "${SUMMARY}" "${RESULTS}"
