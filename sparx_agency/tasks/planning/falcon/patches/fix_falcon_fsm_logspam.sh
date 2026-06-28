#!/bin/bash
# ============================================================
# fix_falcon_fsm_logspam.sh
#
# Silences FALCON's per-callback FSM debug spam that floods the
# console at the FSM tick rate (~90-130 Hz):
#
#   I.... exploration_fsm.cpp:58]  -------START OF CALLBACK-------
#   I.... exploration_fsm.cpp:73]  [FSM] Current state: INIT,
#                                        callback rate: 89.21Hz
#   I.... exploration_fsm.cpp:573] [FSM] Callback time: 0.0100922s
#   I.... exploration_fsm.cpp:574] -------END OF CALLBACK-------
#
# Root cause:
#   exploration_fsm.cpp emits four text LOG(INFO) lines PLUS a bare blank
#   separator on EVERY FSM callback (~90-130 Hz). The image sets
#   GLOG_logtostderr=1 (so all glog INFO goes to the console), which makes
#   these lines bury every other log. They are pure debug trace -- not state
#   changes -- so they add no operational value at steady state.
#
#   They cannot be quieted from the environment: GLOG_minloglevel is
#   global (it would also drop genuinely useful INFO from every other
#   FALCON node), and glog offers no per-file level. roslaunch can't
#   help either -- output="log" only redirects stdout while glog writes
#   to stderr. The only surgical fix is at the source.
#
# Fix:
#   Demote ONLY these statements from LOG(INFO) to VLOG(99). VLOG(N)
#   is compiled in but emitted only when the runtime verbosity >= N,
#   which is never the case by default -- so the lines vanish while the
#   code stays syntactically intact (works even if a statement wraps
#   across lines, since only the macro token is rewritten). Every other
#   FALCON log is untouched. To get them back, run the node with
#   GLOG_v=99.
#
# Matched by content ("START OF CALLBACK" and "callback rate"), not by
# line number, so it survives minor upstream reformatting.
#
# Idempotent + self-verifying. Run AFTER cloning FALCON, BEFORE
# catkin_make. Rebuilds nothing itself.
# ============================================================
set -euo pipefail

FSM="$(find /catkin_ws/src/FALCON -path '*exploration_manager*/exploration_fsm.cpp' | head -n1)"
if [ -z "${FSM}" ]; then
  echo "[fix_fsm_logspam] ERROR: exploration_fsm.cpp not found" >&2
  exit 1
fi
echo "[fix_fsm_logspam] Patching ${FSM}"

# Re-appliable: if a prior run left a backup, restore the clean original first
# so we always patch from scratch (lets an updated patch supersede an older one
# without a manual un-patch). The Docker build runs on a fresh clone, so the
# backup branch only matters for iterative in-container re-runs.
if [ -f "${FSM}.orig.bak" ]; then
  echo "[fix_fsm_logspam] Prior backup found -- restoring clean original first."
  cp "${FSM}.orig.bak" "${FSM}"
else
  cp "${FSM}" "${FSM}.orig.bak"
fi

# Rewrite LOG(INFO) -> VLOG(99) for the two per-callback statements only.
# perl slurp mode ((?s)) so a statement that wraps across lines is matched;
# (?:(?!;).)*? stays inside a single statement (stops before its terminating
# ';'), so we only touch the LOG(INFO) whose body contains the marker.
# The FSM callback emits four per-callback LOG(INFO) markers: "START OF
# CALLBACK", "callback rate", "Callback time", and "END OF CALLBACK".
# Demote each one.
for MARKER in "START OF CALLBACK" "callback rate" "Callback time" "END OF CALLBACK"; do
  perl -0777 -i -pe \
    "s/LOG\\(INFO\\)((?:(?!;).)*?\Q${MARKER}\E)/VLOG(99)\/*FALCON patch: FSM logspam*\/\$1/gs" \
    "${FSM}"
done

# Plus the bare blank-line SEPARATOR the callback prints every tick (the
# exploration_fsm.cpp:575-style empty record). It has no text to key on, so
# match an EMPTY LOG(INFO): the macro followed only by an optional `<< ""` /
# `<< endl` (and whitespace) up to its ';'. The (?:<<\s*(?:""|...))? group is
# optional but the alternatives are all empty-output, so a real `LOG(INFO) <<
# "text"` can never match (its body is neither empty nor a bare separator).
# Best-effort: not required by the verify below, so a differently-written
# separator can't revert the four guaranteed rewrites.
perl -0777 -i -pe \
  's/LOG\(INFO\)(\s*(?:<<\s*(?:""|(?:std::)?endl|"\\n"))?\s*);/VLOG(99)\/*FALCON patch: FSM logspam (separator)*\/$1;/g' \
  "${FSM}"

# Verify: all four statements were rewritten, and the patch did not disturb the
# code structure. The substitution only swaps a macro token and inserts a
# comment, so brace counts MUST be byte-for-byte unchanged from the backup. Do
# NOT compare {-count to }-count directly: this file's RAW counts are naturally
# unequal (braces appear inside string literals/comments), so that test
# false-fails. Compare against the pre-patch backup instead.
N=$(grep -c "FALCON patch: FSM logspam" "${FSM}" || true)
OB=$(grep -o '{' "${FSM}" | wc -l);          CB=$(grep -o '}' "${FSM}" | wc -l)
OB0=$(grep -o '{' "${FSM}.orig.bak" | wc -l); CB0=$(grep -o '}' "${FSM}.orig.bak" | wc -l)
echo "[fix_fsm_logspam] statements demoted: ${N}   braces now: {${OB} }${CB}  (orig {${OB0} }${CB0})"
if [ "${N}" -lt 4 ]; then
  echo "[fix_fsm_logspam] ERROR: expected 4 rewrites, got ${N}. Upstream code may have changed." >&2
  mv "${FSM}.orig.bak" "${FSM}"
  exit 1
fi
if [ "${OB}" != "${OB0}" ] || [ "${CB}" != "${CB0}" ]; then
  echo "[fix_fsm_logspam] ERROR: brace count changed vs original; reverting." >&2
  mv "${FSM}.orig.bak" "${FSM}"
  exit 1
fi
echo "[fix_fsm_logspam] OK: per-callback FSM spam silenced (run with GLOG_v=99 to restore)."
