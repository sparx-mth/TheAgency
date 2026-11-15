# 025 - LLM-first node oracle, end-to-end review, the one-click Gibson benchmark

**Branch:** `feat/objnav-habitat-gibson-nadav`
**Status:** done -- code, 1641 regressions in the objnav suites (1587 before), five live-model probe
scenarios passing on `qwen2.5:14b-instruct`, two smoke episodes through the one-click script, docs
**Roadmap item:** ObjectNav exploration efficiency (010) and robustness (012); follows 024

## The ask
Four things before a long Gibson validation run: (1) rely on the 14B model rather than on default
values now that the benchmark is single-storey -- brief it on the unvisited rooms from the rooms
already identified, on typical object placement, and on how far into the 500-action budget the search
is, so the first pass covers the house and a second pass re-checks the likely rooms; (2) review the
codebase end to end; (3) verify with a few episodes; (4) a one-click script for the whole 1,000-episode
split with lean records and a readable README; (5) a way to follow the run's progress.

## What changed

### The oracle is LLM-first (`core/mapping/topology/search_node_oracle.py`)
- Every room of the storey is a node line, finished and type-excluded ones included, with
  `status=never_entered|entered|scanned(<how>, <ago>)`; the user prompt carries a HOUSE line
  (`methods/house_context.py`): room types found with their status, rooms unidentified, openings
  unlooked, whether unexplored frontier is still reachable, `actions used about N of 500` (25-action
  steps keep the prompt-reuse working).
- The system prompt asks for three written steps before the numbers -- `home`, `house`, `stage` --
  and a `pass` verdict. The loop reads `pass`: finished rooms are withheld from the solver while it is
  `first` and offered again under `second` (or when the floor has no unfinished room, no exit and no
  reachable frontier left: `second_pass_forced`); a revisit scans from a spot 1.5 m from the earlier
  scan points (`room_vantage.vantage_point(avoid=...)`), is never released as finished on the way in,
  and the model is re-asked every budget bucket while revisits are offered.
- Defaults: `type_prior` off, `unexplored_floor`/`unexplored_elsewhere`/`home_floor` 0 -- the model is
  given the facts those encoded. Under scan, no relabel ends a visit unless the type prior is on.
  The storey/stairs rules are a `STAIRS_SUPPLEMENT` shown only when a staircase is a node.
- A loop point with nothing offerable spends no reasoning call (`oracle_calls_skipped`), in
  `reconsider()` too (the review found one 14B call per semantic-signature change while nothing was a
  node).

### The review (three read-only reviewers, each ~1 h) and what it found
- **Run-killer.** Published starts lie up to 0.28 m below the navmesh and habitat-sim settles the agent
  on its first TRANSLATION; the evaluated storey itself steps up to 0.50 m on one stride. The kinematic
  contract raised `EnvContractError` out of `run_benchmark`, which ended the run and replayed on resume.
  Fixed: `KinematicTolerance.settle_m` (0.30, additive, first moving action) and `climb_m` 0.60 for
  Gibson; an off-map agent position reads as the unreachable sentinel; `merge_runs` refuses diagnostic
  shards.
- **Closing cycles.** Near candidates bypassed every re-takeover guard (twelve takeovers of one chair
  from one spot); the first smoke episode spent 217 of 417 actions in VERIFY with 18 releases and 233
  turn reversals. Fixed: the rejection memory records where the agent stood and refuses the same near
  candidate from the same spot; a suspect may step inside the terminal range for its second viewpoint;
  a locked target without a path is kept in frame instead of the idle hold the headless agent turns
  on; a detector back-off frame is no evidence (clocks paused, chain carried).
- **Loop.** A SEARCH-state `route_failed` was dropped (36 actions of blind fallback with the room in
  force); a completed peek re-created the same doorway as a new opening (the first smoke episode spent
  132 actions in six peeks); the vantage EDT ran on the full 800x800 grid per room per selection; the
  map-edge `ValueError` stood outside the fallback guard; the half-level landing idle loop (Klickitat)
  is bounded by adopting a settled plateau 0.5-1.5 m off the spawn plane after 30 actions.
- **Services.** A malformed detector reply for one frame is a frame failure, not the end of the run; the
  per-frame `/health` GET is gone; an LLM timeout is never retried (20 minutes per hung call before).

### The one-click benchmark (`gibson/run_benchmark.sh`, `gibson/BENCHMARK.md`, `gibson/progress.py`)
Services up (Ollama docker/native/external; the CPU detector), preflight, `gibson.run --lean` over the
1,000 published episodes, `progress.json` after every episode (percent, ETA, episode in progress,
running SR/SPL/SoftSPL/DTG overall, per scene, per category), a console bar, `--status DIR`, resume
into an existing `--output`, services down. Lean rows are ~10 KB (102 KB before).

## Verification
- `probe_node_oracle.py` against the live 14B: toilet with no bathroom found -> first pass, the small
  unidentified room 0.55 and the unlooked doorway 0.35, the never-entered bedroom 0.25 (an en-suite),
  scanned kitchen/living room 0.01; sofa with the living room scanned -> unidentified rooms 0.30/0.10,
  the scanned living room 0.50; television -> bedroom 0.10 above kitchen 0.02 and bathroom 0.02; late
  toilet with everything scanned -> `pass: second`, bathroom 0.35 > bedroom 0.15 > kitchen 0.05;
  chair early -> first pass. 26-34 s per call (127 s with the model load).
- Smoke episodes via the script, Darden/000000 (bed), before and after the review fixes -- see the
  table in the summary of this session's work; the first run alone exposed the closing cycles.

## Open
- The 14B on a 32-thread CPU costs 30-90 s per oracle call and the policy ~0.4 s per action; a full
  split is days of wall time. `LLM_REASONING_MODEL=qwen2.5:7b-instruct` or a GPU-hosted model is the
  lever; sharding across machines is supported.
- The review's remaining LOW items (association against the first anchor after a lock; VERIFY's
  two-step `follow` bypassing A*; landmark peeks retired by id) are recorded in the reviewers' reports
  and not addressed here.
