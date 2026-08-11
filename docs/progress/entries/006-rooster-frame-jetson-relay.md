# 006 - Rooster frame capture → direct relay to the Jetson

**Branch:** `feat/rooster-frame-jetson-relay`
**Status:** done
**Roadmap item:** "Test Rooster (Sphera or real drone) frame capture → direct relay to the Jetson, orchestrated via mission_control.py"

## Goal
Prove that frames captured from Rooster (via Sphera sim, or the real ROBOTICAN drone) can be
sent directly to the Jetson (AGX1, `user@192.0.0.89`), and that this relay can be started/
stopped/monitored through `mission_control.py` (the orchestrator) like every other service in
this pipeline — not just as a manually-run one-off script.

## Why
ROBOTICAN's whole vision stack currently runs on the host PC (frame capture + DA3-TRT depth
both local — see the "Vision (host, domain 9)" comment block in `mission_control.py`), unlike
XTEND, which is fully Jetson-hosted. `sparx_agency/robots/common/dir_push_relay.py` +
`dir_watch_path_publisher.py` already exist as a generic host→Jetson file relay (built for
exactly this kind of split, per `dir_push_relay.py`'s own docstring) but are not wired into
`mission_control.py` and have never been tested for the Rooster/ROBOTICAN case — only the
mechanism itself exists, unproven end-to-end for this pipeline.

## Steps
- [x] Confirm SSH connectivity + a real destination directory on the Jetson to push into
      (SSH reachable, host `user-agx1`, Python 3.10.12; created
      `{JETSON_DATA}/rooster_frames_relay`)
- [x] Synthetic test first (no Sphera/drone needed): touched a dummy `.jpg` in
      `/tmp/rooster_frames` (the same dir `rooster_frame_dir_publisher.py` already writes,
      bind-mounted host↔`robotican_dev` per `docker-compose.robotican.yml`), ran
      `dir_push_relay.py` on the host pointed at the Jetson, ran `dir_watch_path_publisher.py`
      on the Jetson — file arrived and its path was published on
      `/rooster/jetson_rgb_frame_path` (`ROS_DOMAIN_ID=5`) in the expected
      `{path} {sec} {nanosec}` format
- [x] Add two new `Service` entries to `mission_control.py` (`ROBOTICAN_SERVICES`, new
      `rooster_jetson` group): the PC-side relay (`rooster_frame_relay_jetson_R1`) and the
      Jetson-side watcher (`rooster_jetson_frame_watch_R1`), following the existing `Service`
      dataclass conventions
- [x] Re-ran the test through `mission_control.py`'s own `start_service`/`stop_service`/
      `get_all_states` (not ad-hoc shell commands): states False→True on start, a real frame
      flowed through end-to-end, states True→False on stop, no stray processes left on either
      machine afterward
- [ ] Repeat with real flowing frames (Rooster Frame Capture actually producing JPEGs from a
      live Sphera flight) instead of a synthetic touch-test — deferred; requires Sphera's `R1`
      up, which per the `fly-rooster-sphera` skill the user starts, not this agent. The relay
      mechanism itself doesn't care what wrote the file, so this is a lower-value follow-up
      rather than a blocker.

## Open questions
- Exact Jetson-side destination directory to push into (proposing something under
  `{JETSON_DATA}/rooster_frames_relay`, to keep it clearly separate from XTEND's own capture
  dirs) — confirm with the user or just pick one and document it here.
- What (if anything) should consume the republished path topic on the Jetson afterward? Out of
  scope for "test the mechanism" — noted for a future roadmap item if this proves out.
- Jetson runs `ROS_DOMAIN_ID=5` (per `mission_control.py`'s `_ROS_ENV`), a different DDS domain
  than Rooster's `ROS_DOMAIN_ID=9` — the republished topic is a self-contained Jetson-local
  ROS2 graph, not bridged back to Rooster's. Confirm that's the intended scope (it matches
  "send it directly to the jetson", not "bridge it back").

## Notes
- 2026-07-29: `{JETSON_REPO}` (`/home/user/agency_ws`) is **not** a git-tracked clone of this
  repo — `git status`/`git log` there show an empty, commit-less `master` with unrelated
  dotfiles as untracked, i.e. its `sparx_agency/` tree is manually copied over at some point,
  not kept in sync via `git pull`. `dir_watch_path_publisher.py` was missing there (last sync
  of `robots/common/` predates its commit `647701b9`) — copied it over directly via `scp` for
  this test. Any future work relying on `{JETSON_REPO}` matching this repo should verify the
  specific file(s) it needs are actually present first, not assume a `git pull` would help.
- 2026-07-29: `mission_control.py` has no `if __name__ == "__main__":` guard — it's a plain
  Streamlit script, UI calls and all, executed top-to-bottom. It can still be `import`ed
  directly to reach `Service`/`start_service`/`stop_service`/`get_all_states` outside a real
  `streamlit run` session: Streamlit's `st.*` calls no-op with a "missing ScriptRunContext...
  can be ignored when running in bare mode" warning instead of raising, when there's no active
  script-run context. Useful for testing the orchestrator's actual code path without driving
  the browser UI.
- 2026-07-29: first `streamlit run` for a manual UI check bound to `0.0.0.0` by default,
  which Streamlit also reported on an *external/public* IP — stopped and restarted with
  `--server.address 127.0.0.1` before doing anything further. Worth remembering any time
  Mission Control gets started ad hoc outside its usual invocation.

## Result
Confirmed working end-to-end, through `mission_control.py`'s own service-management code path:
a JPEG dropped in `/tmp/rooster_frames` is picked up by `dir_push_relay.py` (PC), pushed via
rsync/SSH to the Jetson's `{JETSON_DATA}/rooster_frames_relay`, and republished there by
`dir_watch_path_publisher.py` on `/rooster/jetson_rgb_frame_path` (Jetson-local
`ROS_DOMAIN_ID=5`, intentionally not bridged back to Rooster's `ROS_DOMAIN_ID=9`). Two new
services (`rooster_frame_relay_jetson_R1`, `rooster_jetson_frame_watch_R1`) are now in
`mission_control.py`'s `ROBOTICAN_SERVICES` list, in a new `rooster_jetson` group, additive to
the existing local Rooster/Falcon pipeline (doesn't touch it). Not yet exercised with real
Sphera-flight frames (see deferred step above) or with a real downstream consumer on the
Jetson side of the republished topic — both explicitly out of scope for "test the mechanism."
See `CHANGELOG.md` [Unreleased] for the user-facing entry.
