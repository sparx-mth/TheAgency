# 027 - The image-space pitch servo, the whole-box confident STOP, and the closing regression check

**Branch:** `feat/objnav-habitat-gibson-nadav`
**Status:** in-progress -- code, tests and docs done; the final nine-slot recording is running
**Roadmap item:** ObjectNav target closing: an image-space pitch servo, a whole-box confident STOP, and the toilet fixes verified against couch/tv/bed

## Goal
Three things, in order. (1) Show that the toilet-specific closing rules of 2026-10-11 (the along-ray
association cylinder, the floor-cut sighting, the below-the-frame STOP) did not degrade the other
categories -- couch, tv, bed -- or introduce new false positives, by re-recording the nine inspection
slots under them and comparing outcome by outcome against the baseline recordings. (2) A generic,
class-agnostic camera pitch controller driven by the target's BOX in the frame rather than by the 3-D
memory: within 1.5 m of the target, a box in the lower region of the frame (or about to be cut by the
lower edge on the next step) tilts the camera DOWN, one in the upper region (or about to be cut above)
tilts it UP, a centred box leaves the camera alone; and the terminal STOP is taken on a vertically
centred, whole box with a valid depth measurement and a confident detection. (3) Re-record the affected
classes with the final code, render the MP4s, commit and push.

## Why
The 1,000-episode validation run (entry 026) ended at SR 0.653 with toilet at 0.288, and the nine
failure recordings showed the toilets were found and released on bottom-cut frames. The previous session
fixed that for the tilt-less Gibson protocol with three closing rules and verified them on the three
toilet slots only (2 of 3 became successes). Whether those rules cost anything elsewhere was not measured.
And the existing pitch controller was geometric -- it trusted the memory's height, which a cut frame
biases -- so where LOOK actions exist (the multi-floor development protocol, any tilt-capable platform)
the camera could be driven to hunt against the box it was meant to keep.

## Steps
- [x] Re-record all nine inspection slots on HEAD (the toilet fixes) into `runs/gibson_val_20261008T120112Z.inspection-fix`
      (slots 1-3 reused from the previous session) and tabulate against `runs/…inspection` (the baseline
      closing): toilet 2/3 improved, couch 3/3 same outcome (Wiconisco/000114 STOPs 31 actions earlier on the
      same wrong object via the blind-cone rule -- not a new false positive), bed same, tv Collierville/000043
      same, **tv Collierville/000194 regressed** (SR 1 -> 0, step limit, DTG 4.14 m).
- [x] Trace the regression in `steps.jsonl`: identical through step 237; from 238 the association cylinder kept
      the real television's frames (`depth_associations` 11) and refined the memory by 0.11 m; at step 242 that
      shift crossed `refine_distance_m`, `TargetApproachPath` dropped a working standoff goal and re-selected a
      bearing-relative ring none of whose eight points had a path; the lock was released twice as "no safe path
      from here". The baseline only succeeded because its un-refined memory kept the old goal.
- [x] Fix the fragility, not the symptom: `TargetApproachPath` keeps a standoff that still lies within
      `terminal_distance_m` of the refined memory (`kept`, `goal_kept`), re-references on every refinement
      (the counter no longer runs every action after a failed re-selection) and re-selects a kept goal the
      action a slow drift carries it past the terminal radius (`test_reobservation_refines_goal_but_missing_frames_do_not`).
- [x] `methods/bbox_pitch.py`: the pure image-space servo -- `row_after_tilt`, `box_after_tilt`,
      `box_after_advance` (the box on the frame the next step produces), `vertical_fix` (cut both / cut bottom /
      cut top / lower / upper / centre, pitch limits, a band never narrower than the tilt's rounding dead band),
      `region_compatible`; 17 tests in `tests/test_bbox_pitch.py`.
- [x] Wire it into `TargetClosing`: `_servo` / `_servo_pitch`; `_approach_pitch` is the servo's within
      `bbox_pitch_distance_m` with a fresh box (the camera stays on a centred box even where the geometry would
      force a tilt) and the geometry's otherwise; `_verification_view` asks the servo first; `_inspect` takes the
      pitch from the servo on a fresh frame. Settings `bbox_pitch`, `bbox_pitch_distance_m` 1.50, `bbox_pitch_region`
      0.25; counters `bbox_pitch_corrections`, `bbox_region`, `bbox_pitch_steps`, `box_drift_rows`, `confidence`.
- [x] The terminal STOP gates (`_before_stop`): a fresh in-range box under `stop_confidence` (0.50) is held for the
      frame that keeps it whole, at most `weak_stop_max_holds` (2) times; a box the servo can re-frame earns one
      centring tilt per inspection (`stop_centring`), with the safety net that the frame after the tilt STOPs on the
      previous frame's evidence if the detector drops the box (`centring_step`). The STOP's info carries
      `confidence`, `bbox_region`, `depth_valid`. Without the LOOK pair nothing changes.
- [x] Six integration tests in `tests/test_target_closing.py` (the servo on the approach incl. the forced-geometric
      case, the predicted cut, the centring tilt and the whole-box STOP, the once-per-inspection cap, the bounded
      confidence hold, the Gibson path unchanged); the Hanson off-centre test re-pinned to "a TURN first, at most
      one centring LOOK_DOWN, the STOP within three actions". Runtime package: 734 passed, the 13 pre-existing
      `skfmm`/habitat failures unchanged; core objnav + harness 1266 passed.
- [x] README (standoff keeping, the servo as the controller and the geometry as its fallback, the new subsection),
      CHANGELOG, LESSONS (the 000194 mechanism), this entry, the ROADMAP line.
- [ ] Re-record the nine slots with the final code into `runs/gibson_val_20261008T120112Z.inspection-bbox` and
      compare the three columns (baseline / toilet fixes / final); record the table in the Result.
- [ ] Commit and push.

## Open questions
- The published Gibson protocol has no LOOK pair, so the servo and the centring tilt are exercised by the
  real-converter tests and the multi-floor development protocol, not by the Gibson recordings; the Gibson
  recordings show the toilet fixes, the STOP gates' tilt-less behaviour and the standoff fix. A tilt-enabled
  Gibson variant would be a different `protocol_id` and was not added.
- `stop_confidence` 0.50 touched one of the eleven recorded fresh STOPs (a 0.41 potted plant, correct; a box
  filling the frame, so no frame-keeping action existed and it STOPped at once anyway). Whether a higher
  threshold buys anything on the Wiconisco couch false positives (0.72-0.91) it cannot -- those are detector
  false positives at high confidence; the context check and the room priors are where that fight is.

## Notes
- 2026-10-11: the regression table is three columns because the previous session's `-fix` directory already
  held slots 1-3; `record_episodes.sh` reuses a job directory that holds its episode, so the nine-slot command
  recorded only 4-9 there. The final `-bbox` directory is fresh for all nine.
- 2026-10-11: while a recording run is going, no `.py` under `sparx_agency/` may change -- each job imports the
  code at its start, so a later job would run different code from an earlier one (LESSONS 2026-10-08). The
  servo module and its tests were written as *new* files while the regression recordings ran (nothing at HEAD
  imported them), and the wiring into `target_closing.py` waited for the run to end.

## Result
(when the final recording ends)

