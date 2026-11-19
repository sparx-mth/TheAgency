# ObjectNav: hand-off for the RTX 4090 machine (2026-11-12)

Written for a Claude Code session that starts on the office PC with no memory of the laptop sessions. Read it, then ask Daphna what to do first.
Branch: `daphna/detector-compare` (RF-DETR and `closed_open` detector backends, `trace_tools/`, `detector_sweep/`).

## Goal
Beat ASCENT (arXiv 2505.23019; HM3D v1 val SR 65.4, SPL 33.5) and compare with VLFM (arXiv 2312.03275) on zero-shot ObjectNav,
**on single-floor episodes first**, using a VLM and/or a better detector for a better understanding of the scene. Improvement is measured
in SR *and* SPL. Multi-floor (stairs) is a separate, later topic.

## Rules from Daphna (do not break)
- **No tuning on val.** Develop on HM3D v1 *train* (`--split train`); val once at the end, on scenes not used so far.
  Already looked at (do not count as unseen): val scenes 00853, 00890 (and the detector sweep scenes).
- **No per-label thresholds, no hand-written object-to-room dictionaries, no non-general hacks.** A dictionary in the room prompt is cheating.
- Plan before code: give an implementation plan and wait for approval before changing repo files. Test first for bug fixes.
- Plain Hebrew when she writes Hebrew (no mixed Hebrew/English). Say when a long run finishes, with a time estimate.
- Commit style: 1-2 sentences. Push only when she says so. Never enter her Matterport token or any credential.
- If something needs a human eye (ambiguous labels), ask her; do not decide alone.

## What the laptop sessions found (all small samples, val unless stated)
- 20 HM3D v1 val episodes (scenes 00853, 00890, YOLO-World): SR 2/20. 16 wrong STOPs: 7 on a real object that is not in the goal set
  (all in 00853: "l-shaped sofa" 2.6 m, "wall tv" ~1 m), 3 with no object of that kind within 3 m, 6 on an upper floor that has no such object.
  So 9 of 16 are plain false detections. Scene-specific labelling noise makes 20 episodes unrepresentative: more scenes are needed.
- Detector sweep (render + exact GT boxes, val): only 36-39% of boxes at conf>=0.10 hit an object of their kind. AUC by confidence 0.75.
  CLIP crop scoring, BLIP-2 ITM crop scoring and a "both detectors agree" rule do not help (AUC 0.65-0.78 vs 0.75).
- VLM furniture subtypes (qwen3-vl:4b-instruct, GT boxes, 10 train scenes): full frame + red box beats a tight crop
  (tables 76% vs 69%, cabinets 71-75% vs 59%). Wardrobe and shelf are weak. Dropping an "other" label helps cabinets, not tables.
- Room classification (MORE = `core/mapping/topology/room_classifier.py`, qwen2.5:3b) on 40 hand-labelled train rooms (29 clear):
  adding table/cabinet subtypes or geometry (area, shape, doors) gives **no measurable change**; run-to-run noise is 2-4 rooms. Hallways are the weak
  class (objects outweigh shape). Objects assigned to the wrong side of a wall are rare in HM3D (3 of 287); MP3D was not measured.
- Stairs are implemented (`core/planning/object_search/multifloor/`, 2800 lines) but off (`allow_stair_traversal=False`). 19-20% of HM3D v1 episodes
  have the nearest goal on another floor (height proxy). Not the current focus.

## Requirements on the 4090 machine
- Repo checkout of this branch. Python envs used on the laptop: `~/miniconda3/envs/objnav-habitat` (habitat-sim, habitat-lab, py3.9; used for
  `trace_run.py`) and `~/.venvs/objnav-detector` (ultralytics, RF-DETR; **CPU-only torch on the laptop: install CUDA torch here**).
- Data (laptop: `~/datasets_v01`, 35 GB): HM3D **v0.1** scenes (train: 80 annotated scenes with `semantic.glb/.txt`; val) and
  `objectnav/hm3d/v1/{train,val}/content/<scene>.json.gz`. Download with `python -m habitat_sim.utils.datasets_download --uids hm3d_train_v0.1 hm3d_val_v0.1
  --username <ID> --password <SECRET> --data-path <dir>` (Daphna runs it with her own token; ~35 GB, ~10 MB/s on the laptop).
  v0.1 scenes pair with v1 episodes; v0.2 scenes (`~/datasets`) pair with v2 episodes. Do not mix.
- Models (laptop: `~/models/objnav`): `yolov8x-worldv2.pt`, `rfdetr/`, `coco/` (YOLOv8x), `clip/ViT-B-32.pt`, `blip2/` (Salesforce/blip2-itm-vit-g),
  Ollama store with `qwen2.5:3b-instruct`, `qwen2.5:14b-instruct`, `qwen3-vl:4b-instruct` (do not use `qwen3-vl:4b`: the thinking model burns the token budget).
- Services for a run: Ollama (`OLLAMA_HOST=127.0.0.1:11435`, `OLLAMA_NUM_PARALLEL=1`) and the detection server:
  `python -m sparx_agency.tasks.mapping.scene_graph.serve.detection_server --backend yolo_world --model <yolov8x-worldv2.pt> --device cuda --host 127.0.0.1 --port 18192 --conf 0.05 --classes "$(cat sparx_agency/tasks/planning/objnav_benchmark_runtime/trace_tools/vocabulary.txt)"`.
- Laptop-only constraint (RTX 3070, 8 GB): Ollama on the GPU crashed the driver (Xid 13/119/154, hard freeze) together with Habitat. On the laptop everything
  except Habitat runs on CPU. The 4090 should not have this limit, but start with one GPU service at a time and watch `nvidia-smi`.

## Experiment scripts (now in git: `sparx_agency/tasks/planning/objnav_benchmark_runtime/analysis_tools/`, labels in its `data/`; copy `data/*.json` to `~/tc_sweep/`)
`run_train.sh` (phase-A run on train), `pick_train_episodes.py` (seeded single-floor episode picker), `gtbox_extract.py` (HM3D semantic.glb to
object boxes), `run_tc_vlm.py`, `run_table_variants.py`, `run_cabinet_variants.py`, `run_subtype_all.py`, `run_room_more.py`, `run_room_geometry.py`,
`build_label_sheet.py`, scoring scripts, `guard.sh` (pauses the run under memory, load or GPU-temperature pressure). Scripts read paths under `~/tc_sweep`
and `~/datasets_v01`. Hand labels: `~/tc_sweep/room_labels.json` (40 train rooms). Laptop `/tmp` is wiped on reboot: keep outputs under `$HOME`.

## How to run (as used on the laptop)
- `trace_tools/run_trace.sh` runs the unchanged method with a trace on val episodes (`episodes_lm.json` lists 20).
- Train needs `--split train --episodes-per-scene N` (N=50000 loads a whole shard in under a minute) and `--episode-ids scene/NNNNNN` where NNNNNN is the row index.
- Phase A on the laptop (running when this was written): 50 seeded single-floor train episodes, 10 scenes, output `~/trace_train_a`, ~7 min per episode on CPU LLM.
  Rerun it on the 4090 with the same ids (`~/tc_sweep/train_episodes.json`) for comparability.

## Phase A result on the laptop (HM3D v1 train, 10 scenes, 50 seeded single-floor episodes, YOLO-World CPU, qwen2.5:3b CPU)
Outputs: `~/trace_train_a` (laptop; archived to OneDrive), triage in `~/tc_sweep/triage_train_a.json`. Success rule of that run: 0.1 m to a goal view point.
- **SR 62% (31/50), SPL 0.268** (SPL of successes 0.43; ASCENT has SPL/SR 0.51). By category: bed 12/16, toilet 11/15, chair 6/7, sofa 1/5, tv 1/4, plant 0/3.
- With a 1 m rule SR would be about 68% (34/50): 3 episodes stopped 0.1-0.3 m from the goal. Run both rules before comparing to the papers.
- 19 failures: 10 timeouts, 9 wrong stops (5 within 3 m of a real goal, 3 farther, 1 real object not in the goal set).
- **8 of the 10 timeouts saw the target** (box >= 0.3 in 10-53 frames). With GT: 4 of them were within 1.5 m of a real goal (00025/010760 plant, 00025/013033 plant,
  00422/030802 toilet, 00267/045672 sofa) and the closing released the candidate ("unverified", "inspection saw nothing in any view", "no safe path", "occluded").
  Daphna looked at the frames: the boxes are on the real, clearly visible targets. 2 more were real but blocked/stalled (00099/036901 tv, 00141/021959 toilet),
  1 detector miss at 0.8 m (00250/044207 toilet), 1 search failure (00099/020525 bed).
- Upper bound if those 4 had succeeded: about +8 points. Conclusion: the dominant single-floor failure is **verification/closing rejecting real targets**, not false positives.
  (On the 20 val episodes the picture was the opposite; two scenes and 6 upper-floor starts made them unrepresentative.)
- `trace_tools` saves frames at 320x240 but the boxes in `trace.jsonl` are in 640x480: scale boxes by 0.5 when drawing.

## Next step that was proposed (not approved yet)
1. Diagnostic only: log per verification frame why the candidate was not locked (phase, confidence, consecutive coherent frames, `context_check` `suspect`, depth coherence),
   in `trace_tools/trace_run.py`'s Tracer (no change to the method). Rerun the 4 episodes above and classify the cause. Hypotheses: (A) `context_check` fed by a wrong MORE room label,
   (B) incoherent/insufficient depth on small objects, (C) `max_verify_steps=12` too short, (D) the 1 m rejection memory.
2. Fix according to the cause, in a general way (no per-class thresholds), test first. A VLM (full frame + red box) as an accept signal for a close, clear candidate is the idea to test.
3. Measure paired on the same 50 episodes, then on 100-150 on the 4090, split into "real close target" vs other.
Daphna continues this on the 4090 PC at the office.

## Scope (Daphna, 2026-11-15): detection and classification are ours; planning and navigation are Nadav's
Findings below that belong to Nadav (not worked on here): escape trap (2/50 phase A, 1/20 holdout; both phase-A cases start at z 1.25 / 1.62 with `drop_cells`
1419 / 983 of the spawn-floor guard; 459-469 of 500 steps in escape, 66 forward steps, 0 blocked), memory-STOP refusals (1/50: 31 refusals 0.3 m from the goal,
`_memory_in_sight` finds an occupied cell between), path efficiency (SPL/SR 0.43). Nadav's MP3D entries 044-047 (`feat/objnav-habitat-mp3d-nadav`) already cover the same mechanisms.

## Holdout and detector results (laptop, train; no per-label thresholds)
- Holdout: 20 fresh single-floor train episodes in 10 unseen scenes (`~/tc_sweep/holdout_episodes.json`, seed 23): **SR 65%, SPL 0.285** (phase A: 62%, 0.268). Same picture.
- 6 of 70 episodes (phase A + holdout) end with a STOP where no object of the kind is within 3 m; 4 of the 6 are beds. Detector on fresh train frames with exact boxes
  (YOLO-World, one global threshold 0.30, precision/recall): toilet 75/87, sofa 73/74, chair 60/38, bed 58/55, tv 51/37, plant 22/33.
- Verification deadlock (`00422/030802`): toilet box conf 0.95 in 36 consecutive frames, not locked ("glimpse", no usable depth); the agent stood still 34 steps because the rule
  that steps toward a glimpsed candidate needs distance > terminal + one step (1.62 m). A glimpse of 10+ steps appears in 5 of 50 episodes (2 failures, 3 successes at ~34 wasted steps).
- VLM check on stored 320x240 frames (qwen3-vl:4b-instruct, frame with a red box, "is it a X?"): rejects 5 of 5 true false-detection stops, accepts 42 of 42 successful stops.
  On frames of real, close targets the agent released: toilet and sofa accepted (clear ones), **plants rejected 6 of 6** and one low-confidence toilet rejected. Small samples, low resolution,
  ~50 s per frame on a CPU (expect ~1 s on the 4090). Re-test with full-resolution frames inside the run. The earlier in-loop module (`closing/vlm_confirmation.py`, commit 41eb1bde) was removed at Daphna's request (it carried hand-written per-label definitions, a prior on the environment); any VLM check must be built without it.

Daphna reviewed the frames (2026-11-15): the VLM was right on all 6 wrong stops (including the real tv stopped at 3.3 m). The plants it rejected are real but tiny at 320x240:
a resolution limit. Design to test on the 4090: ask the VLM about a zoomed crop of the box, with some context, cut from the full-resolution frame inside the run, applied by box size in pixels (a general rule, not per label).

## VLM check on 117 detector bboxes (train sweep, zoomed full-resolution crop) and the 4090 plan (2026-11-15)
- Question text decided by Daphna: the benchmark's category name only (`bed`, `couch`, `chair`, `tv`, `toilet`, `potted plant`), no definitions. Crop = bbox grown by 0.5 of its size, at least 224 px, bbox drawn in red.
- Result (`analysis_tools/vlm_box_check.py`, `data/vlm_boxes_results.jsonl`, qwen3-vl:4b-instruct): correct bboxes accepted 58/59, wrong bboxes rejected 28/57 (bed 7/10, tv 6/10, sofa 6/10, chair 4/10, plant 3/10, toilet 2/7);
  tiny plants accepted 9/9 with the zoomed crop. Some "wrong" bboxes may sit on a real object and only fail the IoU rule. Not yet measured inside a run.
- Laptop warning: Ollama's CPU runner (`llama-server`) grew from 4.9 to 10 GB over ~100 images and the laptop froze; `analysis_tools/mem_guard.sh` stops the resumable job when available memory < 1.8 GB. Unload the model (`keep_alive: 0`) every few dozen images.
- 4090 plan: (0) pull, services on CUDA, rerun 3-4 episodes to check the local baseline; (1) offline sweep of all 1,616 train-sweep bboxes with 4B and 8B; (2) new in-loop check before STOP (TDD, flag off by default, category name only);
  (3) paired runs: the 12 wrong-stop episodes first, then the 70 train episodes, then 100-150 new ones; SR and SPL paired with CIs, vetoes per episode, plants separately. Pre-set criteria: at most one correct STOP blocked, at least half of the wrong STOPs rejected, no negative SR difference.

See also `todos/objnav-4090-room-score-handoff.md` for the VLM room-score and frame-selection line (scripts in `analysis_tools/room_score/`).

## Todos, in order
1. **Baseline on train, single-floor only** (phase A): SR, SPL, then failure triage with GT as in `todos/` session: wrong stop vs timeout; for wrong stops,
   is a same-kind object within 3 m, is it in the goal set (`goals_by_category` object ids), is it the goal but farther than the success radius.
   Rewrite the triage script (it lived in laptop `/tmp` and is lost). Use HM3D semantic boxes in public frame (x=gy, y=-gx).
2. **Scale to 100-150 episodes** on the 4090. At 40% base SR, a paired +10 point change needs about 100+ episodes. Report paired differences with CIs.
3. **VLM confirm before STOP**: the old module was removed (hand-written per-label definitions); build a new check that asks the VLM about the box without per-label text, paired against the baseline.
4. **Detector choice by false-positive stops**, not by global AUC: YOLO-World vs `closed_open` vs RF-DETR on the same episodes.
5. **SPL**: measure wasted steps in scans, revisits and escapes; test what shortens the path (search order, MORE, node oracle).
6. **Bigger models on the 4090**: `qwen3-vl:8b` for confirm, 14B for the reasoning calls, CUDA detectors. Compare to the 3B/4B laptop setting.
7. Still open from `todos/objnav-open-items.md`: escape retrace, thick mapped walls, per-class suspect penalty (not approved), budget counting only SEARCH actions.
8. Optional: MP3D wall-crossing check (Nadav's remark that objects cross walls more in MP3D). Needs MP3D GT; not on disk.
9. Last: one val run on scenes not used so far (18 of the 20 annotated val scenes), versus ASCENT's numbers. Success rule in the papers: STOP within 1 m;
   the habitat-lab HM3D config uses `success_distance` 0.1 to view points: check which one the runner uses before comparing.

## Pitfalls
- v1 train shards hold ~50k rows sorted by category, so "the first N rows" are one category. Pick ids from the whole shard.
- Start validation (`--validate-starts`) can refuse a scene; `run_trace.sh` retries without it and writes `validation_skipped.txt`.
- Only 20 of 102 v0.1 val scenes and 80 of 803 train scenes have semantic annotations.
- LLM answers are not fully deterministic even at temperature 0 (Ollama); repeat a baseline run before reading small differences.
- `test_hm3d.py` (untracked on the laptop) targets a different loader version and fails here; ignore it.
