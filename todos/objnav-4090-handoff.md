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

## Todos, in order
1. **Baseline on train, single-floor only** (phase A): SR, SPL, then failure triage with GT as in `todos/` session: wrong stop vs timeout; for wrong stops,
   is a same-kind object within 3 m, is it in the goal set (`goals_by_category` object ids), is it the goal but farther than the success radius.
   Rewrite the triage script (it lived in laptop `/tmp` and is lost). Use HM3D semantic boxes in public frame (x=gy, y=-gx).
2. **Scale to 100-150 episodes** on the 4090. At 40% base SR, a paired +10 point change needs about 100+ episodes. Report paired differences with CIs.
3. **VLM confirm before STOP** (already in the repo, commit 41eb1bde, entry 042): evaluate the full-frame + red-box prompt against the current one, paired.
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
