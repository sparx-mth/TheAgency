# Analysis tools: furniture subtypes, room labels, geometry, and the phase-A train run

Experiment scripts from the laptop sessions (see `todos/objnav-4090-handoff.md`). They are not part of the method. They expect a working folder `~/tc_sweep`
with the rendered frames and ground truth; `data/` holds the hand labels and episode ids to copy there.

| Script | What it does |
|---|---|
| `gtbox_extract.py <out dir> <scenes.json>` | boxes, names and regions of every HM3D annotated object (from `semantic.glb`/`.txt`) |
| `pick_train_episodes.py <scenes.json> N` | seeded single-floor v1 train episodes per scene, written to `~/tc_sweep/train_episodes.json` |
| `run_train.sh` | the unchanged method on those episodes (`--split train`); `guard.sh` pauses it when the laptop is under pressure |
| `run_tc_vlm.py`, `run_table_variants.py`, `run_cabinet_variants.py`, `run_subtype_all.py`, `eval_tc.py` | VLM furniture subtype test on exact GT boxes (needs Ollama `qwen3-vl:4b-instruct` on port 11435) |
| `build_label_sheet.py`, `run_room_more.py`, `run_room_geometry.py`, `score_room_*.py` | hand-labelled rooms and MORE with/without subtypes and geometry |

Scene frames come from `detector_sweep/sample_frames.py`-style rendering (the train variant was an ad-hoc script and is not included).
Paths are under `$HOME` (`~/tc_sweep`, `~/datasets_v01`, `~/models/objnav`); change them with the machine.
