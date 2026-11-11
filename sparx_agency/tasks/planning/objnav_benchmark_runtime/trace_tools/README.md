# Trace tools: what the room search sees, asks and credits to rooms

`trace_run.py` runs the unchanged method on chosen episodes and writes `trace.jsonl` beside a `frames/` folder. Every detector call
(boxes, pose with `z`, `floor_id`, the share of the frame's depth that falls in each room) and every LLM call (prompt, reply, and for
room-label calls a snapshot of the rooms' cells and the confirmed landmarks with their rooms) is recorded.

- `run_trace.sh`: the whole run (3B room labels and node oracle). Needs Ollama with `qwen2.5:3b-instruct` and the YOLO-World detector
  started with the vocabulary in `vocabulary.txt` (port 18192). `VERSION=v2` matches scenes under `scene_datasets/hm3d` of release
  0.2; `v1` episodes need release 0.1 scenes (see `hm3d/assets.py`). `EPISODES_PER_SCENE=N` takes the first N episodes of each scene.
- `build_review.py <trace dir> <out.html>`: the room review page (map, landmarks, frames, label).
- `episodes_lm.json`: the 20 v1 episodes of the first landmark trace.
