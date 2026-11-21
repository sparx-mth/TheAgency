# Hand-off for the 4090: a VLM score for "is the target in this room", and which frames to score (2026-11-19)

Written on the laptop for a Claude Code session on the RTX 4090 PC with no memory of the laptop sessions. Read `todos/objnav-4090-handoff.md` first for the rules and the
earlier (detector/verification) work; this file is only about the room-score line. Branch: `daphna/detector-compare` (the office branch `daphna/objnav-verify-diagnosis-rebased` is ahead of it by 32 commits).

## Rules (Daphna, do not break)
- Scope: detection and classification are ours; planning, navigation and mapping are Nadav's. No gimmicks, no per-label thresholds, no object-to-room tables or per-label text.
  Prior knowledge is allowed only inside the VLM/LLM itself. Ground truth is for labelling and analysis only, never an input of the method.
- Train scenes only for development (HM3D v1 train); val once at the end. Plan before code, tests first. Never push unless Daphna says so. Plain Hebrew when she writes Hebrew.
- The question text may use the benchmark's own category query (`bed`, `couch`, `chair`, `tv`, `toilet`, `potted plant`) and nothing else.

## The question Daphna wants answered
Can a VLM score "how likely is the target in this room / is it worth continuing the search toward it" and feed it to RPT* **without room-type classification**?
Two sub-questions: (1) when the agent enters a new room area it keeps frames in a buffer **from its normal walk, with no special route and no extra steps**; which
frames of the buffer give the most accurate score (candidates: much floor visible, the VLM judges a frame informative, many objects, diversity, ...)? (2) option A: a room is
still a node but has no type; option B: no rooms at all, clusters of frontiers scored from the frame that looks toward them (not started).

## What was done on the laptop (RTX 3070 8 GB, laptop frames are 320x240 JPEGs saved every 5th call)
- `analysis_tools/room_score/build_buffers.py` builds `data/buffers.json`: 191 room visits (57 episodes, 20 train scenes, 1,336 unique frames, median 5 frames per buffer) from the saved traces
  (`~/trace_train_a`, `~/trace_holdout`), with per-frame depth share inside the room (`share_in_room`, `floor_share`), pose, detector classes, the episode's target and ground-truth presence of
  each of the six categories in the room's HM3D region (from the benchmark's goal set; evaluation only). Only 30 of 191 room visits hold the episode's own target; 123 of 191 targets are bed or toilet.
- Scoring script `score_frames_logprob.py` loads Qwen3-VL-2B-Instruct with `transformers` on the GPU (fp16, 4.3 GB; Ollama is not needed) and reads the answer probability: `P(yes)` =
  sigmoid(logsumexp(log p of Yes/yes) - logsumexp(log p of No/no)) at the first generated position. A JSON answer with percentages was saturated (almost only 0 and 100), so use `P(yes)`.
- Prompt 1 (old, six objects per frame, `data/vlm_frame_logprob.jsonl`): "This image shows one view inside a room of a home. Is there a [OBJ] somewhere in this room (it may be outside this view)? Answer yes or no."
- Prompt A (prior): "Imagine walking into the room shown in this image. How likely is it that you would find a [OBJ] in it? Answer yes or no."
- Prompt B (search direction): "A robot is searching for a [OBJ]. This image shows a room it could go to next. Is it worth going there to continue the search? Answer yes or no."
- `score_frames_target.py` asks A and B about the episode's target only (1.1 s/frame on the 3070); `score_frames_six.py` asks all six objects for A and B (6.4 s/frame); `chain_six.sh` queues it. `eval_selectors.py` compares frame
  selection rules (random, even, floor, room share, objects, diverse, VLM sum) at K=1/3/5, three aggregations (mean, max, noisy-OR), AUROC per target, all rooms and rooms where the detector saw the target in no frame.

## Results (laptop, all train, Qwen3-VL 2B and 4B, 191 room visits; `analysis_tools/room_score/eval_prompts.py` reproduces every table from `data/`)
**Correction of an earlier reading:** on the episode target only, prompt B (search direction) looked best out of view (AUROC 0.82 against 0.63), but that subset has only 8 positive rooms. With six objects per frame (many more labelled pairs)
it does not hold: B is not better and A is not better than the plain prompt. Do not report "B is better".

Episode target only (one question per frame, what the agent would ask); AUROC with scene-bootstrap 95% interval:

| model and prompt | all rooms (30 hold the target) | target not seen by the detector (8 hold it) |
|---|---|---|
| 2B old ("is there X in this room") | 0.845 [0.74, 0.94] | 0.632 [0.47, 0.77] |
| 2B A (prior, "would you find X") | 0.856 [0.76, 0.94] | 0.731 [0.54, 0.85] |
| 2B B (is it worth searching there) | 0.872 [0.77, 0.94] | 0.818 [0.55, 0.95] |
| 4B A | 0.723 [0.54, 0.88] | 0.433 [0.13, 0.59] |
| 4B B | 0.767 [0.65, 0.88] | 0.547 [0.31, 0.74] |

Six objects per frame (2B), mean AUROC over targets with at least 4 positives: all rooms old 0.790, A 0.758, B 0.709; target not seen old 0.598, A 0.582, B 0.565 (A minus old -0.024 [-0.108, +0.036], B minus old
-0.040 [-0.176, +0.080]). B is near chance for tv (0.487). Per target (old prompt, all rooms): toilet 0.91, bed 0.85, couch 0.78, chair 0.76, tv 0.73, plant 0.71; not seen: chair 0.62, bed 0.70, plant 0.58, tv 0.64, couch 0.45.
- Frame selection hardly matters (old prompt, K=3, not seen): random 0.599, even 0.601, floor 0.622, room share 0.629, diverse 0.619, objects 0.555, VLM sum 0.590; all frames 0.598; mean/max/noisy-OR equal (0.620-0.622).
- Reading: the score separates rooms that show the object, and hardly separates rooms where it is out of view (about 0.6 for every prompt, model size and selection rule tried). The 4B model with partial CPU offload (`device_map='auto'`) was not better than 2B.
  This matches the office result (AUROC 0.57-0.73, RPT* ordering gain only where the target is seen, not replicated on fresh scenes). Limits: 320x240 JPEGs every 5th call, median 5 frames per buffer, few positives (8 out of view for the episode target).
- Files: `data/vlm_frame_logprob.jsonl` (six objects, prompt 1), `data/vlm_frame_six_2b_{A,B}.jsonl`, `data/vlm_frame_target_{A,B}.jsonl` (2B, episode target), `data/vlm_frame_target4b_{A,B}.jsonl`, `data/buffers.json`. The 1,336 frames (13.5 MB) are not in git:
  archive `objnav_room_score_2026-11-21.tar.gz` (OneDrive).

## Plan for the 4090
0. Pull, CUDA env, download Qwen3-VL-2B/4B/8B-Instruct from HuggingFace, copy the laptop archive, run `eval_selectors.py` and `eval_prompts.py` on the laptop scores to reproduce the tables above.
1. New data: 100-150 seeded single-floor train episodes in scenes not used for design, **balanced across the six target categories**, with **dense full-resolution frames** (640x480, every call or every other call). This needs a small
   change in `trace_tools/trace_run.py` (`FRAME_EVERY` and the saved size): plan it first, test first. Dense buffers (tens of frames) are what make frame selection testable; the laptop buffers have only 5.
2. Score every frame with 2B, 4B, 8B and prompts 1, A, B, for the episode's target and for all six objects. Choose the prompt and model on a split of scenes, report on the other.
3. Evaluate: AUROC with scene-level bootstrap intervals, K=1/3/5/8, all rules and aggregations, per target and pooled, separately for rooms where the target is not seen by the detector. Pre-set criterion to keep a score: AUROC of at least 0.70 with an interval above 0.60 on
   unseen train scenes, and a selection rule whose interval beats random.
4. RPT* ordering test (the office has it: `tests/test_rpt_order_eval.py` and the ordering scripts on `daphna/objnav-verify-diagnosis-rebased`) with the surviving score against nearest-first **and against the planner's own score** (room labels from
   `RoomTypeClassifier` plus the node oracle probabilities), which is still missing. Without that baseline nothing can be said about "better than the current score".
5. Only if 2-4 are positive: option B (frontier clusters, no rooms).

## Pitfalls
- `transformers` 4.57: use `dtype=` (not `torch_dtype`), `Qwen3VLForConditionalGeneration`, `processor.tokenizer.padding_side='left'` for batches; cost grows linearly with batch size (each prompt re-encodes the image).
- Ollama's CPU runner leaked memory (4.9 to 10 GB over ~100 images) and froze the laptop; GPU Ollama + Habitat froze it earlier. On the 4090 run one GPU job at a time anyway. `gpu_guard.sh` kills the (resumable) scorer when available memory is under 1.8 GB or the GPU over 83 C.
- Targets are 123/191 bed or toilet in the laptop buffers and only 30 rooms hold the episode target: balance the new set. "Out of view" is defined by the detector, not by ground-truth visibility.
- The old `closing/vlm_confirmation.py` was removed at Daphna's request (hand-written per-label definitions); do not bring it back.
