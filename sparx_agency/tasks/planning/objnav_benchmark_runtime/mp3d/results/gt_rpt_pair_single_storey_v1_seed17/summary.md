# MP3D single-storey benchmark: frontier VLM vs. room LLM, GT perception

The frontier VLM succeeded on **149 of 180** episodes; the room LLM succeeded on **146 of the same 180**. The visual pipeline also had higher mean SPL and took fewer steps. Neither the SR nor the SPL difference was statistically significant at the 5% level on this set.

| Metric (all 180 episodes per agent) | Legacy `rpt_room_llm` | Visual `rpt` | Visual minus legacy |
|---|---:|---:|---:|
| Successes | 146 | 149 | +3 |
| Success rate (SR) | 81.11% | 82.78% | +1.67 percentage points |
| SPL | 0.3450 | 0.3772 | +0.0321 |
| SoftSPL | 0.3713 | 0.3996 | +0.0283 |
| Mean final distance to goal | 1.842 m | 1.402 m | -0.440 m |
| Mean actions | 200.04 | 184.42 | -15.62 |
| STOP / step limit | 156 / 24 | 158 / 22 | +2 / -2 |

Both succeeded on 135 starts and both failed on 20. Of the discordant pairs, 14 succeeded only with the VLM and 11 only with the room LLM. The paired 95% episode-bootstrap interval for the SR difference was **[-3.89, +7.22] percentage points** (exact McNemar p = 0.690). For SPL it was **[-0.0022, +0.0666]** (paired sign-flip p = 0.071). This compares the two complete policies, not the effect of changing a language model while holding the policy fixed.

## What was held constant

- **Episodes:** [the fixed single-storey MP3D v1 validation set](../../sets/single_storey_flat_v1.json), eight scenes, simulator and model seed 17. The episode IDs, published starts and goals, input-file hashes, and initial geodesic distances were checked across arms. All means include step-limit failures. The cohort is complete; it is **not** the full MP3D validation split.
- **Scoring:** Habitat-Sim 0.2.4, protocol `mp3d-objectnav-v1-val/2`, 500-action limit and a required STOP within the protocol's 0.10 m success radius.
- **Perception:** Habitat ground-truth boxes, labels and instance masks for both agents (64-pixel minimum, 0.25 m on-object filter). Neither YOLO nor a learned segmenter was used. The visual policy requested the goal and stairs; the legacy policy retained its multi-category context.
- **Execution:** The policies ran sequentially from commit `65a58747d23cbba2279ec53c54f098708e66544d`, with the same Python source fingerprint in all 16 scene records. All workers exited zero, with no missing or repeated episodes, agent errors, or recorded pipeline failures. The simulator used Mesa software rendering; actual model inference ran locally on the RTX 4090, without stand-ins or answer replay.

The visual arm made 4,227 frontier-scoring calls to `qwen2.5vl:7b` (`boxed-frontier-v2`). The legacy arm used `qwen2.5:3b-instruct` for both room labels and scoring, with 601 room-oracle queries (separate from labelling). Both models used Q4_K_M weights. Ollama reported 12.9 GiB resident for the visual model and 2.2 GiB for the text model after their respective preflight probes; these are not peak VRAM measurements.

## Records

- [Paired results and verification](comparison.json): model digests, cohort and input hashes, per-scene and per-category outcomes, success flips, and paired tests.
- [Legacy episodes](rpt_room_llm/episodes.csv) and [visual episodes](rpt/episodes.csv): join on `episode_id` to reproduce the comparison.
- [Legacy configuration](rpt_room_llm/run.json) and [visual configuration](rpt/run.json): one original scene record per arm. The `-dirty` Git stamp reflects an unrelated untracked file; tracked source was unchanged during measurement. The report-title fix was made after both campaigns finished and did not affect actions or scores.
