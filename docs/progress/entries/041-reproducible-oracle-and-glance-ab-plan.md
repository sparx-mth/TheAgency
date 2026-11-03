# 041 - Reproducible LLM answers, a reproducibility check, and the plan for the geometric-glance A/B

**Branch:** `feat/objnav-habitat-hm3d-nadav`
**Status:** steps 1-2 done (uncommitted at the time of writing); the A/B (step 3) planned, not launched
**Follows:** entry 040: two re-runs of 40 cases disagreed on 19 episodes, most of it oracle noise

## 1. Can Ollama answer a prompt the same way every time?
Probes on 20-40 prompts recorded from two real episodes, sent through the run's own `LLMClient` (Ollama 0.33.2,
`qwen2.5:3b-instruct` Q4, RTX 5070 Laptop, `OLLAMA_NUM_PARALLEL=1`, temperature 0, seed 17):

| mode | answers that changed with the request order | s per call |
|---|---|---|
| one request at a time | 13-18 of 40 (15 of 17 beyond the reason text) | 2.6-3.7 |
| a one-token flush before each request | 13-17 of 40 | 2.7 |
| a per-request tag at the head of the system prompt, fresh prompts | 15-17 of 20 | 3.3 |
| prompts the server had already seen | 0 of 20 (but they differ from the fresh answers) | 3.2 |
| the model unloaded before every request (`LLM_EXCLUSIVE=1`) | **0 of 20**, also with two workers at once | 6.4 |

What the server does: it keeps the KV state of many earlier prompts, not only the last. A prompt sent 19 long prompts
earlier was evaluated in 30 ms, a fresh one in 520 ms, and a prompt sent right after the flush in 56-95 ms instead of
740-1,180 ms. How much of a prompt is reused, and from which entry, depends on history, and a greedy decode follows
the numerics. Only a cold runner makes the answer a function of the request. The cost is a model load and a cold
prompt evaluation: +2.9 s a call, about 1.8 times the LLM time.

## 2. The answer store
`core/mapping/topology/llm_answer_cache.py`: `LLMAnswerStore`, attached to `LLMClient` (both routes; the preflight
probe bypasses it). It is keyed on the exact request: backend, model, both prompts, temperature, seed, window, reply cap
and format.
- `LLM_ANSWERS_RECORD=<dir>` keeps every answer a run used, one shard per process, prompts included.
- `LLM_ANSWERS_REPLAY=<dir>` serves a recorded run's answers.
- A question asked again within a run gets its first answer.
- Episode rows count `llm_answers` (replayed / repeated / asked / recorded).

The limit: once a glance changes what the agent has seen, the prompts differ and the store misses. Arm B shares arm
A's answers only up to its first geometric glance.
- On 8 random single-storey episodes, 4 never took a geometric glance: those episodes replay A completely.
- Pooled over calls, 36% of the calls came before the first geometric glance. One 500-step failure that glanced at
  step 44 holds 47 of the 85 calls.
- Expect 35-45% of arm B's calls to hit, and every call of the episodes B never glances in.

Misses: a deterministic backend (`LLM_EXCLUSIVE=1`) makes a miss reproducible, but buys no statistical power. After
the first glance the arms ask different questions, and the answer to a different question is effectively an
independent draw whether or not the backend is deterministic. What makes the A/B measure the glances is the store:
until B does something A did not, B gets A's answers, so the arms part only where glances act.

## 3. Verified: the same configuration twice
Eight random single-storey episodes from 8 scenes, 3 on the GPU renderer and 5 on Mesa, geometric glances on.
Steps are compared after dropping wall-clock fields only.

| runs | identical episodes | LLM answers |
|---|---|---|
| 1 vs 2 (cold start, each asking afresh) | 7 of 8 | 145 of 145 the same |
| 3 vs 4 (cold start, after the fix below) | **8 of 8**, 1,191 steps | 145 of 145 the same |
| 3 vs 5 (warm, replaying run 3) | **8 of 8** | 154 replayed, 0 asked |

Run 1 vs 2 found a real bug. In 00813/000000, at action 83, one run's kinematic follower re-planned and the other
kept its plan; maps, poses, answers and actions were identical. `KinematicFollower._collision_mask` cached its mask by
`id(world)`. A new planning world allocated at the freed one's address read the same id, and the plan was judged
against an older map. The fixed code reproduces run 2's verdict, so run 1 had used the stale map. The key is now a
digest of the occupied cells (`LESSONS.md` 2026-10-31).

Resources, runs 3-5:
- CPU 6% mean (15% max), RAM 48% max, VRAM 47% max.
- GPU utilisation 31% mean, above 60% in 46 of 503 samples: prompt evaluation after each reload.

## 4. The A/B plan (not launched)
- **Episodes:** the 1,614 single-storey HM3D v1 val episodes (the shortest path changes height by at most 1 m;
  `runs/determinism/single_storey_1614.json`), in one seeded random order. A pilot takes the first 300. An extension
  continues the same order, so the pilot is the head of the full sample, not a selection.
- **Arms:** A has `geometric_glances` off; B has `{"geometric_glances": true}` via `--policy-config`. Both run from one
  frozen snapshot, with ground-truth perception, `qwen2.5:3b-instruct` in both roles and seed 17. A scene keeps its
  renderer in both arms; the laptop runs 1 GPU worker and 2 Mesa workers.
- **Order:** A first, recording its answers; then B replays A's answers and records its own.
- **Analysis** (a scratch prototype, `ab_analysis.py`, checked on the reproducibility runs; it lands in `hm3d/`
  with the run, on `objnav_benchmark/stats.py`):
  - Primary: ΔSR (B − A), 95% paired-bootstrap interval, exact McNemar test on the flips.
  - Secondary: ΔSPL, paired-bootstrap interval, sign-flip test.
  - Integrity: every episode in which B took no geometric glance must end exactly as in A. Any that does not is a
    bug, and stops the analysis.
  - Diagnostics: the arm-B hit rate, glances per episode, and what first parted each changed pair.

**Sample size** for McNemar (two-sided α 0.05, power 0.8): it depends on p_d, the share of episodes whose success
differs between the arms. On the 40 hard cases p_d was 0.30, with the oracle noise included. Here, with the noise
removed and half the episodes never glancing, 0.05-0.10 is likely; the pilot measures it.

| p_d | pairs for ΔSR 3 pts | pairs for ΔSR 2 pts | smallest ΔSR at 300 | at 1,614 |
|---|---|---|---|---|
| 0.05 | 434 | 979 | 3.6 pts | 1.6 pts |
| 0.08 | 695 | 1,567 | 4.6 pts | 2.0 pts |
| 0.10 | 869 | 1,959 | 5.1 pts | 2.2 pts |
| 0.15 | 1,305 | 2,940 | 6.3 pts | 2.7 pts |

So 300 pairs is a pilot: it can show a 4-6 point effect, not a 2-3 point one. Only the full single-storey split reaches
2-3 points, unless p_d comes out low. SPL is cheaper to resolve: for an SPL difference of 0.02 at a paired SD of
0.10-0.20, 196-784 pairs are enough.

**Runtime** on the laptop, from the measurements above:
- Per worker: 0.39 s a step outside the LLM, 0.13 LLM calls a step (about 22 calls for a ~170-step episode).
- LLM: one serialized server at 3.5 s a call warm, 6.4 s cold.
- The run is LLM-bound: about 77 s of LLM time per arm-A episode against 22 s of worker time per episode.

| | warm (store only) | cold (`LLM_EXCLUSIVE=1`) |
|---|---|---|
| pilot, 300 pairs | ~10 h (A 6.4 + B 3.8) | ~19 h |
| 1,000 pairs | ~34 h | ~63 h |
| full split, 1,614 pairs | ~55 h | ~100 h |

**Recommendation:** run warm with the store. It pairs the arms exactly as the cold mode would, at half the time, and
each arm stays reproducible by replaying its own record. Keep `LLM_EXCLUSIVE=1` for debugging a single episode. Run
the 300-pair pilot, measure p_d, then extend along the same order to the n it implies.

**Open for the decision:**
- Whether the GPU's bursts above 60% are acceptable; in warm mode they were 1% of samples in entry 040's run.
- Whether to raise `OLLAMA_NUM_PARALLEL` for throughput. The store makes the resulting nondeterminism harmless, but it
  means restarting the Ollama container.
- Which effect size is the decision threshold.

## 5. The pilot (2026-11-01, 300 pairs, `runs/glance_ab/`)
- **Setup:** arm A without geometric glances, recording its LLM answers; arm B with glances, replaying them. Ollama
  ran with 2 slots, 1.4 times the throughput.
- **Interruptions:**
  - The first start stopped at 50 episodes: the answer store was keyed without the episode (`LESSONS.md` 2026-11-01).
  - Two RAM pauses: at 00:14, two Mesa workers on heavy scenes; at 12:15, the user's own applications. After the
    first, one Mesa worker; the user accepted about 61 % to finish.
  - Arm B's Mesa half first died on a resume-only flag.
  - Finished 2026-11-01 18:10 UTC.

| | SR | SPL | DTG (m) |
|---|---|---|---|
| A, without glances | 0.853 | 0.469 | 0.88 |
| B, with glances | 0.837 | 0.465 | 1.04 |
| B - A | -1.7 pts, 95% CI [-5.0, +1.7], McNemar p = 0.44 | -0.003, CI [-0.023, +0.016], p = 0.75 | |

- **Flips:** 11 to success, 16 to failure; flip rate 9.0 %.
- **Integrity:** B never glanced in 124 episodes, and all 124 end exactly as in A.
- **Answer replay:** 35 % of B's LLM answers came from A.
- **Decision rule:** OFF. The SR gain is not significant, and the interval excludes a gain above 1.7 points. At the
  measured flip rate, detecting +3 points takes 783 pairs and +2 points 1,764, about 18 h and 56 h more at 0.038 h a
  pair. But the pilot makes a true gain of that size unlikely: with a true +1 point, extending to all 1,614 pairs
  ends significant with a probability of about 8 %.
