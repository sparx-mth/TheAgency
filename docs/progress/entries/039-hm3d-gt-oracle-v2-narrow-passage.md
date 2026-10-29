# 039 - HM3D v1 val, ground-truth perception, after the narrow-passage navigation overhaul (GT oracle v2)

**Branch:** `feat/objnav-habitat-hm3d-nadav`
**Status:** done
**Roadmap item:** the perception upper bound of the room-search method (HANDOFF.md, 2026-10-26)

## Goal
Re-measure the method under ideal perception (habitat-sim's semantic sensor for boxes, labels and masks) on the full
HM3D v1 val split after commits `604e6e76` (the collision cascade, the closing's dead ends, rooms finished from the
doorway, the storey localiser) and `50bbd877` (the discrete-action lattice follower, the medial-axis cost, the doorway
scrape guard and look-down, strict kinodynamic tracking), against the sweep of 2026-10-26.

## Result (2026-10-29 14:46 UTC)

Run: `gt_oracle_2000ep_v2_20261028T134440Z` -- 8 scene processes at once on the one GPU (EGL renderers ~0.5-0.8 GB each) beside a native Ollama with
3 request slots (`qwen2.5:3b-instruct` labels + `qwen2.5:14b-instruct` node oracle, both resident on the card), lean
records, no recording; VRAM held at 88-96 %, the LLM's request queue the throughput ceiling (CPU ~25 %, RAM ~17 GB).

| | episodes | SR | SPL | SoftSPL | DTG (m) | mean steps | mean path (m) | steps (successes) | STOP / limit / error | refused steps / ep |
|---|---|---|---|---|---|---|---|---|---|---|
| **GT oracle v2 (this run)** | 2000 | **65.3 %** | **0.353** | 0.371 | **3.43** | 278 | 23.1 | 163 | 1324 / 676 / 0 | 6.1 |
| GT oracle sweep 2026-10-26 | 2000 | 58.4 % | 0.328 | 0.346 | 3.97 | 288 | 24.4 | 150 | 1192 / 774 / 34 | 11.9 |
| change | | +6.9 pp | +0.03 | +0.03 | -0.54 m | -10 | -1.3 m | +13 | | -5.9 |

`campaign_summary.json`: SR 0.653 (95 % CI 0.632-0.674), SPL 0.353 (CI 0.340-0.367), SoftSPL 0.371, DTG 3.43 m, 2000 episodes, complete split: True.

### Per category

| category | episodes | SR v2 | SR sweep | SPL v2 | SPL sweep | DTG v2 (m) | DTG sweep (m) |
|---|---|---|---|---|---|---|---|
| bed | 433 | 57.3 % | 51.7 % | 0.280 | 0.260 | 4.59 | 5.00 |
| chair | 428 | 80.4 % | 73.6 % | 0.467 | 0.448 | 1.31 | 1.71 |
| plant | 84 | 57.1 % | 51.2 % | 0.313 | 0.260 | 7.38 | 6.81 |
| sofa | 376 | 73.1 % | 72.1 % | 0.438 | 0.451 | 2.34 | 2.69 |
| toilet | 398 | 60.3 % | 44.2 % | 0.285 | 0.211 | 3.67 | 4.91 |
| tv_monitor | 281 | 53.7 % | 49.5 % | 0.289 | 0.273 | 4.80 | 5.37 |

### Per scene

| scene | episodes | SR v2 | SR sweep | SPL v2 | SPL sweep | DTG v2 (m) | DTG sweep (m) | mean steps v2 | sweep |
|---|---|---|---|---|---|---|---|---|---|
| 00800-TEEsavR23oF | 99 | 64.6 % | 53.5 % | 0.370 | 0.314 | 6.51 | 7.25 | 282 | 302 |
| 00802-wcojb4TFT35 | 99 | 65.7 % | 55.6 % | 0.367 | 0.332 | 1.71 | 2.48 | 272 | 300 |
| 00813-svBbv1Pavdk | 99 | 96.0 % | 91.9 % | 0.555 | 0.572 | 0.35 | 0.62 | 133 | 131 |
| 00814-p53SfW6mjZe | 99 | 43.4 % | 40.4 % | 0.241 | 0.213 | 5.88 | 6.52 | 355 | 361 |
| 00820-mL8ThkuaVTM | 99 | 24.2 % | 19.2 % | 0.129 | 0.119 | 6.80 | 6.66 | 419 | 421 |
| 00824-Dd4bFSTQ8gi | 99 | 86.9 % | 79.8 % | 0.438 | 0.410 | 0.52 | 1.03 | 204 | 221 |
| 00829-QaLdnwvtxbs | 99 | 97.0 % | 85.9 % | 0.537 | 0.528 | 0.07 | 0.73 | 125 | 153 |
| 00832-qyAac8rV8Zk | 99 | 40.4 % | 43.4 % | 0.194 | 0.219 | 6.77 | 6.88 | 373 | 336 |
| 00835-q3zU7Yy5E5s | 99 | 52.5 % | 41.4 % | 0.341 | 0.280 | 3.11 | 3.60 | 284 | 323 |
| 00839-zt1RVoi7PcG | 119 | 62.2 % | 62.2 % | 0.362 | 0.367 | 3.84 | 3.27 | 277 | 266 |
| 00843-DYehNKdT76V | 99 | 65.7 % | 60.6 % | 0.399 | 0.333 | 2.53 | 3.23 | 280 | 294 |
| 00848-ziup5kvtCCR | 99 | 81.8 % | 68.7 % | 0.445 | 0.339 | 0.95 | 1.65 | 221 | 250 |
| 00853-5cdEh9F2hJL | 99 | 86.9 % | 67.7 % | 0.400 | 0.310 | 1.14 | 3.01 | 261 | 296 |
| 00873-bxsVRursffK | 99 | 59.6 % | 46.5 % | 0.302 | 0.253 | 3.76 | 4.74 | 302 | 331 |
| 00876-mv2HUxq3B53 | 99 | 77.8 % | 80.8 % | 0.400 | 0.468 | 1.86 | 2.05 | 245 | 218 |
| 00877-4ok3usBNeis | 99 | 82.8 % | 80.8 % | 0.485 | 0.491 | 1.56 | 1.64 | 201 | 195 |
| 00878-XB4GS9ShBRE | 99 | 34.3 % | 26.3 % | 0.167 | 0.159 | 4.08 | 4.45 | 388 | 395 |
| 00880-Nfvxx8J5NCo | 99 | 87.9 % | 79.8 % | 0.444 | 0.435 | 0.84 | 1.90 | 224 | 227 |
| 00890-6s7QHgap2fW | 99 | 42.4 % | 36.4 % | 0.244 | 0.183 | 8.51 | 9.89 | 366 | 394 |
| 00891-cvZr5TUy5C5 | 99 | 54.5 % | 46.5 % | 0.245 | 0.233 | 7.71 | 7.98 | 337 | 342 |

## Notes
- The run is the perception UPPER BOUND (`publishable: false`): the detector and segmenter are bypassed by the simulator's
  semantic sensor; the method -- mapping, rooms, the LLM oracle, RPT*, the lattice follower, the closing -- is unchanged.
- Per-episode rows: `<root>/<scene>/episodes.jsonl` (lean); the frozen configuration in `<root>/<scene>/run.json`;
  the layer's statistics (`kinematic`, `doorway_scrape`, `doorway_look_down`) in every row's `agent_info.policy`.
- The sweep it is compared with: `gt_oracle_2000ep_20261026T131737Z` (SR 58.4 %, SPL 0.328, DTG 3.97 m, 2,000 episodes).
