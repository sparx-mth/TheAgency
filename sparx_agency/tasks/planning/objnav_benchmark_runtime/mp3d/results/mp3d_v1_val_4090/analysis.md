# MP3D v1 val on the RTX 4090 profile: where the run succeeded and where it did not

Campaign `runs/mp3d_v1_val_4090`, 2026-11-24 15:52 UTC to 2026-11-25 05:38 UTC (13 h 46 min), `complete: true`,
2,195 of 2,195 episodes over 11 scenes, no agent-error rows. Launched with
`mp3d/run_4090.sh --gt-detailed --run`: the `rpt` agent (VLM frontier search, `qwen2.5vl:7b` on port 11450,
2 request slots), 4 shard workers, **ground-truth perception** (habitat-sim's semantic sensor in place of
YOLO-World and SAM). The numbers are therefore an upper bound under ideal perception and are not publishable
benchmark results. `summary.md` holds the headline table and provenance; `storey_report.md` the storey split.

## Headline

| episodes | SR [95 %] | SPL [95 %] | SoftSPL | DTG | SR at ApexNav's 0.2 m |
|---|---|---|---|---|---|
| 2,195 | 0.632 [0.612, 0.652] | 0.296 [0.284, 0.307] | 0.321 | 4.08 m | 0.644 (+26 episodes) |

## Where it fails

808 failures: 720 ran out of the 500-action budget, 88 called STOP in the wrong place.

- **Goals on another storey are never reached.** 388 episodes have every goal view point more than 0.5 m from
  the start height: 0 successes, 387 step limits, DTG 13.6 m. Same-storey episodes (1,807): SR 0.768, SPL 0.360,
  DTG 2.03 m. The cross-storey episodes alone are 48 % of all failures.
- **Same-storey step limits: 333.** Over all 720 step-limit episodes the median distance to the goal at the end
  is 11.1 m (403 at 10 m or more, 283 between 3 and 10 m, 34 under 3 m): the target was not found, not missed
  at close range.
- **Failed STOPs: 88.** 57 stopped within 1 m of a goal view point (43 under 0.5 m) -- near misses of the 0.1 m
  criterion; 20 stopped 3 m or more away.
- **Scenes.** `X7HyMhZNoso` 0.220 and `8194nk5LbLH` 0.259 (three quarters of their episodes hit the step limit),
  `x8F5xyUWy9e` 0.480, `EU6Fwq7SyZv` 0.571; the other seven lie between 0.650 and 1.000 (`pLe4wQe7qrG`).
- **Categories.** Lowest: `gym_equipment` 0.023 (44 episodes), `tv_monitor` 0.250 (8), `bed` 0.271 (48),
  `sofa` 0.377 (53), `towel` 0.425 (73). Highest: `table` 0.844 (340), `seating` 0.759 (87), `cabinet` 0.733
  (180). `seating` pairs a high SR with a low SPL (0.199).
- **Path efficiency.** Successful episodes take 133 actions on average at SPL 0.468.

## The VLM

45,648 answers recorded (20.8 per episode), server-side median 0.36 s and 95th percentile 0.77 s per call. The
shard consoles log 683 `frontier_vlm` failures where the reply was not valid JSON (runs of `<|im_start|>`
tokens); each falls back to the nearest frontier for that decision.

## Compute

GPU compute averaged 20 % (median 1 %) against the profile's ~90 % target; VRAM 16.8 GB mean, 17.9 GB peak of
24.6 GB; CPU 20 % of 24 cores; RAM 20.6 GB mean. Throughput was about 160 episodes an hour. The four
single-threaded simulator workers were the limit, not the card: the profile's worker count was kept as
configured, and it is locked to a campaign root once admitted. `utilization_summary.json` has the figures.

## What happened during the run

- One crash. `oLBMNvg9in8/shard_5` exited 1 at `oLBMNvg9in8/000093`, action 85, on the kinematic self-check
  ("TURN_RIGHT should turn in place"). The recording wrapper did not forward the simulator's step-filter
  placement to the check; fixed in `a4be4813` (`LESSONS.md` 2026-11-25). No tolerance was changed.
- That shard's remaining 10 episodes were continued by hand with the harness's
  `--resume --resume-accepting-changes source_sha256`, as a fifth worker beside the campaign for about 15
  minutes. Its 8 finished episodes were kept.
- 66 shards ran on the source as launched and 46 on the source after the fix (each shard's `run.json` records
  its fingerprint). The fix touches only the self-check's input.
- No stalls: the 30-minute watchdog never fired.
- The tree was not clean at launch: `--gt-detailed` and its service changes were uncommitted work on top of
  `56896d06`. `source-changes.diff` here is that difference as the launcher recorded it, and
  `workstation_4090.json` the admitted profile (4 workers and every service setting).

## Where the detailed logs are (on the workstation, not in git)

- `runs/mp3d_v1_val_4090/<scene>/shard_<k>/episodes.jsonl` -- the full policy report per episode.
- `runs/mp3d_v1_val_4090/<scene>/shard_<k>/recordings/<key>/` -- `steps.jsonl` (every decision, pose and
  command), `trajectory.csv`, `metrics.json`, `map_journal.npz`.
- `runs/mp3d_v1_val_4090_vlm_answers/` -- every VLM prompt, image digest and reply, by episode.
- `runs/mp3d_v1_val_4090/utilization.csv`, `run.log`, `<scene>/shard_<k>.console.log`.
- `runs/mp3d_v1_val_4090_support/` -- launcher output, the commands run, the probe that reproduced the crash.


