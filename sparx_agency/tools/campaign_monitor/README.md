# campaign_monitor — how far the NavDP run has got, and how long is left

A full NavDP build is two long jobs back to back: a flight-recording campaign
that runs for hours, then a fine-tune that runs for hours more. Neither prints a
percentage. `collect.py` logs one line per flight and has no idea how many the
campaign wants; `train.py` logs a metrics table and has no notion of wall-clock
remaining. This joins each to a target and shows the number people actually
want, which is when it will be done.

```bash
python3 -m sparx_agency.tools.campaign_monitor.dashboard \
    --recordings ~/data/sim/office --episodes 3000 --max-bytes 250e9 \
    --dataset ~/data/navdp/world_goal/dataset \
    --run ~/data/navdp/world_goal/run1
```

Add `--once` to print a single frame and exit, which is what to use from a
script, a cron job, or a connection that cannot hold a live screen. The system
`python3` is enough — there is no torch, no `psutil` and no ROS import anywhere
in this package.

```
1  data collection ──────────────────────────────────────────────
  flights      ███████████▎                  32%  968 / 3,000
  disk         █████▏                        14%  35.2 GB / 250.0 GB
  landed       ████████████████████████████▌ 84%  813 of 968
  5 live worker(s)  ·  412,330 frames  ·  318 flights/h
  landed 813  crashed 84  missed_goal 51  stalled 20
  ETA 6h 24m  (collection)
```

## What it reads

Everything comes from files the running jobs already write. Nothing here
attaches to a process, imports torch, or needs the job to cooperate — so the
dashboard can be started, killed and restarted at any point in a multi-day run
without disturbing it, and several people can watch the same campaign.

| panel | source |
|---|---|
| collection | each recording's `meta.json`, plus `du` over the campaign directory |
| offline pipeline | the six output files `run_pipeline.sh` skips a stage on |
| training | `run.json` and `metrics.jsonl` in the run directory |
| machine | `/proc/stat`, `/proc/meminfo`, `nvidia-smi`, `statvfs` |

**Progress is counted from recording directories, never from the campaign
manifest.** `collect.py` writes `campaign_w*.json` once, when a worker exits, so
a manifest-based reading shows nothing for the first hour and then jumps by
sixty. A recording gains its `meta.json` the moment its flight ends, which makes
the directory listing an accurate incremental record. A directory with images
and no `meta.json` is the flight currently in the air, and is counted separately
— real disk and real progress, but not yet a usable sample.

## Reading the colours

They are the opposite way round from a health dashboard, deliberately. This run
is trying to *use* the machine, so on the CPU, RAM and GPU-utilisation bars
**green means busy** and red means idle — an idle GPU during collection is the
problem worth noticing. On the two bounded resources, VRAM and disk, green goes
back to meaning "plenty left".

The utilisation target is 80 %, which is `dashboard.UTILISATION_TARGET`.

## ETA, and how much to trust it

Collection throughput is measured over the **last 40 finished flights**, not the
whole campaign, so the figure tracks the worker count that is running now
instead of being dragged down by the three and a half minutes each worker spends
booting Kit. Expect it to be pessimistic for the first few minutes of a campaign
and steady after that.

Training throughput is measured **between the oldest and newest record still in
the tail**, not as `step / wall_s` over the whole run. The whole-run average
breaks on a resumed run: the step counter carries on from the checkpoint while
`wall_s` restarts at zero, so a run resumed at step 45,000 reported 11.4 steps/s
for a machine doing 3.6, and an ETA a third of the truth. A resumed run says so
on the elapsed line — `elapsed 1h 36m since resuming at 45,000` — because the
step count and the clock otherwise look inconsistent with each other.

A **wrong `--run` path is called out rather than shown as "not started"**. The
two produce the same empty reading and the first is much the more common, so a
run directory that does not exist prints `no such directory` and the path.

The training total is not written anywhere by the trainer, so `training.py`
reconstructs it: `--max-steps` when it was given, otherwise a `... optimiser
steps` or `... steps per epoch` note in `run.json`, otherwise nothing — and a
missing total shows as a dotted bar rather than a fabricated percentage.

## Files

| file | what it owns |
|---|---|
| `collection.py` | scanning a campaign directory into flights, outcomes and a rate |
| `training.py` | `run.json` + `metrics.jsonl` → steps, losses, ETA; and the six-stage checklist |
| `resources.py` | CPU, RAM, GPU and disk sampling, and a cached `du` |
| `bars.py` | progress bars, colours, durations — no knowledge of what it is drawing |
| `dashboard.py` | the CLI that puts the four panels on one screen |

## Related

* `tasks/planning/sim_flight_recording/campaign_supervisor.py` writes
  `supervisor.json` and `supervisor.log` into the campaign directory; the
  dashboard does not need them, but they say why a worker was recycled.
* `tasks/planning/sim_flight_recording/inspect_recording.py` is the other half
  of watching a campaign — it renders contact sheets and plan views, which is
  how you see that the *content* is right rather than merely that it exists.
