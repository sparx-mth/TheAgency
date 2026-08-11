# memwatch — what a run actually holds, and whether the map explains it

Samples the exploration node's resident memory during a flight and splits the
trace into the two things worth telling apart: the **allocation**, which happens
once at startup, and the **growth**, which is everything after it.

That split is the whole diagnosis. FALCON's voxel map is a dense array sized on
the first tick and never resized, so if the map is the cost the trace steps early
and then sits flat. Anything still climbing an hour in is something else.

```sh
# terminal 1
./run_falcon_pegasus.sh 6_whole_office

# terminal 2
python -m sparx_agency.tasks.planning.falcon_pegasus.memwatch \
    --run 6_whole_office --out /tmp/office_mem.csv
```

```
  samples   15 over 29 s
  startup     426.5 MB   (allocation, once the node settles)
  final       404.8 MB
  peak        427.4 MB
  growth      -21.6 MB after startup,  -123.0 MB/min
            (over only 29 s -- too short to mean anything ...)
            (16 trailing reading(s) dropped as the node exiting)

  the voxel grid alone was predicted at   266.6 MB, which is 63% of the startup figure.
```

Pass `--run` and it costs the run's voxel grid with `mapsize` and reports the
grid as a share of the measured startup. That is the number that decides where to
look: a grid that is most of the startup figure means shrinking the box or
coarsening the resolution will help, and one that is a small fraction means it
will not.

`--summarise <csv>` re-reads a finished run without sampling anything.

## What it measures, and what it refuses to claim

- **Per process, not per container.** The container also runs the trajectory
  server, the bridge, the recorder and roscore; the question is what the map
  costs. The container total is recorded in the CSV alongside for comparison.
- **The largest matching process, not the sum.** More than one process carries
  `exploration_node` on its command line — roslaunch launched it, so its own
  cmdline contains the name — and summing them folds a supervisor's few megabytes
  into the figure, then reports the supervisor alone once the node has gone.
- **The teardown is dropped.** A run that ends normally ends with the node
  letting go of its map; those readings are not measurements. Trailing readings
  below half the peak are trimmed and the count is reported. Only from the end,
  so a genuine dip mid-run still counts.
- **A short window is called short.** The working set swings by tens of megabytes
  between planning cycles — this office run oscillates about 22 MB — so a slope
  fitted over less than two minutes is fitting that oscillation. Below
  `MIN_GROWTH_WINDOW_S` the report says so rather than quoting a trend.

## Measured so far

One stub flight of `6_whole_office` (10 cm voxels, 6335 m³ allocated):

| | |
|---|---|
| predicted voxel grid | 267 MB |
| measured startup | **427 MB** |
| grid as a share of startup | **63 %** |
| growth | **not established** — the window was 29 s |

So the prediction holds and the allocation is up front, as designed. The
remaining ~160 MB is the process itself, ROS, and the mapper's own working
buffers. **Growth over a long flight is still unmeasured** and needs a run of at
least a few minutes of continuous mapping.

## Layout

| file | holds |
|---|---|
| `sample.py` | reading `/proc` through `docker exec`, and the CSV format |
| `summary.py` | allocation, peak, growth, and what it refuses to claim |
| `__main__.py` | the command line |

```sh
pytest sparx_agency/tasks/planning/falcon_pegasus/memwatch
```
