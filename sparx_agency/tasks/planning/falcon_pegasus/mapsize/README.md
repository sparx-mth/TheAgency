# mapsize — set the exploration area in five numbers, and see what it costs

FALCON wants eighteen numbers to describe where to explore: a min and a max on
each axis for three nested boxes. They are not independent, the containment rules
between them are only checked once the node is up, and one of them silently sets
the mapping resolution. This package holds the five numbers that actually vary
and derives the other thirteen.

**It serves both FALCON stacks.** It lives here because this is where the schema
started, not because it belongs to the simulator:

| stack | files | drawn box |
|---|---|---|
| `falcon_pegasus/` — Isaac Sim | `runs/*.yaml` (also carries the aircraft) | a thin slab at cruise height, for the recorder |
| `falcon/` — the real XTEND, Gazebo, Sphera | `maps/*.yaml` | the whole allocated grid, unless the file says otherwise |

Both launchers call the same command and mount the expanded result, so the file
the planner reads always matches the file you edited.

```sh
# what a run will allocate — either stack, same command
python -m sparx_agency.tasks.planning.falcon_pegasus.mapsize runs/6_whole_office.yaml
python -m sparx_agency.tasks.planning.falcon_pegasus.mapsize ../falcon/maps/hospital.yaml

# try a resolution without editing the file
python -m sparx_agency.tasks.planning.falcon_pegasus.mapsize runs/6_whole_office.yaml \
    --resolution 0.2 --detailed
```

```
  box     28.1 x   71.9 x   1.2 m   =     2424 m3
  map     32.1 x   75.9 x   2.6 m   =     6335 m3
  grid     321 x    759 x    26     =     6.3M voxels @ 0.10 m
  RAM   267 MB
```

`run_falcon_pegasus.sh` runs the same thing with `--out` before starting the
container, so a bad area fails in a tenth of a second with a sentence instead of
inside Docker with a glog `CHECK` and a stack trace.

## The area block

```yaml
map_config:
  area:
    building:        [-23.0, -33.2, 5.1, 38.7]   # x0 y0 x1 y1, what to explore
    flight_band:     [1.0, 2.2]                  # z the planner may fly in
    vertical_extent: [-0.2, 2.4]                 # z to allocate, floor..ceiling
    resolution:       0.10                       # metres per voxel, explicit
    margin:           2.0                        # grid beyond the box, horizontal
```

| key | what it means | how to choose it |
|---|---|---|
| `building` | the exploration box footprint | **edges must land on walls.** An edge in open space makes frontiers along the cut whose viewpoint rings fall half outside and get discarded; the aircraft flies to the boundary and the planner cycles. Inset from the outer walls, never through the building. |
| `flight_band` | the z slice the planner may use | the airspace *above the clutter*, not the whole room. A floor at 0.6 m in this office put the aircraft at desk height and it wedged itself twice. |
| `vertical_extent` | the z range allocated | floor to ceiling. Wider than the flight band because the camera sees the floor and the mapper needs somewhere to put it. |
| `resolution` | metres per voxel | see below. Memory goes as the inverse cube, so this is the biggest lever you have. |
| `margin` | how far the grid reaches past the box, horizontally | 2 m. One number for a symmetric grid, or `[low_side, high_side]` where it is not — `small_house.yaml` really is `-10.5..10.0` around a `-8..8` box. With a slab, at least 0.5 m. |
| `visualisation` | *optional* — an explicit drawn box, `[x0, y0, z0, x1, y1, z1]` | Only when you want to draw less than you allocate. `office.yaml` and `small_house.yaml` do. |

Derived for you: the `map` box (`box` + `margin` horizontally, `vertical_extent`
vertically), and the `vbox`, which is the first of these that applies —

1. `visualisation`, if the file names one;
2. a 20 cm slab centred on `run.cruise_altitude_m`, for a run file with an
   aircraft — the recorder wants one horizontal cut, and every extra layer is
   another full pass over the grid inside the node servicing depth frames;
3. otherwise the whole allocated grid, which is what four of the six `falcon/`
   environments already did by hand.

## Why resolution is explicit

Upstream picks it from the exploration box's volume: under 4000 m³ it maps at
10 cm, at or above it at 20 cm. That couples two things that should not be
coupled, and it does so in the surprising direction — **shrinking your exploration
box can multiply memory by eight**, by dropping under the threshold into the finer
grid. The whole-office runs sit at 2424 m³, just inside that: 267 MB at 10 cm,
33 MB at 20 cm, for exactly the same flight.

So `resolution` is stated. The patched `map_server` reads
`/map_config/map_size/resolution` and uses it when set; an unpatched FALCON
ignores the key and falls back to the volume rule, so nothing breaks either way.
The report warns whenever your choice differs from what the rule would have done,
and says which way the memory moves.

## What the memory figure counts

Six arrays are sized to the whole `map` box and allocated on the first tick —
nothing is sparse and nothing grows later, so a cubic metre never visited costs
the same as one mapped in detail.

| array | bytes/voxel | |
|---|---|---|
| occupancy | 4 | one `enum class` |
| TSDF | 16 | value and weight, both `double` |
| ESDF | 8 | one `double` |
| ESDF scratch ×2 | 16 | **only ever used over the local update region** |
| frontier flag | 0.125 | one bit |
| | **44.125** | |

The two scratch buffers are more than a third of the bill for something that
never needs to be full-sized. See `../SLIDING_WINDOW_MAP.md`.

## Layout

| file | holds |
|---|---|
| `area.py` | `Box`, `ExplorationArea`, and the derivation of the three boxes |
| `memory.py` | the per-voxel accounting and the volume rule |
| `expand.py` | run-file loading, expansion, and validation with readable errors |
| `report.py` | the terminal report |
| `__main__.py` | the command line |

```sh
pytest sparx_agency/tasks/planning/falcon_pegasus/mapsize
```

The test that matters is in `tests/test_area.py`: the derived boxes must
reproduce, to the centimetre, the eighteen numbers the run files carried by hand
before the migration. If that fails, the migration moved a wall.
