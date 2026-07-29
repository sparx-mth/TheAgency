# ASGARD

**The simulator is not in this repository.** ASGARD is a third-party **Unity**
application — `AsgardSystems / SwarmUI` — that runs as a separate process (on
Windows under `%LOCALAPPDATA%Low/AsgardSystems/SwarmUI/`, on Linux under
`~/.config/unity3d/AsgardSystems/SwarmUI/`). It renders the world, simulates the
drones, and exposes an **HTTP REST server** on a per-instance port (`8081`,
`8082`, …).

Everything in this folder is **client code**: a thin REST wrapper around that
server, plus a scripted multi-drone mission built on top of it. There is no
physics, no rendering, and no environment here — only `requests.post(...)` calls.

The scenario the code implements is a **swarm pest hunt**: N drones take off,
fly to randomized start positions around a target, scan for "pests", deconflict
with each other over who takes the kill, dive on the nearest one, and
"exterminate" it.

```
   this repo                                    external Unity app
┌────────────────────┐                    ┌──────────────────────────────┐
│ asgard_wrapper.py  │  process spawner   │                              │
│   └ AsgardPlanner  │  mission FSM       │   ASGARD / SwarmUI           │
│       └ AsgardCtrl │ ──── HTTP/JSON ──> │   127.0.0.1:8081..8085       │
└────────────────────┘  requests          │   (world, physics, drones)   │
                                          └──────────────────────────────┘
                                                        │ writes
                                                        ▼
                                          ConfigFiles/*.json, Results/*.json
```

> **Status: this folder does not currently run.** It was lifted from
> `pre_baseline/old_stuff/Asgard_Daniel/` in two commits (`7524611`, `eebe05e`)
> and the refactor broke the imports. See [Known problems](#known-problems).

### Where the simulator comes from

**It is not open source and it is not publicly available.** No source, binary,
submodule, Dockerfile, build script, or licence for it exists anywhere in this
repo, and a public search turns up nothing under either name. It is a closed
third-party (or partner) build that has to be obtained from whoever owns it.

The evidence for what it is comes from the Unity data paths in the original
config: Unity writes to `AppData/LocalLow/<CompanyName>/<ProductName>` on Windows
and `~/.config/unity3d/<CompanyName>/<ProductName>` on Linux, which fixes the
Unity project settings as **Company = `AsgardSystems`, Product = `SwarmUI`**.
Those paths point at a specific developer's Windows machine (`C:/Users/Danielco/…`)
with a commented-out Linux alternative, so at least two people have had it
installed. It is a full GUI application, not a headless library — the client
still carries a `connect_button_click()` method and prints "mission planner
connected", it reads a scenario file the app authors (`ConfigFiles/*Current.json`),
and it writes batch outcomes (`Results/<run>/Combined Results.json`).

> ⚠️ **Name collision:** "SwarmUI" is also a popular open-source Stable Diffusion
> web UI ([mcmonkeyprojects/SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI)).
> Completely unrelated — do not follow those docs.

The practical consequence: **the simulator cannot be modified from this repo.**
The 13 endpoints below are the entire contract. If ASGARD needs to expose
something it does not expose today (a camera stream, an episode reset, a
synchronous step), that is a request to its owner, not a change we can make.

---

## Files

| File | Lines | What it is |
|---|---|---|
| `asgard_controller.py` | 295 | **The API client.** `AsgardController` — one class, one method per REST endpoint, plus small motion helpers. The only file that talks to the simulator. |
| `asgard_planner.py` | 251 | **The mission.** `AsgardPlanner` — a per-drone state machine (`random_scan` → `attack` → `finish_mission`) with the swarm deconfliction logic. |
| `asgard_wrapper.py` | 237 | **The launcher.** `Wow` / `RunWrapper` / `Wrapper` / `PreRun` — spawns one OS process per simulator instance and one per drone, and picks randomized start positions. |
| `geometric.py` | 46 | 3 static math helpers: distance, direction vector, height differences. |
| `results.py` | 60 | Offline scoring — reads the Unity app's `Combined Results.json` and prints a success percentage. |
| `asgard_controller_test.py` | 27 | Not a pytest test — a manual smoke script (takeoff, yaw, strafe) you run by hand. |
| `__init__.py` | 0 | Empty. |

Nothing else in `sparx_agency/` imports any of this. The folder is an island: it
uses no `core/` algorithm, no `core/common/types`, no ROS, and has no
`adapters/` or `config/` subdirectory like `XTEND/` and `ROBOTICAN/` do.

---

## The API

All calls are JSON over HTTP to `{ip}:{port}`, `Content-type: application/json`.
Every request except the two GETs carries `DroneID`. Endpoint semantics below are
**inferred from the client code** — the server is closed and was not available to
verify against.

| Verb | Endpoint | Request body | Client method |
|---|---|---|---|
| GET | `/` | – | `start()` — reachability probe, expects `200` |
| GET | `/GetDrones` | – | `getAllDrones()` → `{"DroneIDList": [...]}` |
| POST | `/Connect` | `{DroneID}` | `connectDrone(id)` — claim control of a drone |
| POST | `/DroneParamsByDroneID` | `{DroneID}` | `getDroneParams(id)` — **static** properties |
| POST | `/DroneDataByDroneID` | `{DroneID}` | `getDroneData()` — **live** telemetry + sensing |
| POST | `/Move` | `{DroneID, Direction:{x,y,z}, Speed}` | `moveDrone(dir, speed)` — velocity command; `{0,0,0}` = stop |
| POST | `/MoveTo` | `{DroneID, toPosition:{x,y,z}, Speed}` | `moveToPosition(pos, speed)` — go-to waypoint; can reply `"Failed"` |
| POST | `/SetSpeedByDroneID` | `{DroneID, Speed}` | `setDroneSpeed(speed)` |
| POST | `/SetRotationByDroneID` | `{DroneID, rotation:"RIGHT"\|"LEFT", Degrees}` | `rotateDrone(dir)`, `rotateDroneYaw(deg)` — body heading |
| POST | `/SetDroneYaw` | `{DroneID, Degrees}` | `setYawRotation(deg)` — used for the scan sweep |
| POST | `/SetDronePitch` | `{DroneID, Degrees}` | `setPitchRotation(deg)` — camera/gimbal pitch (`90` = look down) |
| POST | `/Extermination` | `{DroneID}` | `exterminateDrone()` — fire at whatever is in range |
| POST | `/TransmitPackage` | `{DroneID, Level, Data}` | `transmitPackage(level, data)` — drone-to-drone message |

Responses are returned as **raw text** (`response.text`); the caller `json.loads`
it. Nothing checks status codes after the initial probe.

**Two yaw endpoints exist and the code uses them differently.**
`/SetRotationByDroneID` takes an absolute heading plus a turn side, and
`rotateDrone()` computes that heading from the current one; `/SetDroneYaw` takes
plain degrees and the planner alternates `+60 / -60` with it to sweep. Which one
is absolute and which is relative is not established by this code.

### `getDroneParams` — static, per drone

Keys used by the code:

| Key | Used for |
|---|---|
| `droneID` | identity; the string passed as `DroneID` in every later call |
| `exterminationRadius` | kill range — the planner fires once distance drops below it |

### `getDroneData` — live, per drone

| Key | Type | Notes |
|---|---|---|
| `position` | **string** `"(x, y, z)"` | not a JSON object — see the parsing note below |
| `rotation` | string/number | heading in degrees, `float()`-able |
| `isActive` | bool | `False` ⇒ the drone is dead/disabled; the planner ends its mission |
| `lineOfSightTargets` | `{"Targets": [...]}` | **the entire perception system** — see below |
| `networkPackage` | `{"packages": [...]}` | messages received from other drones |

`position` is a **stringified tuple** and the code parses it two different ways:
`ast.literal_eval(...)` → `(x, y, z)` tuple (planner), and
`AsgardController.extractPosition(...)` → `{"x":…, "y":…, "z":…}` dict (distance
math). Both target the same field.

### `lineOfSightTargets` — ground-truth sensing

Each entry in `Targets`:

| Key | Meaning |
|---|---|
| `ID` | unique entity id |
| `Type` | `Roaming` / `Static` / `Evasive` ⇒ **pest**. Anything else ⇒ **another drone** |
| `Alive` | dead entities are filtered out |
| `position` | same `"(x, y, z)"` string format |

This is the simulator handing over a **perfect, pre-labelled list of what is
visible** — not pixels. `extract_pests_from_LOS()` splits it into
`(pest_list, drones_list)`; `extract_pests_from_LOS_using_network()` additionally
drops pests that a teammate already claimed over the network.

### `networkPackage` — the swarm radio

`transmitPackage(2, {"state": pest_id})` broadcasts a claim. Receivers read
`packages[i]["2"]` (the key is the `Level`) and recover the id with
`.split("State: ")[1]` — i.e. the server renders the payload dict into a string
before delivering it.

---

## Coordinate frame — Unity, not FLU

ASGARD uses **Unity's left-handed, Y-up frame**, which is *not* the repo-wide
body FLU convention (see `CLAUDE.md`):

| Axis | ASGARD (Unity) | Repo FLU (REP-103) |
|---|---|---|
| `x` | **right** | forward |
| `y` | **up** | left |
| `z` | **forward** | up |

The direction constants at the top of `asgard_controller.py` encode this:
`directionUp = {y: 1}`, `directionForward = {z: 1}`, `directionRight = {x: 1}`.
The planner reads altitude as `position[1]`, confirming `y` is up.

`go_to_direction_relative()` converts a body-relative command into this world
frame with a rotation about the **y** axis, using the drone's reported heading.
`go_to_direction_absolute()` skips that and commands world axes directly.

**Anything bridging ASGARD to `core/` must convert frames explicitly.** Nothing
in this folder does that today.

---

## The mission (`AsgardPlanner`)

One `AsgardPlanner` per drone, each in its own OS process.

**Startup**, in `loop()`: lift off to `flying_height` → pitch the camera to `90°`
(straight down) → `MoveTo` its assigned start position until within 2 m →
busy-wait until the shared wall-clock `t_start` so the whole swarm begins
together.

**Then, forever:**

```
                   ┌─────────────────┐
                   │   random_scan   │  pick 1 of 4 directions, rotate,
                   │                 │  fly, sweep yaw ±scan_degrees
                   └────────┬────────┘
              pest in LOS   │
                            ▼
                   ┌─────────────────┐
                   │ awareness_smart │  who takes this target?
                   └────────┬────────┘
        peer is on it ──────┤──────── target lost / peer is down
        (sleep, retry)      │              (back to scan)
                            ▼
                   ┌─────────────────┐
                   │     attack      │  MoveTo nearest pest; once
                   │                 │  dist < exterminationRadius → fire
                   └────────┬────────┘
                            ▼
                     finish_mission     process exits, launcher restarts it
```

**The deconfliction protocol is altitude.** There is no negotiation channel in
`smart` mode — a drone signals intent by *descending* toward its target, and
peers read that from `lineOfSightTargets`:

- another drone more than `threshold` **below** me → treated as "drone down",
  abandon the target and climb back to scan altitude (`check_drones_down`);
- another drone **within** `threshold` in height → "peer already on this target",
  sleep a random 1–5 × `sleep_factor` and re-evaluate (`check_another_drones_on_target`);
- otherwise → claim it, switch to `attack`.

`awareness_default` (`smart: False`) skips all of that and attacks immediately.
`network: True` swaps the LOS filter for the radio-aware one, so claims are
explicit instead of inferred from altitude.

The whole `while` body is wrapped in a bare `except` that prints the exception
and ends the mission — any error looks like a completed run.

---

## Process model (`asgard_wrapper.py`)

Four layers, each fanning out:

| Class | Responsibility |
|---|---|
| `Wow` | Holds one `RunWrapper` per simulator instance; `Process()`-spawns them all. `__main__` sets up 5 instances on ports `8081`–`8085`. |
| `RunWrapper` | **Infinite episode loop** for one simulator: `subprocess.Popen` a fresh Python that runs `main(ip, port)`, wait for it, terminate, sleep, repeat. Episodes are reset by *killing the process*. |
| `main(ip, port)` | One episode: load `config.yml` → retry `PreRun.random_starting_locations()` until it works → list the drones → build a `Wrapper` → start it → `quit()`. |
| `Wrapper` | Holds one `AsgardPlanner` per drone; `Process()`-spawns each one's `loop()`. |

`debug: True` collapses each fan-out to a single instance running inline, which
is the only way to get a traceback out of this design.

### `PreRun` — randomized episode setup

Reads the Unity app's own config JSON (`config_path`) for the ground-truth world
position of pest `pest_idx_to_attack`, then places each drone on a random point
of a **horizontal ring** around it: radius uniform in `[r1, r2]`, altitude
`+flying_height`, azimuth random but forced at least `min_deg_diff` apart. If the
two drones' approach paths would cross in the ground plane (`shapely` segment
intersection over x–z), it swaps their assignments.

This is the domain-randomization knob for the scenario — and it is hardcoded to
**exactly two drones** (the crossing check indexes 0 and 1).

---

## Configuration

There is **no config file in this folder.** The code opens `./config/config.yml`
relative to the working directory. The only copy in the repo is the original at
`pre_baseline/old_stuff/Asgard_Daniel/config/config.yml`:

| Key | Example | Meaning |
|---|---|---|
| `ip`, `port` | `http://127.0.0.1`, `8081` | simulator address (overridden by the launcher) |
| `debug` | `False` | run one wrapper / one agent inline instead of fanning out |
| `network` | `False` | use radio claims instead of altitude inference |
| `smart` | `True` | enable the deconfliction logic |
| `flying_height` | `50` | scan altitude above ground |
| `speed` | `10` | takeoff speed (also sets liftoff duration) |
| `drones_speed` | `4` | cruise speed during the mission |
| `scan_degrees` | `60` | yaw sweep amplitude, sign-flipped every scan step |
| `threshold` | `8` | altitude difference that separates "peer on target" from "peer down" |
| `t_go_to_target` | `4` | seconds to descend before re-evaluating |
| `sleep_factor` | `2.2` | back-off scale when yielding a target to a peer |
| `r1`, `r2` | `10`, `30` | start-ring inner/outer radius |
| `min_deg_diff` | `60` | minimum angular separation between start positions |
| `pest_idx_to_attack` | `0` | which pest defines the start ring |
| `pest_id_to_ignore` | `'2'` | pest excluded from detection |
| `trigger_time` | `20` | seconds after launch when the synchronized run begins |
| `run_speed_factor` | `1` | divides every sleep — the sim-time acceleration knob |
| `config_path` | Unity `ConfigFiles/*Current.json` | ground-truth world layout |
| `results_path` | Unity `Results/` | where the app writes episode outcomes |

The last two are **hardcoded Windows absolute paths** for a specific developer.

---

## Results (`results.py`)

Offline scoring, run after a batch of episodes. Finds the newest folder under
`results_path`, reads `Combined Results.json`, and for each episode looks at
`drones[i].status` and `pests[i].isDetected`:

- counts a **success** when the two drones' statuses are `(1, 3)` or `(3, 1)`;
- **excludes** the episode from the denominator when no pest was detected at all;
- prints `counter/N` and the percentage.

The meaning of the status codes is not documented anywhere in this repo. Like
`PreRun`, it assumes exactly two drones and two pests.

---

## Running it

Nothing here starts the simulator — launch the Unity SwarmUI app yourself, one
instance per port, before running any of this.

```bash
# Smoke-test the API against a single running instance on :8081
.venv/bin/python -m sparx_agency.robots.ASGARD.asgard_controller_test

# Full multi-instance swarm run (expects ./config/config.yml in the CWD)
.venv/bin/python -m sparx_agency.robots.ASGARD.asgard_wrapper
```

Both currently fail — see below.

**Dependencies:** `requests` and `PyYAML` are in `requirements/`. `shapely`
(imported by `results.py`, used by `PreRun`) is **not installed** in `.venv` and
is not in any requirements file.

---

## Known problems

Verified by importing and running the modules, not inferred:

1. **Broken import — nothing but the controller loads.** `asgard_planner.py` and
   `asgard_wrapper.py` both do `from ...asgard_controller import Controller`, but
   the class was renamed to `AsgardController` in the port. `ImportError`.
2. **`geometric.py` has no import statements at all.** The refactor split it out
   of the old `utils.py` and left `math` and `numpy` behind — every method raises
   `NameError` on first use.
3. **`PreRun` in `asgard_wrapper.py` is missing its imports too** — `plt`,
   `random`, `math`, `json`, `ast`, `np`, `LineString` are all undefined.
4. **`results.py` imports `shapely`** (which it never uses) and `shapely` is not
   installed, so the module cannot be imported.
5. **`asgard_controller_test.py` calls `rotateDronePitch()`**, which does not
   exist — the methods are `rotateDroneViewPitch()` / `setPitchRotation()`.
6. **Every controller commands drone 0.** The port added
   `self.droneid = self.drone_list[0]['droneID']` to the end of `__init__`,
   which overwrites the `drone` argument the planner passes in. It also added
   `self.start()` there, so *every* construction re-`/Connect`s *all* drones.
   Multi-drone swarming cannot work as written; `pre_baseline` did neither.
7. **`connect_simulation()` reads `sys.argv[1]` and `sys.argv[2]`** as port and
   time-step. Any invocation with command-line arguments — including pytest —
   hijacks the connection target.
8. **`start()` busy-spins forever** if the simulator is not up: no timeout, no
   back-off (the `time.sleep(10)` is commented out), inside a bare `except`.
9. **`extractPosition()` silently corrupts positions.** Its regex `\d+\.\d+`
   drops minus signs (`-12.5` → `12.5`) and skips integer-valued components
   (`"(1.0, 2.0, 3)"` yields only two numbers, shifting the axes). Every distance
   comparison in the planner runs on these values.
10. **The self-restart loop targets a module that no longer exists** —
    `RunWrapper` shells out to `python -c "import wrapper; wrapper.main(...)"`,
    but the file is now `asgard_wrapper.py` inside a package.
11. **Bare `except:` everywhere**, including `main()`'s retry loop, which prints
    the ip/port and retries forever regardless of the actual error.
12. **Style diverges from the repo**: camelCase C#-style method names, `match`/
    `case` (3.10+), no docstrings on most methods, no type hints, no tests.

---

## What ASGARD does *not* give you

Relevant if you are planning to train anything against it:

- **No camera, RGB, depth, or any image stream.** Perception is the ground-truth
  `lineOfSightTargets` list. There is no visual observation to feed a policy.
- **No map, occupancy grid, obstacle geometry, or collision feedback.**
- **No episode API** — no reset, step, seed, or pause. An episode ends when the
  agent process exits and the launcher spawns a new one; randomization happens
  client-side in `PreRun`.
- **No reward signal at runtime.** Outcomes are only readable afterwards from the
  app's `Combined Results.json`.
- **No synchronous stepping.** Everything is wall-clock `time.sleep()`, scaled by
  `run_speed_factor` — timing is open-loop and non-reproducible.
- **No `core/` integration** — no `Pose2D` / `State3D` / `ControlCommand`
  conversion, no adapter, no `config/vla/` YAML, no ROS bridge.

In short: ASGARD as wired up here is a **scripted-behaviour swarm scenario with
oracle perception**, not a sensorimotor training environment.
