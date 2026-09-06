# `sim_overlay` — the sim files this repo owns

Files here are **TheAgency's authoritative copies** of files that belong, on
disk, to the external `sjtu_project` checkout. `bringup_world.sh` mounts this
directory into the world container and copies them over the checkout's own
copies on every bring-up, before anything is built or launched.

**Why this exists.** The spawn pose has to be settable per run — a search
campaign that starts every trial in the same corner measures the route out of
that corner more than it measures the search. The stock `spawn_drone.py`
hard-codes `(1.0, 1.0, 2.0)` and reads only `argv[1]` (the URDF) and `argv[2]`
(the namespace), so making it settable means editing a file in a repo that is
not this one.

Editing `sjtu_project` in place was how this worked first, and it is the wrong
answer: the edit is invisible to this repo's history, a fresh clone of
`sjtu_project` silently reverts the behaviour, and the change is lost the next
time someone rebuilds that workspace. Nothing about a run would look wrong —
every trial would simply start in the same corner again, which is exactly the
failure the change exists to prevent.

So the copies here are the source of truth, and `sjtu_project` is treated as a
checkout to be overlaid rather than a repo to be edited.

## What is overlaid

| this repo | the checkout |
|---|---|
| `sjtu_drone_bringup/sjtu_drone_bringup/spawn_drone.py` | `sjtu_drone/sjtu_drone_bringup/sjtu_drone_bringup/spawn_drone.py` |
| `sjtu_drone_bringup/launch/sjtu_drone_gazebo.launch.py` | `sjtu_drone/sjtu_drone_bringup/launch/sjtu_drone_gazebo.launch.py` |

`spawn_drone.py` takes an optional third argument, `"x,y,z"` or `"x,y,z,yaw"`;
the launch file passes it from the `SJTU_DRONE_SPAWN` environment variable,
which `bringup_world.sh` forwards into the container. **Unset spawns exactly
where it always did**, so every existing caller is unaffected and the overlay
is a pure extension.

## Both the source tree and the install tree

`bringup_world.sh` copies into **both**, and needs to:

* without `--skip-build`, `colcon build` compiles `sjtu_drone_bringup` from the
  SOURCE tree, so an overlay that only touched `install/` would be overwritten
  by the build;
* with `--skip-build` — which `run_scene_graph.sh` always passes — nothing is
  built and only `install/` is read, so an overlay that only touched the source
  tree would never take effect.

Copied rather than bind-mounted for the same reason: a read-only bind mount
over `install/` makes `colcon build` fail when someone drops `--skip-build`,
and a writable one lets the build silently replace this repo's copy with the
checkout's.

## Adding a file

Put it here under its package-relative path, add a row to the table above, and
add it to `OVERLAY_FILES` in `bringup_world.sh`. The copy is idempotent and
runs on every bring-up, so a stale checkout repairs itself.

## What this does NOT do

It does not vendor the simulator. The worlds, meshes, drone description and
built workspace still live in `sjtu_project`, and `SJTU_PROJECT_DIR` is still
required. What it removes is the need for anyone to hand-edit that checkout:
clone it fresh, and this repo puts its own behaviour back on every run.
