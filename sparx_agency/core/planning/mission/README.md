# `core/planning/mission` — mission-level target selection

Pure, ROS-free helpers for turning a *named catalog of world-placed objects* into a
concrete flight target: a detector prompt (the object's label) plus a coordinate goal
(its world `x, y`).

## What's here

| Symbol | What |
|---|---|
| `ObjectGoal` | A frozen dataclass: a named object at a fixed world position (`label`, `x`, `y`, `z`, in metres). `caption()` renders a `"label  (x, y)"` menu/log line. |
| `ObjectCatalog` | An ordered, immutable collection loaded from an objects JSON *list*. `from_json_file` / `from_json`, `labels` / `unique_labels` / `by_label`, `random`, plus `len` / iteration / indexing. |

## The file format

A top-level JSON **list**, one entry per object; extra keys (e.g. `frame_idx`,
`tag_ids`) are ignored, so a detection dump doubles as a catalog:

```json
[
  {"label": "refrigerator", "position_m": {"x": -0.98, "y": -4.12, "z": 0.48}},
  {"label": "chair",        "position_m": {"x": 0.32,  "y": -4.74, "z": 0.48}}
]
```

Positions are metres in the same world frame as the localization / `/waypoint_nav/goal`,
so `(x, y)` is published straight through as the planner's point goal. Labels are
normalised (stripped, lower-cased) on load and are **not** unique (a room can hold two
chairs), so `by_label` returns a list. Malformed input raises `ValueError` — no silent
defaults.

## Consumers

`tasks/planning/falcon/adapter/scripts/mission_director_node.py` (ROS1) loads a catalog,
picks an object (random or via a matplotlib list window), and publishes the label +
coordinate goal + enable to arm the FALCON object-approach mission. See that node and
`tasks/planning/falcon/OBJECT_APPROACH.md`.

## Constraints

ROS-free and **Python 3.8 compatible** (the FALCON Noetic adapter imports `core` under
3.8): no `match`/`case`, no `@dataclass(slots=True)`, no bare PEP 604 `X | Y` unions.
