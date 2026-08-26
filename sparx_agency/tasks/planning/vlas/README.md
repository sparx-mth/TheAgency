# VLAs — vision-language-action / learned navigation policies

Every learned policy that **outputs an action or a trajectory** lives here, split
across three layers. That split is the whole design; everything else follows from
it.

```
core/planning/vlas/<vla>/     the policy.   ROS-free, numpy-only at import, py3.8.
tasks/planning/vlas/<vla>/    the work.     TRT build, fine-tune, serve, ROS nodes.
robots/<PLATFORM>/            the robot.    topics, intrinsics, actuation.
```

**A policy never names a robot, and a robot never names a policy.** They meet in a
YAML under `robots/<PLATFORM>/config/vla/`. That is what makes "run FlowNav on the
Robotican instead of the XTEND" a config change rather than a port.

## What counts as a VLA

A learned policy whose output is an **action or trajectory**. Perception
backbones — DepthAnything, YOLO-World, NanoOWL, VILA — output depth, boxes or
labels, so they are **not** VLAs and stay under `core/mapping/`. Classical
planners (A*, RRT*, pure pursuit) stay in their own `core/planning` packages.
Write this rule down once and stop re-litigating it per model.

## The policies today

| policy | goal | integration style | trt | finetune | ros | platforms |
|---|---|---|---|---|---|---|
| **navdp** | point | HTTP server (host) | ✅ built | ✅ | ROS1 (FALCON) | XTEND, Sphera |
| **flownav** | image | HTTP server (host) | ✅ built | ✅ | ROS1 (FALCON) | XTEND |
| **internvla_n1** | language | HTTP server (external repo) | ⚠️ S1 only | — | ROS2 | Rooster R1 / Sphera, SJTU sim |
| **omnivla** | language / pose / image | **in-process torch** | — | — | ROS2 | Rooster R1 / Sphera |
| **nomad** | (exploration) | **external process**, waypoints over a topic | — | — | ROS2 | Rooster R1 / Sphera |

**What is actually flown:** navdp (XTEND, via FALCON) and internvla_n1 (SJTU sim,
via `tasks/planning/sjtu_internvla_n1/`). flownav is built but unflown; omnivla
and nomad are parked earlier work, kept deliberately and held to a lower
standard — they have no `core/` layer, are not in the registry, and their ROS2
bridges import a robot adapter directly, which the layering forbids. Do not copy
them as a pattern, and do not treat their state as a to-do list.

⚠️ internvla_n1's TensorRT path converts **System 1 only** and is not wired into
any node: System 2 is 98.5% of a call, so it moves the pipeline ~4%. See
`core/planning/vlas/internvla_n1/trt/README.md`.

Three different integration styles is not an accident to be cleaned up — it is
what the upstream projects actually are. The contract in
`core/planning/vlas/interfaces/policy.py` is what makes them interchangeable to a
caller anyway.

Empty `trt/` and `finetune/` directories are **not** created for policies that do
not have them. When you need one, mirror `navdp/`.

## Layout

```
tasks/planning/vlas/
  common/                       shared by every policy — only genuinely identical code
    hardware/detect.py          GPU/Jetson detection -> the engines/<tag> dir name
    engine/fp16_onnx.py         (was byte-identical in navdp and flownav)
    finetune/                   the model-agnostic training machinery
      common/                   ESDF targets, label encoding, augmentation, EMA, L2-SP
      datasets/                 flight-recording schema, bag extraction, torch Dataset
  <vla>/
    trt/                        ONNX export -> engine build -> benchmark
      configs/build_policy.json build-time knobs
      export/ engine/ benchmark/
      engines/<hardware_tag>/   BUILD OUTPUT — gitignored, per-device, never portable
    finetune/
      configs/<vla>_finetune.yaml
      finetune_model.py loss.py train.py
    serve/
      configs/                  RUNTIME knobs (NavDP's sampler lives here)
      <vla>_trt_server.py       the HTTP server the ROS nodes talk to
    ros2/                       ROS2 node wrappers
    upstream/                   files we inject into a third-party repo (NOT importable)
```

### Why `serve/configs/` and `trt/configs/` are separate

`trt/configs/build_policy.json` decides how an engine is **built**.
`serve/configs/inference_speed.yaml` decides what the drone **flies** — it is
where NavDP's `sampler: ddim, num_inference_steps: 4` lives. Losing that file
silently reverts the server to the 10-step DDPM baseline, which is a
flight-behaviour change, not a config nit. Keeping them apart makes it obvious
which one you are editing.

## Adding a new VLA

1. `core/planning/vlas/<name>/` — `client.py` (the wire contract) and any pure
   geometry/decoding, **including the preprocessing** (`preprocess.py`): the
   fine-tune, the offline tools and the server must all resize and order channels
   identically, and the only way to guarantee that is one implementation. Subclass
   `common/http_client.HttpPolicyClient` and use `common/image_codec.py`; define
   `<name>Error(VlaError)` in `errors.py`.
   **Numpy-only at import, Python 3.8 syntax** — `core/planning/vlas/common/tests/
   test_core_import_contract.py` will fail you otherwise.
2. `core/planning/vlas/<name>/policy.py` — ~40 lines implementing
   `NavigationPolicy`, plus one entry in `core/planning/vlas/registry.py`.
3. `tasks/planning/vlas/<name>/` — add only the sub-areas you actually have.
   Reuse `common/hardware/detect.py` and `common/engine/fp16_onnx.py` in `trt/`;
   reuse `common/finetune/{common,datasets}` in `finetune/`.
4. Add `<name>` to the table above.

Nothing under `robots/` changes.

## Adding a platform to an existing VLA

1. Copy the nearest `robots/<PLATFORM>/config/vla/*.yaml` and retarget the topics.
2. Only if the actuation protocol differs from an existing one, add an adapter
   next to `robots/ROBOTICAN/adapters/rooster_manual_control.py`.

Nothing under `core/planning/vlas/` or `tasks/planning/vlas/<name>/` changes. This
is deliberately untested for platforms that do not exist yet — the structure is
ready, the implementations are not written in advance.

## Running the tests

```bash
.venv/bin/python -m pytest \
  sparx_agency/core/planning/vlas sparx_agency/tasks/planning/vlas \
  sparx_agency/robots/ROBOTICAN/adapters/tests
```

`tests/test_server_contract.py` (both VLAs) needs `flask`, which is not in the
project venv; it is skipped by `--ignore` or runs in the model conda envs.

## Where the ROS1 nodes are

FALCON's five VLA consumers — `navdp_click_node.py`, `flownav_node.py`,
`astar_navdp_fallback_node.py`, `combination_planner_node.py`,
`hybrid_planner_node.py` — stay in `tasks/planning/falcon/adapter/scripts/`.
They are FALCON task glue (BEV, A* arbitration, `thinking.py`), not generic VLA
runners, and `run_falcon.sh` bind-mounts them by exact filename. They import the
policies from `core/planning/vlas/` like any other consumer.
