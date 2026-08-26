# `core/planning/vlas` — the ROS-free half of every learned policy

The algorithmic core of each VLA: the wire contract, the pure geometry, and the
numpy TensorRT runtime. No ROS, no torch, no robot names.

The executable half — building engines, fine-tuning, serving, ROS nodes — lives
in [`tasks/planning/vlas/`](../../../tasks/planning/vlas/README.md), which also
documents the layout, the per-policy table, and how to add a policy or a platform.

## Two hard rules

Both exist because the FALCON ROS1 adapter imports this package inside a Noetic
container with **Python 3.8**, numpy, and essentially nothing else. Break either
and the XTEND flight stack dies at node startup, on the drone.

1. **No eager heavy imports.** `torch` / `tensorrt` / `pycuda` / `cv2` /
   `requests` / `PIL` are imported *inside methods*, never at module scope.
2. **Python 3.8 syntax.** No `match`/`case`, no PEP 604 `X | Y` in a module
   without `from __future__ import annotations`, no `@dataclass(slots=True)`.

These are enforced, not just documented — see
[`common/tests/test_core_import_contract.py`](common/tests/test_core_import_contract.py),
which imports every policy in a fresh interpreter and AST-scans the whole tree.

## Layout

```
vlas/
  interfaces/       NavigationPolicy (ABC) + PolicyObservation/PolicyResult
                    goals.py: PointGoal / ImageGoal / LanguageGoal / PoseGoal
  registry.py       name -> lazy policy factory
  common/           VLA-agnostic runtime
    errors.py       VlaError, the base every policy error derives from
    image_codec.py  RGB/uint16-depth PNG — the wire format every server speaks
    http_client.py  HttpPolicyClient: URL, timeout, logging, POST, trajectory decode
    trt/            the single TRTEngineRunner (was duplicated per policy)
  navdp/            point-goal diffusion   — client, geometry, local_goal,
                    preprocess, trt/
  flownav/          image-goal flow-matching — client, preprocess, trt/
  internvla_n1/     dual-system VLN        — client, types, errors, trt/
```

## Where a shared thing goes — three rings, not one

`common` is not one place. Something used by every subsystem belongs in the
widest ring; something only the VLAs speak belongs in this package's own
`common/`; something only one policy needs stays in that policy's directory.
Putting a VLA-specific contract in `core/common` is as wrong as leaving universal
arithmetic inside a policy.

| ring | holds | example |
|---|---|---|
| `core/common/` | unambiguous universal maths and vocabulary | `math/se2.py` — the body↔world pair |
| `core/planning/vlas/common/` | what every *policy* needs and nothing else does | `http_client.py`, `plan_commit/`, `image_codec.py` |
| `core/planning/vlas/<vla>/` | one policy's own contract and geometry | `navdp/preprocess.py` |

The SE(2) body↔world transform is the worked example. It had two implementations
in this package — a Python loop in `navdp/geometry.py` and a numpy stack in
`common/plan_commit/` — and the tempting fix was to share it here. It went to
[`core/common/math/se2.py`](../../common/math/se2.py) instead, because rotating a
point by a yaw is not a VLA concept: the followers, the mapping BEV and the
localization filters all do the same arithmetic. `navdp/geometry.py` keeps its
own names and docstrings (FALCON imports those symbols by name) and delegates the
maths.

## What is shared, and what deliberately is not

Shared, because it was literally duplicated: the TensorRT engine runner, the PNG
codec, the HTTP plumbing, the error base.

Shared for a third reason — **the tooling and the aircraft must agree**:
`navdp/preprocess.py` and `flownav/preprocess.py`. Preprocessing is not a detail
of one caller. If a fine-tune pipeline resizes or orders channels differently
from the server that flies the result, nothing fails: the loss still falls and
the policy is quietly worse. NavDP's copy lived twice under `tasks/` and the two
disagreed on colour order; it belongs next to the runtime everything drives.

Shared for the opposite reason — because it would otherwise be duplicated the
moment a second policy flies: [`common/plan_commit/`](common/plan_commit/README.md),
which commits to one prediction and flies it as a route before asking for
another. Every policy here answers per frame and every aircraft flies a route;
the piece between the two belongs to neither, so it lives here. Do not write a
second one inside a policy directory.

### The HTTP client, and why N1 took so long to join it

`HttpPolicyClient` holds URL, timeout, the lazy `requests` import, logging and
POST. NavDP and FlowNav subclassed it from the start; InternVLA-N1 carried its
own copy of all five for months, and the copy is where two flight bugs lived (a
logger that killed the node on its first warning, and a re-init that asked for
the wrong agent name forever). It now subclasses too, which needed two honest
additions rather than a copy-paste:

* **`_attempt()` / `Attempt`** — `_post()` collapses everything that is not a 200
  into `None`, which is right for a policy step and wrong for a *session* route:
  N1 must create its agent before stepping it, and 201, 409 and a timeout mean
  three different things to `/agent/init`.
* **`_say(level, msg)` and opt-in pooling** — ROS1 clients pass a bare callable
  (`rospy.logwarn`); ROS2 clients pass a logger object needing severity dispatch,
  with **one call site per severity**, because `rclpy` caches a severity per call
  site and raises on the second one. Pooling is opt-in because turning on
  connection reuse under the ROS1 flight path is a change nobody asked for.

**Not** shared, because the policies genuinely differ: `trt/scheduler.py` (NavDP's
numpy DDPM/DDIM with critic ranking vs FlowNav's deterministic flow-matching Euler
loop), `trt/postprocess.py`, `trt/policy.py`. Merging those would produce a
parameter soup, not reuse.

## What is here but not yet flown

`internvla_n1/trt/` is a complete, tested System-1 runtime that **no consumer
imports today**: converting System 1 alone moves the end-to-end pipeline by ~4%,
because System 2 is 98.5% of a call, so the ROS2 node still drives the HTTP
server. It is kept rather than deleted because the measurement, the parity
harness and the mixed-precision selection are the expensive parts and they are
done — see [`internvla_n1/trt/README.md`](internvla_n1/trt/README.md). Do not
read its presence as "this is what flies".

## Two ways to call a policy

**Natively** — what the FALCON ROS1 nodes do. Full policy-shaped API:

```python
from sparx_agency.core.planning.vlas.navdp import NavDPPointgoalClient
client = NavDPPointgoalClient("http://127.0.0.1:8888")
client.reset(intrinsics)
traj = client.best_trajectory(client.pointgoal_step(rgb, depth, gx, gy))
```

**Uniformly** — for a caller that should not care which policy it is driving (an
arbiter, a benchmark, a new robot's runner):

```python
from sparx_agency.core.planning.vlas.registry import default_vla_registry
from sparx_agency.core.planning.vlas.interfaces import PointGoal, PolicyObservation

policy = default_vla_registry().create("navdp", url="http://127.0.0.1:8888")
result = policy.step(PolicyObservation(rgb=rgb, depth_m=depth, intrinsics=K),
                     PointGoal(forward_m=2.0, left_m=0.5))
if result.ok:
    fly(result.trajectory)
```

Both are supported on purpose. The native clients keep their exact behaviour
because a live flight stack depends on it; the uniform contract is the layer on
top.

## Errors: drop vs malformed

A **transport drop** returns `None` (native) / a not-`ok` result (uniform) and is
logged. At video rate over loopback a dropped frame is routine; the caller
re-sends next frame.

A **malformed response** raises. It arrived but cannot be flown, and silently
returning "no result" would let the caller keep flying a stale path.

Every policy error derives from `VlaError` (and still from `RuntimeError`), so an
arbiter driving several policies can catch them together while
`except NavDPError` keeps catching exactly what it always did.
