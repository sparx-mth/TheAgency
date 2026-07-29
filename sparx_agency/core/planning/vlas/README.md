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
  navdp/            point-goal diffusion   — client, geometry, local_goal, trt/
  flownav/          image-goal flow-matching — client, preprocess, trt/
  internvla_n1/     dual-system VLN        — client, types
```

## What is shared, and what deliberately is not

Shared, because it was literally duplicated: the TensorRT engine runner, the PNG
codec, the HTTP plumbing, the error base.

**Not** shared, because the policies genuinely differ: `trt/scheduler.py` (NavDP's
numpy DDPM/DDIM with critic ranking vs FlowNav's deterministic flow-matching Euler
loop), `trt/postprocess.py`, `trt/policy.py`. Merging those would produce a
parameter soup, not reuse.

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
