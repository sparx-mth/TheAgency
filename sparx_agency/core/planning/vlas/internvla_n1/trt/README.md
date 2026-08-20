# InternVLA-N1 System 1 — TensorRT deployment runtime

What flies. Numpy-only at import, Python-3.8 clean, no torch and no diffusers:
TensorRT and pycuda are lazy-imported by the shared
`core/planning/vlas/common/trt/engine_runner.py`. The build tooling lives in
`tasks/planning/vlas/internvla_n1/trt/` and is not imported from here.

## What it replaces

The three heavy forwards of `InternVLAN1ForCausalLM.generate_traj`
(`system1 = "nextdit_async"`). Everything stochastic or data-dependent — the
flow-matching schedule, the candidate fan-out, the trajectory integration, the
walk into discrete actions — stays in numpy, so a difference between this and the
torch reference is attributable to the engines and to nothing else.

```
images (1,2,224,224,3)  --vision engine-->     dino_feat (1,512,384)
  + traj_latents (1,4,3584) from System 2
                        --condition engine-->  condition (1,36,768)
  broadcast to 32 candidates, uploaded once, resident for all ten steps
                        --denoise engine x10-> trajectories (32,32,3)
  postprocess.action_queue                  -> at most 4 discrete actions
```

## Usage

```python
from sparx_agency.core.planning.vlas.internvla_n1.trt.policy import InternVLAN1System1TRT

policy = InternVLAN1System1TRT(engine_dir, samples=32, steps=10)
actions, trajectories = policy.predict_actions(images, traj_latents)
# actions: up to 4 of {1 forward, 2 turn left, 3 turn right}; the agent
# executes actions[0] and queues the rest.
```

`images` is `(1, 2, 224, 224, 3)` float32 in `[0, 1]` — the pixel-goal frame
first and the current frame second, exactly as `internvla_n1_agent.step` stacks
them. `traj_latents` is `(1, 4, 3584)` from System 2's `generate_latents`.

## Three things worth knowing before changing it

- **`steps` is a runtime knob.** The denoise engine is one Euler step, not ten
  unrolled, so the count changes without a rebuild. It is also the largest lever
  available — the loop is 94% of a call — and it **changes behaviour**:
  `benchmark/step_count.py` measures the trade, and 8 steps already flips the
  executed action in 1 scenario of 16.
- **`samples` is not a knob.** The engines are built static at 32 candidates.
  It is also nearly free: the denoise step is launch-bound at these shapes, so
  batch 1 and batch 32 cost within 2% of each other.
- **The engines may be mixed precision.** `selected.json` names one engine per
  graph, each carrying its own — TensorRT 11 is strongly typed, so precision is a
  property of each engine independently. The condition graph is pinned FP32
  because TensorRT miscompiles it in FP16 on sm_120; the build package's README
  has the measurement.

**A missing `selected.json` is the fail-safe working**, not a lookup to route
around: the precision race withdraws the selection when nothing passed its
quality gate, so its absence means nothing was blessed.

Single-robot only: the engines are static at 32 candidates, so a second robot
needs its own process.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
    sparx_agency/core/planning/vlas/internvla_n1 -q
```

They run in `.venv`, which has no torch — which is what proves the import
contract rather than merely asserting it.
