# trt_optimizer — network-agnostic TensorRT optimization, with the reasoning kept

Takes any torch network and produces the fastest TensorRT deployment it honestly
can on *this* device — plus a report saying what it converted, what it refused to
convert, and what accuracy that cost. It is deliberately **not** a
"convert everything to ONNX" script: most of its value is in the components it
decides to leave alone, and in refusing to let you claim a speedup you have not
measured against a baseline.

Four near-identical TensorRT pipelines already exist in this repo
(`tasks/planning/vlas/navdp/trt/`, `.../flownav/trt/`,
`tasks/mapping/yolo_world_trt/`, and the builder inside
`tasks/common/model_registry/`). This package is the generalization of what they
have in common, plus the two things none of them do: **decide what is worth
converting from measurements**, and **verify that the precision you asked for
actually survived into the engine**.

## The two refusals

Both are structural, not advisory.

1. **Nothing is converted before it is measured.** `decide.decide()` raises on a
   plan whose components have no `latency_ms`. Deciding from parameter counts is
   how a project spends a week on the graph that was easy to export and
   discovers it was 6% of the budget.
2. **There is no "after" without a "before".** The baseline is captured in
   `plan()`, before any engine exists, and carried into the report. A report with
   no baseline cannot be produced.

## The pipeline

```
acquire ─► inspect ─► plan ──────────► export ──► build ──► bench ──► report.md
   │          │         │                 │          │  ▲       │
   │          │         ├─ dissect        │          │  │       ├─ p50/p90/p99
   │          │         ├─ profile(BEFORE)│          │  └───────┤  decision metrics
   │          │         └─ decide ◄ amdahl│          │   race   └─ gates → selected.json
   │          │                           │          │  (one build+gate per
   │          │       portable, any box ──┘          │   candidate precision)
   │          └─ toolchain + silicon + what it can actually build
   └─ clone / locate, inventory, licence. Imports nothing, installs nothing.
```

``run`` does all of it in one command; every stage is also addressable on its own.

| stage | runs where | produces | locked to |
|---|---|---|---|
| `inspect` | anywhere | what this toolchain can build | — |
| `plan` | a machine with torch + the GPU | `plan.json`, the BEFORE number, a verdict per component | — |
| `export` | anywhere torch runs (CPU is fine) | `engines/onnx/*.onnx` + `manifest.json` | portable |
| `build` | **the target device**, in the interpreter that will serve | `engines/<target_tag>/*.engine` + `.engine.json` | GPU compute capability **and** the exact TensorRT build |
| `bench` | **the target device** | `trt_report.md` / `.json`, the gate verdict | — |

## The decision rules

`decide.py` applies these in order; the first match wins, and the number that
decided it goes into the report verbatim.

| # | condition | verdict | why |
|---|---|---|---|
| 1 | cadence is `once_per_episode` / `once_per_process` | `cache_output` / `leave_in_torch` | it cannot move the steady-state rate however slow it is |
| 2 | autoregressive: KV cache, `generate()`, sampling loop | `llm_runtime` | not an ONNX graph. TensorRT-LLM's PyTorch backend on a dGPU, llama.cpp/GGUF on Orin |
| 3 | otherwise hostile to export | `leave_in_torch` | quotes the specific blocker |
| 4 | share of the decision < 5% | `leave_in_torch` | complexity and numerical risk exceed the gain |
| 5 | projected end-to-end gain < 2% | `leave_in_torch` | Amdahl, with the arithmetic shown |
| 6 | the graph is precision-sensitive | `trt_fp32` | a strongly-typed TensorRT has no per-layer FP32 fallback to rescue a deep residual stream |
| 7 | otherwise | `trt_fp16` | |

Two extra levers are emitted alongside the primary verdict when they apply:

- **`reduce_calls`** for a `per_step` component running ≥ 4 times per decision.
  Step count is the dominant term in a diffusion/flow head and costs nothing to
  change — attack it *before* the kernels, and truncate late/low-noise steps
  before early ones.
- **`cache_output`** for a `per_plan` component still costing ≥ 20% of the
  averaged budget — run it on its own thread with an explicit staleness policy
  so the control loop never blocks on it.

## What the hardware actually does

Measured on this machine, not read off a datasheet. Re-measure on yours.

- **TensorRT 11 removed weak typing.** `trt.BuilderFlag` has no `FP16`, `INT8`,
  `BF16` or any precision-constraint member, `ILayer.precision` and
  `set_output_type` are gone, and every `IInt8Calibrator` class is gone. Engine
  precision is *exactly* what the ONNX carried. `builder.create_network(0)`
  already reports `STRONGLY_TYPED == True`.
  → so `engine/precision.py` bakes precision into the graph, and
  `engine/build.py` verifies it survived by measuring weight bytes per parameter
  against `EngineStat.TOTAL_WEIGHTS_SIZE`. **A successful build proves nothing.**
- **DLA is dead in TensorRT 11.** `Runtime.num_DLA_cores` returns 0, even though
  `DeviceType.DLA`, `GPU_FALLBACK` and `MemoryPoolType.DLA_*` all still exist in
  the bindings. TensorRT 10.7 was the last release supporting it. `engine/dla.py`
  gates on the version *and* a live core count, never on `hasattr`.
- **On GeForce Blackwell (sm_120), INT8 is the only quantized format that pays.**
  A 2048³ GEMM measured FP16 75.5 TFLOP/s, FP8 82.8 (1.10×), INT8 149.5 (1.98×),
  NVFP4 68.6 — **NVFP4 was slower than FP16**. `decide.precision_ladder()` orders
  formats accordingly and demotes FP8/NVFP4 to "benchmark before believing".
- **Both builder memory pools default to ~100% of the device** (8,080,064,512
  bytes measured on an 8 GB card). Capping `WORKSPACE` alone — which every older
  builder in this repo does — does not bound the peak, because `TACTIC_DRAM` is
  what spikes during tactic evaluation. On a Jetson `TACTIC_DRAM` defaults to 75%
  of *unified* memory. `engine/builder_config.py` caps both.
- **Clocks are not lockable here without root** and `Applications Clocks` report
  N/A, so the SM clock floats between 457 MHz idle and 3090 MHz boost. Absolute
  Hz numbers are comparable only within one interleaved run; `bench/latency.py`
  says so in every report and `drift_check()` catches thermal throttle.

## Adding a network

Implement `adapter.ModelAdapter` in that network's own package — under
`tasks/planning/vlas/<vla>/trt/` for a VLA, beside the model otherwise — and
register a factory. The generic pipeline needs seven answers:

| method | what it must say |
|---|---|
| `load` | build the reference torch model, in eval mode |
| `cadences` | component name → `Cadence`. A starting point: the profiler counts real calls and overrides you |
| `graphs` | one `GraphSpec` per engine. **Split where cadence changes**, not where the source has a class |
| `wrappers` | engine key → an `nn.Module` taking that graph's inputs positionally |
| `patch` | this model's own export fixes (bake the pos-embed, delete a no-op CFG branch) |
| `scenarios` / `run_reference` / `run_engines` | one full decision, both ways |
| `decision_metrics` + `gates` | **the important one** — see below |

### `decision_metrics` is the method that matters

There is no default implementation, on purpose. A generic tool can compute
relative L2 between two tensors; that number is nearly worthless for a robot —
published work finds raw action MSE correlates about −0.61 (Spearman) with
rollout success, versus −0.87 for a task-aware variant. Gate on what changes the
aircraft's behaviour: did the selected trajectory flip, did the stop decision
flip, did the commanded heading move past the controller's deadband, did a
waypoint cross an obstacle. Report tensor L2 too — as a diagnostic, never as the
gated quantity.

## Runbook

Build and run in the interpreter that will serve the engine. On this machine
that is the `navdp` conda env, **not** `.venv` (which has no torch, tensorrt,
pycuda or onnx).

The example below uses the shipped image-classifier adapter, so it runs with no
checkpoint and no download. Swap `--adapter-module` / `--adapter` for your own.

```bash
cd <repo root>
CONDA=~/miniconda3/envs/navdp/bin/python
TRT="$CONDA -m sparx_agency.tasks.common.trt_optimizer"
A="--adapter-module sparx_agency.tasks.common.trt_optimizer.adapters.image_classifier \
   --adapter image_classifier"

# 0. what can this machine build at all?
$TRT inspect

# 1. get the model and inventory it (clones nothing you did not ask for,
#    imports nothing, installs nothing)
$TRT acquire https://github.com/<org>/<repo>

# 2. everything: plan -> export -> build -> race every precision -> gate -> report
$CONDA -m sparx_agency.tasks.common.trt_optimizer $A run \
    --ckpt none --out-dir /tmp/cls --scenarios 16

# ...or drive the stages separately
$CONDA -m sparx_agency.tasks.common.trt_optimizer $A plan  --ckpt none --out /tmp/plan.json
$TRT budget --plan /tmp/plan.json --precision fp16
$CONDA -m sparx_agency.tasks.common.trt_optimizer $A export --ckpt none --out-dir /tmp/cls/onnx
$TRT build --onnx-dir /tmp/cls/onnx --out-dir /tmp/cls/$TAG --precision fp16
$CONDA -m sparx_agency.tasks.common.trt_optimizer $A bench \
    --ckpt none --onnx-dir /tmp/cls/onnx --engine-dir /tmp/cls/$TAG

# 3. should any of it go on the Jetson's DLA? (asks the graph, not you)
$TRT dla --onnx /tmp/cls/onnx/<graph>.onnx
```

Note the flag order: `--adapter` / `--adapter-module` are global and come
**before** the subcommand.

`run` exits non-zero when no precision passes its gate, and withdraws
`selected.json` so nothing unblessed can be served.

## Reference adapters — copy one of these

`adapters/` ships two working `ModelAdapter` implementations, and neither is a
navigation model. They exist to be copied, and to prove the contract is not
shaped around any one task:

| adapter | model | what "the same answer" means |
|---|---|---|
| `adapters/image_classifier.py` | any torchvision classifier | top-1 / top-5 agreement |
| `adapters/segmentation.py` | torchvision segmentation | pixel agreement + mean IoU |

A third, non-toy one lives with its network:
`tasks/planning/vlas/internvla_n1/trt/adapter.py` optimizes a dual-system VLA's
System 1, and gates on the **discrete action queue** the robot executes rather
than on anything about the trajectory tensors it came from.

That difference is the whole point. `decision_metrics` has no default
implementation because only the network's author can say what a changed answer
is — and it is never tensor L2. See `adapters/README.md` for the per-family
table (detection, depth, pose, ASR, tracking, generative).

## Dynamic shapes

Prefer a static graph: it is faster and has no profile-switch cost in its tail.
When an input genuinely varies at run time — a detector fed whatever resolution
the camera produced, a batch sized by how many candidates survived a filter —
declare a `ShapeProfile(min, opt, max)` on the `GraphSpec`. The exporter then
declares exactly those axes free, the op gate stops demanding static shapes for
that graph, and the builder attaches a matching TensorRT optimization profile.

`opt` is the shape tactics are tuned for: make it the common case, and keep the
range no wider than the input actually varies.

Dynamic engines are executed by `engine/runner.py`. The deployment runtime in
`core/` deliberately refuses them — a control loop wants deterministic latency —
so that choice stays yours.

## INT8 and below

There are two genuinely different routes and `engine/calibrate.py` owns both:

- **TensorRT ≤ 10** (the Orin's JetPack stack) still has
  `IInt8EntropyCalibrator2`. Collect in-domain arrays with
  `collect_calibration_arrays()`, build a calibrator with
  `make_entropy_calibrator()`, pass it through `pipeline.build(calibrators=...)`.
- **TensorRT ≥ 11** (this machine) removed every calibrator class. INT8 must
  arrive as QuantizeLinear/DequantizeLinear nodes already in the ONNX, which
  needs `nvidia-modelopt` — not installed here. `qdq_instructions()` prints the
  exact recipe.

`build_engine` calls `require_int8_buildable()` as a **pre-flight**, because an
un-quantized graph parses perfectly and builds a valid engine at the wrong
precision. Calibrate on real data: 128–512 samples, in-domain only.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -m pytest sparx_agency/tasks/common/trt_optimizer/tests -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  ~/miniconda3/envs/navdp/bin/python -m pytest sparx_agency/tasks/common/trt_optimizer/tests -q
```

Both are meaningful and neither is redundant: `.venv` proves the pure modules
import and work with **no** torch/tensorrt/onnx present, which is what lets a
plan be inspected on a machine that cannot build. The conda run adds the torch,
ONNX and TensorRT paths, including a real end-to-end engine build.

**`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is required for BOTH, and omitting it on
`.venv` produces a false green** — not an error. The ROS2 plugins under
`/opt/ros/jazzy` are autoloaded and eagerly import every collected module, so the
first module-level `pytest.importorskip("torch")` aborts the whole directory
sweep. The run then prints `1 skipped` and **exits 0** while collecting none of
the 600 tests. In the conda env the same autoload instead crashes collection on a
missing `lark`. Read the count, not the exit code.

A passing run in this repo can still exit non-zero — read the summary line,
never `$?`.

## Two behaviours you will see in the build notes

Both were found by building real engines on this machine, and both are recorded
in every engine's sidecar `.json` so a later reader knows what actually happened.

- **The FP16 conversion ladder.** Keeping `LayerNormalization` / `Softmax` /
  `ReduceMean` in FP32 is better numerically, and it is what this repo's shared
  `fp16_onnx` helper does. On a transformer graph it can also leave a boundary
  the TensorRT parser refuses outright — measured on a 4-layer
  `nn.TransformerEncoder`: the attention in-projection `Gemm` comes out Half
  while the bias TensorRT broadcasts against it stays Float, and the parse dies
  with *"ElementWiseOperation SUM must have same input types"*. The older
  builders here work around that by building the whole graph FP32 (NavDP's
  `strongly_typed_fp32_engines`). This one walks down `precision.FP16_LADDER`
  instead, keeps FP16, and records which rung it landed on. You will see the
  failed first attempt in the TensorRT log; that is expected, and the note says so.
- **The Blackwell optimization-level clamp.** TensorRT 11.1 on sm_120
  **segfaults** intermittently while building a non-FP32 graph at
  `builder_optimization_level >= 2` — measured at roughly one run in three at
  level 3, never at level 0 or 1, and never for the same graph at FP32.
  Reproduced with a bare `IBuilderConfig` carrying none of this package's knobs,
  so it is TensorRT's bug, not ours. A segfault cannot be caught, so
  `builder_config.safe_optimization_level()` avoids the combination rather than
  retrying it, and says so in the notes. Raise it deliberately once TensorRT
  fixes this.

## Four defects this package had, found by optimizing a real network

All four were silent -- the pipeline kept running and produced a plausible
artifact -- and three of them would have shipped a wrong engine. Regression tests
are in `tests/test_regressions.py` and `tests/test_fp16_graph_validation.py`.

- **An invalid FP16 conversion reached the builder.**
  `onnxconverter_common.convert_float_to_float16` does not raise when a blocked
  op leaves a mistyped cast behind: it returns a graph whose declared tensor
  types disagree with its ops. onnxruntime refuses to load such a graph;
  **TensorRT parses it and builds a silently wrong engine** -- measured at 1.0e-1
  relative L2 against 3.0e-4 for the same graph one rung down. `to_fp16_onnx`
  now runs `onnx.checker.check_model(full_check=True)` and raises
  `Fp16ConversionInvalid`, which the ladder walk treats as a rung failure.
- **The FP16 ladder had no middle rung.** Blocking `Softmax`/`ReduceMean` is not
  merely *stricter* than blocking `LayerNormalization` alone: it changes the cast
  topology the converter has to repair, and on a transformer graph it repairs it
  wrongly. `FP16_LADDER` gained `("LayerNormalization",)`, which measured both
  valid and **more accurate** than the rung above it.
- **`precision_sensitive` was declared, documented and recorded -- and nothing
  read it.** `pipeline.build` now builds such a graph FP32 whatever precision was
  requested (from the GraphSpec, or the manifest when no specs are passed), which
  is the whole point of the flag on a strongly-typed TensorRT where precision is
  per engine. `_engine_files` follows it: a selection at FP16 falls back to the
  FP32 build for a pinned key, because a selection missing an engine points the
  runtime at a pipeline it cannot complete.
- **`plan` handed synthetic components to the profiler.** `dissect` emits a
  `<parent>.other` bucket to keep the parameter total exact; it is a sum over the
  tree with no forward to hook, and `profile_components` rightly raises on a name
  it cannot resolve. `dissect.is_synthetic()` names them and `plan` skips them.

Three smaller gaps closed at the same time: `export/parity.py` (tiers (a) and (b)
were documented as mandatory and had no implementation), `pipeline.load_plan`
plus `bench --plan` (the staged path produced a report with no component table
and no "deliberately not converted" section), per-graph `params` in the manifest
so precision verification happens by default, and `plan --max-depth`, because the
default of 2 puts the component frontier on `nn.ModuleList` containers for most
transformer stacks and returns an unprofiled plan.

## Traps

- **`.gitignore` swallows the output.** `*.onnx`, `*.engine`, `*.plan`,
  `*.calib` and `*.cache` are ignored globally, and
  `sparx_agency/tasks/planning/vlas/*/trt/engines/` is ignored wholesale — but
  that glob covers *only* that path. A new network's `engines/` elsewhere in the
  tree is **not** covered and will show up in `git status`. Add a rule when you
  add a network. A clean `git status` after a full build is expected.
- **Reports are build output too.** Written into the engine directory they are
  gitignored; pass a tracked directory to `write_report` if a result must
  survive a clean checkout.
- **torch 2.9+ changed the default ONNX exporter.** `onnx_export.py` pins
  `dynamo=False`, because the `torch.export` backend produces a different graph
  and every tolerance and op-gate rule here is calibrated against the TorchScript
  one.
- **Opset 17 is a floor, not a preference.** It is the first opset emitting a
  single `LayerNormalization` node rather than a decomposed
  ReduceMean/Sub/Pow/Div chain, which alone fixes most transformer FP16 drift.
- **Do not build engines on a Jetson at 15 W.** Tactic timing takes hours and
  `TACTIC_DRAM` fights the rest of the stack. Build at MAXN with `jetson_clocks`,
  keep the timing cache, then switch to 15 W to fly.
- **`onnxslim` aborts on aarch64.** It invokes onnxruntime, whose CPU-feature
  detection SIGABRTs there, and a native abort cannot be caught. Pass
  `--no-slim` on a Jetson.
