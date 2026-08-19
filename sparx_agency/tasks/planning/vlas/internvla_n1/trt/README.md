# InternVLA-N1 System 1 — TensorRT build tooling

Build-time only. The runtime that flies is
`sparx_agency/core/planning/vlas/internvla_n1/trt/`, which imports nothing from
here and needs neither torch nor diffusers.

## What this converts, and what it refuses to

InternVLA-N1 is a dual-system policy. The released `InternVLA-N1-DualVLN`
checkpoint is **16.77 GB of bf16 weights**, and they split like this:

| | params | bf16 | share |
|---|---:|---:|---:|
| **System 2** — Qwen2.5-VL-7B, autoregressive | 8292 M | 16.58 GB | **98.9%** |
| **System 1** — DINOv2-S + MemoryEncoder + QFormer + NextDiT | 91.4 M | 0.18 GB | 1.1% |

**System 2 is not converted, and cannot be.** It generates behind a KV cache, so
it is an LLM-runtime problem rather than an ONNX graph — and its 16.58 GB do not
fit on an 8 GB card at any width above 4 bits. `trt/adapter.py` therefore
describes System 1 alone. See the report for the whole-pipeline arithmetic that
follows from that, which is the most important number in this directory.

`model.py` builds System 1 standalone, loading only its 183 MB of tensors out of
the 16.77 GB checkpoint, so the build runs on any machine.

## The three engines

Split where the **cadence** changes, not where the source has a class:

| engine | runs | inputs → outputs | precision |
|---|---|---|---|
| `internvla_n1_s1_vision` | once per call | `images (1,2,224,224,3)` → `dino_feat (1,512,384)` | FP16 |
| `internvla_n1_s1_condition` | once per call | `dino_feat`, `traj_latents (1,4,3584)` → `condition (1,36,768)` | **FP32, pinned** |
| `internvla_n1_s1_denoise` | **ten times** per call | `latents (32,32,3)`, `timestep (32,)`, `condition (32,36,768)` → `velocity (32,32,3)` | FP16 |

The denoiser is **one** Euler step, not ten unrolled: unrolling would multiply
build time and engine size and freeze the step count into the engine, and the
step count is the cheapest behavioural lever this policy has.

`condition` is pinned FP32 by a measurement, not a preference — see
**Negative results** below.

## Runbook

Everything runs in an interpreter with torch and TensorRT (`~/miniconda3/envs/navdp`
on this machine, not `.venv`). The InternNav source must be present; point
`$INTERNNAV_HOME` at it or let `upstream.py` find `~/trt/internnav/code`.

```bash
cd <repo root>
CONDA=~/miniconda3/envs/navdp/bin/python
A="--adapter-module sparx_agency.tasks.planning.vlas.internvla_n1.trt.adapter \
   --adapter internvla_n1_s1"
CK=~/trt/internnav/weights/InternVLA-N1-DualVLN
E=sparx_agency/tasks/planning/vlas/internvla_n1/trt/engines

# clone the upstream source (installs nothing, imports nothing)
$CONDA -m sparx_agency.tasks.common.trt_optimizer acquire \
    https://github.com/InternRobotics/InternNav

# measure first, decide second, convert third
$CONDA -m sparx_agency.tasks.common.trt_optimizer $A plan \
    --ckpt $CK --max-depth 1 --out ~/trt/internnav/plan.json
$CONDA -m sparx_agency.tasks.common.trt_optimizer $A export --ckpt $CK --out-dir $E/onnx

# tiers (a) and (b): FP32 on CPU, before anything is built
$CONDA -m sparx_agency.tasks.planning.vlas.internvla_n1.trt.validate_parity \
    --ckpt $CK --onnx-dir $E/onnx

# build every precision, gate each, let the measurement pick
$CONDA -m sparx_agency.tasks.common.trt_optimizer $A bench \
    --ckpt $CK --onnx-dir $E/onnx --engine-dir $E/nvidiageforcertx_sm120 \
    --plan ~/trt/internnav/plan.json --scenarios 16
```

`--max-depth 1` matters: at the default of 2 the component frontier is
`nn.ModuleList` containers, which have no forward to time, and the plan comes
back unprofiled.

### Benchmarks

```bash
# the headline A/B/C, interleaved in one process (clocks are not lockable here)
$CONDA -m ...trt.benchmark.end_to_end     --ckpt $CK --engine-dir $E/<tag>
# which graph costs the FP16 gate? all eight assignments, no rebuild
$CONDA -m ...trt.benchmark.mixed_precision --ckpt $CK --engine-dir $E/<tag>
# what fewer denoise steps cost in agreement -- a trade for a human to make
$CONDA -m ...trt.benchmark.step_count      --ckpt $CK --engine-dir $E/<tag>
```

## The quality gate

`traj_to_actions` **averages** the 32 candidates (after integrating each, not
before), walks the mean path, and emits up to four discrete actions of which the
agent executes the first. So "the same answer" is the action queue, and the gate
is:

| metric | gated | why |
|---|---|---|
| `first_action_match` | **≥ 1.0** | the command the robot actually executes |
| `action_seq_match` | **≥ 0.95** | the queue; its tail is re-derived next call, so it may slip |
| `action_count_delta`, `endpoint_err_m`, `traj_rel_l2` | no | diagnostics |

Because the candidates are averaged rather than selected, error that is
*uncorrelated* across the ensemble largely cancels while error in the shared
condition tensor does not. That asymmetry is why the condition graph is the one
that had to be pinned.

## Three things upstream gets wrong, and what is done about them

1. **The pinned diffusers cannot load the released checkpoint.**
   `requirements/internvla_n1.txt` pins `diffusers==0.33.1`, whose
   `LuminaFeedForward` dropped the `int(2 * inner_dim / 3)` step. The pinned code
   builds FFN-1536; the released weights are FFN-1024, and 36 tensors fail.
   `model.build_dit` passes `ffn_dim_multiplier=2/3`, which restores the trained
   width exactly. With it, all 600 System-1 tensors match key-for-key.
2. **The classifier-free-guidance branch is a no-op.** `generate_traj` runs the
   denoiser on `cat([zeros_like(cond), cond])` and combines with
   `guidance_scale=1.0`, where `uncond + 1.0 * (cond - uncond) == cond`. The null
   half is computed and thrown away, so the DiT runs at batch 64 to produce a
   batch-32 answer. `validate_parity.check_cfg_identity` verifies this
   numerically (rel_l2 1e-9, the float32 associativity floor) rather than
   trusting the algebra.
3. **`rgb_model` is called around `nn.Module.__call__`.** `generate_traj` calls
   `get_intermediate_layers` directly, so no forward hook fires and a profiler
   measures the DINOv2 trunk as **zero**. `model._as_patch_extractor` rebinds
   `forward` to what the policy actually runs.

Two further deviations, forced by this machine: `sdpa` instead of the hardcoded
`flash_attention_2`, and namespace stubs (`upstream.py`) so
`internnav/model/encoder/__init__.py` does not drag in transformers, pydantic and
the unrelated VLN baselines.

## Negative results

**TensorRT 11.1 on sm_120 miscompiles the condition graph in FP16.** Not a
precision limit — a wrong engine:

| | ORT fp16 | **TRT fp16** | TRT fp32 |
|---|---:|---:|---:|
| vision | 7.0e-4 | 1.9e-3 | 5.9e-4 |
| **condition** | **3.1e-4** | **3.6e-1** | 1.6e-4 |
| denoise | 1.2e-3 | 1.3e-3 | 1.4e-4 |

The identical FP16 ONNX runs correctly under onnxruntime and wrongly as an
engine, identically at builder optimization levels 0, 1 and 2. The output keeps
the reference's mean and standard deviation while all 32 QFormer memory tokens
are wrong by ~1.2 against a std of 0.96; the 4 `cond_projector` tokens in the
same graph are correct to 8e-4. In torch, FP16 for the same modules costs 5e-4.
Pinning the graph costs 1.0% of the decision, so `precision_sensitive=True` and
it is not worth chasing further. **Re-test on a TensorRT upgrade.**

**Blocking `Softmax`/`ReduceMean` during FP16 conversion produces invalid ONNX.**
The shared `fp16_graph` helper's strongest keep-list emitted a cast node
declaring float16 for a float input on both the condition and denoise graphs.
onnxruntime refuses to load such a graph; **TensorRT parsed it and built an
engine.** The toolkit now validates every converted graph with
`onnx.checker.check_model(full_check=True)` and has a `LayerNormalization`-only
rung between the strongest keep-list and the converter defaults — which measured
both valid *and more accurate* than the rung above it. NavDP and FlowNav use the
same helper and are worth re-checking.

**Reducing `num_sample_trajs` buys almost nothing.** In torch the denoise step
measured 8.12 ms at batch 32 and 7.96 ms at batch **1** — the graph is
launch-bound at that size, not FLOP-bound. Candidates are nearly free; the cost
is per-step fixed overhead across 12 tiny transformer blocks.

## Files

```
trt/
├── upstream.py            find the InternNav checkout; namespace stubs
├── model.py               System 1 standalone + checkpoint loader
├── wrappers.py            one export wrapper per engine
├── adapter.py             the ModelAdapter: cadences, graphs, gates
├── validate_parity.py     tiers (a) and (b), FP32 on CPU
├── benchmark/
│   ├── end_to_end.py      shipped vs levers vs engines, interleaved
│   ├── mixed_precision.py all eight per-graph precision assignments
│   └── step_count.py      the behaviour-changing lever, measured not taken
└── engines/               build output, gitignored
```
