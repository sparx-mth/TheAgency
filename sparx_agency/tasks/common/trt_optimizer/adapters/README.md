# Reference adapters — the copy-me examples

Everything in `trt_optimizer` is network-agnostic except one small object: the
**adapter**. It is the seam where a network answers the handful of questions the
generic machinery cannot answer for itself, and it is the only file you write to
put a new model through the pipeline.

The two adapters here are real and they run — `tests/test_adapters.py` checks
both, and `tests/test_adapters_e2e.py` takes the classifier all the way to a
built FP16 engine and back. Neither has anything to do with robots, navigation
or trajectories, which is the point: the toolkit was born beside a
vision-language-action policy and owes nothing to it.

## The seven questions

| method | what it must answer |
|---|---|
| `load` | build the reference torch model, in eval mode. Everything is measured against this |
| `cadences` | component name (or name prefix) → how often it runs. The profiler counts real calls and overrides you |
| `graphs` | one `GraphSpec` per engine, with fully static shapes. **Split where the cadence changes**, not where the source has a class |
| `wrappers` | engine key → an `nn.Module` taking that graph's inputs positionally and returning tensors, not dicts |
| `patch` | this model's own export fixes (bake a positional embedding, delete a no-op branch). Generic patches are the exporter's job |
| `scenarios` / `run_reference` / `run_engines` | one full decision, both ways, over representative inputs |
| `decision_metrics` + `gates` | **the one that matters** — see below |

## Which one to copy

- **`image_classifier.py`** — copy this for anything whose output is *one
  answer per input*: classifiers, scene/gesture recognisers, quality scorers,
  reranker heads, any single-tensor-in, single-tensor-out network. It is the
  smaller of the two and shows the minimum an adapter can be.
- **`segmentation.py`** — copy this when the forward returns a **dict** (as
  every torchvision segmentation model does) or when the output is *dense*: per
  pixel, per token, per voxel, per time-step. It shows the export wrapper that
  turns a dict-returning `forward` into named output tensors, and a
  decision-metric set with a completely different shape.

Both default to `weights=None` so no checkpoint is downloaded, and both seed
construction so an untrained reference is bit-identical across processes.
Their `scenarios()` returns random noise **only** so the reference runs offline
— replace it with real captures before you believe a single gate number.

## `decision_metrics` is the method that matters

`ModelAdapter.decision_metrics` has no default implementation, on purpose. A
generic tool can compute the relative L2 error between two tensors; that number
tells you how far the arithmetic moved and nothing at all about whether anyone
noticed. What you gate is **whether the decision changed** — and the decision is
different for every task family:

| task family | gate on | why |
|---|---|---|
| classification | top-1 / top-k agreement rate | the emitted answer is the `argmax`, not the logits |
| detection | matched-box IoU, box-count delta, confidence-ordering flips | a box that moved 2 px is fine; a box that vanished or reordered is not |
| segmentation | pixel agreement **and** mean IoU | pixel agreement alone hides a small class disappearing |
| depth / regression | fraction within a relative-error band (δ₁-style), plus ordering preservation | absolute error is dominated by the far field nobody acts on |
| ASR / transcription | word error rate | tensor distance over logits is meaningless after decoding |
| tracking | ID switches, track fragmentation | per-frame boxes can all be near-perfect while identity still breaks |
| pose / keypoints | PCK at the tolerance the consumer uses | the consumer has a threshold; use theirs |
| ranking / retrieval | top-k set overlap, NDCG delta | scores may drift freely as long as the order holds |
| generative | the task-specific human-facing metric | never tensor L2 — an equally good sample is far away in tensor space |
| control / robotics | did the selected action, trajectory or stop decision flip | published work finds raw action MSE correlates only ≈ −0.61 (Spearman) with rollout success |

**Tensor error is always a diagnostic and never the gated quantity.** Report it
— `mean_abs_logit_error`, relative L2, max softmax delta — because when an
agreement rate drops it is what tells you whether the cause is FP16 rounding or
a wrong graph. Then gate on the agreement.

Two rules of thumb the shipped examples follow:

1. **Gate the weaker metric harder.** In `image_classifier.py`, top-5 agreement
   is more forgiving than top-1, so it carries the tighter threshold (0.999 vs
   0.99): a prediction may swap with a near-tied neighbour, but the reference
   answer falling out of the shortlist means the class was genuinely lost.
2. **A metric with no gate still gets reported.** `evaluate_gates` treats a
   *gated* metric that is missing from the results as a failure, never a skip —
   a gate that silently does not run is worse than no gate at all.

## Registering and running

Register at the bottom of the module, then import the package to make it
visible:

```python
adapter_mod.register(MyAdapter.name, MyAdapter)
```

```bash
CONDA=~/miniconda3/envs/navdp/bin/python
MOD=sparx_agency.tasks.common.trt_optimizer.adapters.image_classifier

# --ckpt is whatever this adapter's load() takes. For these two that is a
# torchvision weights spec or a state_dict path; "" means random weights.
$CONDA -m sparx_agency.tasks.common.trt_optimizer plan \
    --adapter-module $MOD --adapter image_classifier \
    --ckpt IMAGENET1K_V1 --out /tmp/resnet18_plan.json
$CONDA -m sparx_agency.tasks.common.trt_optimizer export \
    --adapter-module $MOD --adapter image_classifier \
    --ckpt IMAGENET1K_V1 --out-dir /tmp/resnet18/onnx
```

Keep torch out of module scope — import it inside `load` — so a plan stays
inspectable on a machine that cannot build anything.
