"""A working :class:`ModelAdapter` for an image classifier -- the simplest shape.

This file is a reference implementation meant to be *copied*, and it is also the
package's proof that the adapter seam has nothing to do with the network family
the toolkit happened to be built against. A classifier has no trajectory, no
goal, no episode and no robot; it still answers all seven questions the generic
pipeline asks, and the pipeline builds and gates an engine for it unchanged.

Classification is the clearest illustration of the one question the generic
machinery cannot answer for itself: **what counts as the same answer**. For a
classifier that is not the logit vector, it is the ``argmax``. An engine whose
logits moved by 0.3 in L2 and never changed a prediction is a perfect engine; an
engine whose logits moved by 0.01 and flipped one prediction in fifty is a
regression a tensor-error budget would have waved through. So the gated metrics
here are the two *agreement rates*, and the tensor errors ride along as
diagnostics that explain why an agreement moved without ever blocking a build.

The default is ``resnet18`` with ``weights=None``: a reference that downloads a
45 MB checkpoint the first time anyone runs it is a reference nobody runs.
Untrained weights prove the plumbing and prove nothing about accuracy -- see
:meth:`ImageClassifierAdapter.scenarios`, which says so where it matters. torch
and torchvision are imported inside the methods, never at module scope, so
listing the registry stays possible on a machine with neither installed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sparx_agency.tasks.common.trt_optimizer import adapter as adapter_mod
from sparx_agency.tasks.common.trt_optimizer.spec import Cadence, GraphSpec

#: Default torchvision architecture. Small, ubiquitous, and exports to a graph
#: of Conv/BatchNorm/Relu/Pool/Gemm that every TensorRT generation handles.
DEFAULT_ARCH = "resnet18"

#: Default static input shape, NCHW. Static on purpose: the shared
#: ``TRTEngineRunner`` rejects an engine with a dynamic dimension, because a
#: profile switch costs a one-time ``enqueue`` penalty that lands in tail latency.
DEFAULT_IMAGE_SHAPE = (1, 3, 224, 224)

#: How many classes to consider "the shortlist" for the top-k agreement metric.
TOP_K = 5


#: Values of ``--ckpt`` that mean "no pretrained weights, random init".
_NO_WEIGHTS = ("", "none", "null", "random", "untrained", "-")


class ImageClassifierAdapter(adapter_mod.ModelAdapter):
    """Optimize any torchvision image classifier with this toolkit.

    Args:
        arch: any name accepted by ``torchvision.models.get_model``
            (``resnet18``, ``mobilenet_v3_small``, ``vit_b_16``, ...).
        weights: torchvision weights specification (``"IMAGENET1K_V1"``,
            ``"DEFAULT"``) or a path to a ``state_dict`` file. ``None`` -- the
            default -- builds the architecture with random weights and touches
            no network.
        image_shape: static NCHW input shape.
        precision_sensitive: mark the exported graph as precision-sensitive, so
            :mod:`..decide` builds it FP32. Leave False for a convolutional
            trunk; set it True for a deep transformer classifier (``vit_*``),
            whose residual stream drifts under a strongly-typed FP16 build that
            has no per-layer FP32 fallback to rescue it.
        init_seed: seed used while constructing an *untrained* model, so two
            processes that build the same architecture get bit-identical
            weights. Without it the ONNX exported by one process could not be
            compared against the torch reference loaded by another, and every
            parity number in the report would be noise.

    Attributes:
        name: the registry key, and the model label on the report.
    """

    name = "image_classifier"

    def __init__(self, arch=DEFAULT_ARCH, weights=None,
                 image_shape=DEFAULT_IMAGE_SHAPE, precision_sensitive=False,
                 init_seed=0):
        self.arch = str(arch)
        self.weights = weights
        self.image_shape = tuple(int(d) for d in image_shape)
        self.precision_sensitive = bool(precision_sensitive)
        self.init_seed = int(init_seed)

    @property
    def graph_key(self):
        """Engine key: the ONNX stem, the engine stem and the report row.

        It carries the architecture rather than this adapter's registry name so
        that engines for two different backbones never collide in one
        ``engines/`` directory.
        """
        return "%s_classifier" % self.arch

    # -- Model -----------------------------------------------------------
    def load(self, checkpoint=None, device="cpu"):
        """Build the classifier in eval mode on ``device``.

        Args:
            checkpoint: overrides ``weights`` when truthy. Either a torchvision
                weights specification or a path to a ``state_dict``.
            device: torch device string.

        Returns:
            The ``nn.Module`` every accuracy claim is measured against.

        Raises:
            ValueError: naming candidate architectures, when ``arch`` is not a
                torchvision model.
            FileNotFoundError: when a checkpoint path was given and is absent.
        """
        import torch
        from torchvision.models import get_model, list_models

        spec = checkpoint if checkpoint else self.weights
        if isinstance(spec, str) and spec.strip().lower() in _NO_WEIGHTS:
            # The CLI always passes --ckpt, so a caller with no checkpoint has to
            # be able to say so. Accept the obvious spellings rather than letting
            # torchvision fail with a bare KeyError from its weights enum.
            spec = None
        state_path = None
        if spec is not None and Path(str(spec)).suffix in (".pt", ".pth"):
            state_path = Path(str(spec))
            if not state_path.is_file():
                raise FileNotFoundError(
                    "classifier checkpoint %s does not exist" % state_path)
            spec = None

        # Constructing under a scoped seed keeps an untrained reference
        # reproducible across processes without perturbing the caller's RNG.
        rng_state = torch.get_rng_state()
        torch.manual_seed(self.init_seed)
        try:
            model = get_model(self.arch, weights=spec)
        except ValueError:
            raise ValueError(
                "%r is not a torchvision classifier; available names include %s"
                % (self.arch, ", ".join(list_models()[:12]) + ", ..."))
        finally:
            torch.set_rng_state(rng_state)

        if state_path is not None:
            model.load_state_dict(torch.load(str(state_path),
                                             map_location="cpu"))
        return model.eval().to(device)

    def cadences(self):
        """Every component runs once per decision: a classifier is one forward.

        The empty key is a *prefix* match covering every component name the
        dissector emits (see ``dissect._resolve_cadence``), which is the honest
        declaration here -- there is no conditioning to cache, no episode, and
        no inner loop, so naming the modules one by one would only invite the
        list to rot when the architecture changes.
        """
        return {"": Cadence.PER_FRAME}

    # -- Graphs ----------------------------------------------------------
    def graphs(self):
        """One engine covering the whole forward pass.

        There is no cadence boundary to split at: the stem, the trunk and the
        head all run exactly once per image, so splitting them would only add
        host round-trips between engines.
        """
        return [GraphSpec(
            key=self.graph_key,
            inputs={"image": self.image_shape},
            outputs=["logits"],
            component=self.arch,
            cadence=Cadence.PER_FRAME,
            calls_per_decision=1.0,
            precision_sensitive=self.precision_sensitive,
            opset=17,
            notes=("whole-model graph: every component shares one cadence, so "
                   "there is no boundary worth splitting at"))]

    def wrappers(self, model):
        """The model is its own export wrapper: one tensor in, one tensor out."""
        return {self.graph_key: model}

    def patch(self, model):
        """No model-specific export patch is needed here; an explicit no-op.

        This is the seam where a real network bakes a positional embedding or
        deletes a traced-away branch. The *generic* patches -- SDPA on its math
        backend, the ``nn.MultiheadAttention`` fast path disabled -- are applied
        by the exporter for every model and do not belong here.
        """
        return model

    # -- Execution -------------------------------------------------------
    def scenarios(self, count, seed=0):
        """``count`` deterministic random NCHW batches.

        **These are not images.** They are standard-normal noise, on the scale a
        real ImageNet-normalized tensor occupies, and they exist for exactly one
        reason: so this reference adapter runs end to end on a machine with no
        dataset and no network access. Random noise is out of distribution for
        any pretrained vision trunk, so a calibration or an accuracy gate
        computed on it produces a confident number that does not transfer.
        **Replace this method with a loader over real held-out images before you
        believe any gate result**, and prefer images the deployed camera
        actually produces over a public validation set.

        Args:
            count: how many scenarios to generate.
            seed: RNG seed; the same seed always yields the same batches, so an
                engine and a torch reference can be compared across processes.

        Returns:
            List of float32 arrays shaped like the declared graph input.

        Raises:
            ValueError: if ``count`` is not positive.
        """
        if int(count) <= 0:
            raise ValueError("scenarios(count=%r): need at least one scenario "
                             "to compare an engine against" % (count,))
        rng = np.random.default_rng(int(seed))
        return [rng.standard_normal(self.image_shape).astype(np.float32)
                for _ in range(int(count))]

    def run_reference(self, model, scenario):
        """One torch forward. Returns the logits as float32 numpy."""
        import torch

        device = next(model.parameters()).device
        tensor = torch.from_numpy(np.ascontiguousarray(scenario,
                                                       dtype=np.float32))
        with torch.no_grad():
            logits = model(tensor.to(device))
        return logits.detach().cpu().numpy().astype(np.float32)

    def run_engines(self, runtimes, scenario):
        """One engine forward. Returns the logits as float32 numpy.

        Raises:
            KeyError: naming the engine this adapter needs, when it is absent.
        """
        runner = runtimes.get(self.graph_key)
        if runner is None:
            raise KeyError(
                "no runtime for engine %r (have: %s); build that engine before "
                "benchmarking" % (self.graph_key,
                                  ", ".join(sorted(runtimes)) or "none"))
        return np.asarray(runner.infer({"image": np.ascontiguousarray(
            scenario, dtype=np.float32)})["logits"], dtype=np.float32)

    # -- Quality ---------------------------------------------------------
    def decision_metrics(self, reference, candidate):
        """Did the *prediction* change? Tensor error is only the explanation.

        The two agreement rates are the gated quantities:

        * ``top1_agreement`` -- fraction of samples whose ``argmax`` is
          unchanged. This is the decision a classifier actually emits.
        * ``top5_agreement`` -- fraction of samples whose reference top-1 class
          is still somewhere in the candidate's top-5. It is deliberately the
          *weaker* of the two, which is why it carries the tighter threshold: a
          prediction may legitimately swap with a near-tied neighbour, but the
          reference answer falling out of the shortlist means the engine has
          genuinely lost the class.

        The other two are diagnostics and must never be gated:

        * ``mean_abs_logit_error`` -- mean ``|reference - candidate|`` over every
          logit. Tells you how far FP16 moved the arithmetic; says nothing about
          whether anyone noticed.
        * ``max_softmax_delta`` -- the largest probability any single class
          moved. The number to read when a downstream stage thresholds on
          confidence (``accept if p > 0.6``): safe while far below that margin.

        Args:
            reference: logits from :meth:`run_reference`, ``(N, C)`` or ``(C,)``.
            candidate: logits from :meth:`run_engines`, the same shape.

        Returns:
            Mapping of metric name -> float.

        Raises:
            ValueError: on a shape mismatch -- comparing two differently shaped
                outputs is a bug in the adapter, not a low score.
        """
        ref = _as_logits(reference, "reference")
        cand = _as_logits(candidate, "candidate")
        if ref.shape != cand.shape:
            raise ValueError(
                "reference logits %r and candidate logits %r have different "
                "shapes; the engine is not computing the same thing"
                % (ref.shape, cand.shape))

        k = min(TOP_K, ref.shape[1])
        ref_top1 = np.argmax(ref, axis=1)
        cand_order = np.argsort(-cand, axis=1)
        in_top_k = np.any(cand_order[:, :k] == ref_top1[:, None], axis=1)
        probs_ref, probs_cand = _softmax(ref), _softmax(cand)
        return {
            "top1_agreement": float(np.mean(cand_order[:, 0] == ref_top1)),
            "top5_agreement": float(np.mean(in_top_k)),
            "mean_abs_logit_error": float(np.mean(np.abs(ref - cand))),
            "max_softmax_delta": float(np.max(np.abs(probs_ref - probs_cand))),
        }

    def gates(self):
        """Gate the agreements, never the tensor error.

        One flipped prediction in a hundred is the most an accelerated
        classifier may cost, and the reference answer may leave the top-5
        shortlist at most once in a thousand.
        """
        return {
            "top1_agreement": (">=", 0.99),
            "top5_agreement": (">=", 0.999),
        }


def _as_logits(values, label):
    """Coerce one decision's logits to a float64 ``(N, C)`` array.

    Raises:
        ValueError: on anything that is not a vector or a batch of vectors.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(
            "%s logits have shape %r; expected (num_classes,) or "
            "(batch, num_classes) with at least two classes"
            % (label, np.asarray(values).shape))
    return array


def _softmax(logits):
    """Row-wise softmax, shifted by the row max so large logits cannot overflow."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=1, keepdims=True)


adapter_mod.register(ImageClassifierAdapter.name, ImageClassifierAdapter)
