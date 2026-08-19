"""A working :class:`ModelAdapter` for semantic segmentation -- a different answer.

The point of shipping a *second* reference adapter is not that segmentation is
important. It is that ``decision_metrics`` comes out completely different, from
the same contract, without the generic pipeline changing by one line.

A classifier emits one label per image, so "the same answer" is an agreement
rate over ``argmax``. A segmenter emits one label per *pixel*, and the same
question changes shape: a handful of pixels flipping along an object boundary is
invisible in practice, while a whole small object losing its class is a serious
regression -- and both look identical to a mean tensor error, and nearly
identical to a plain pixel-agreement rate. So this adapter gates on pixel
agreement **and** mean IoU, which is per class: an object covering 0.4% of the
frame weighs the same as the background there, and losing it costs a lot of IoU
while barely moving pixel agreement.

Read the two adapters side by side. Everything above ``decision_metrics`` is
near-identical boilerplate; everything below it is where the network's own
author has to think. That asymmetry is why
:meth:`..adapter.ModelAdapter.decision_metrics` has no default implementation.

Weights default to ``None`` **and so does the backbone's** -- torchvision's
segmentation builders default ``weights_backbone`` to an ImageNet checkpoint and
would otherwise download it on first use.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sparx_agency.tasks.common.trt_optimizer import adapter as adapter_mod
from sparx_agency.tasks.common.trt_optimizer.spec import Cadence, GraphSpec

#: Default torchvision architecture: the smallest real segmenter available
#: (~3.2 M parameters), so the reference builds in seconds.
DEFAULT_ARCH = "lraspp_mobilenet_v3_large"

#: Default static input shape, NCHW.
DEFAULT_IMAGE_SHAPE = (1, 3, 224, 224)

#: torchvision segmentation heads return a dict; this is the dense-logits key.
#: TensorRT engines have named output *tensors*, never dicts, so the export
#: wrapper is where the dict is unwrapped.
OUTPUT_KEY = "out"


#: Values of ``--ckpt`` that mean "no pretrained weights, random init".
_NO_WEIGHTS = ("", "none", "null", "random", "untrained", "-")


class SemanticSegmentationAdapter(adapter_mod.ModelAdapter):
    """Optimize any torchvision semantic-segmentation model with this toolkit.

    Args:
        arch: ``lraspp_mobilenet_v3_large``, ``deeplabv3_mobilenet_v3_large``,
            ``fcn_resnet50``, or any other torchvision segmentation builder.
        weights: torchvision weights specification, or a path to a
            ``state_dict``. ``None`` builds random weights and downloads nothing.
        image_shape: static NCHW input shape.
        precision_sensitive: build the graph FP32 instead of FP16.
        init_seed: seed used while constructing an untrained model, so the
            reference is reproducible across processes and an engine exported by
            one can be compared against a model loaded by another.

    Attributes:
        name: the registry key, and the model label on the report.
    """

    name = "semantic_segmentation"

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
        """Engine key: the ONNX stem, the engine stem and the report row."""
        return "%s_segmenter" % self.arch

    # -- Model -----------------------------------------------------------
    def load(self, checkpoint=None, device="cpu"):
        """Build the segmenter in eval mode on ``device``.

        Args:
            checkpoint: overrides ``weights`` when truthy -- a torchvision
                weights specification or a path to a ``state_dict``.
            device: torch device string.

        Returns:
            The ``nn.Module`` every accuracy claim is measured against.

        Raises:
            ValueError: naming candidates, when ``arch`` is not a torchvision
                segmentation model.
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
                    "segmentation checkpoint %s does not exist" % state_path)
            spec = None

        rng_state = torch.get_rng_state()
        torch.manual_seed(self.init_seed)
        try:
            # weights_backbone=None is load-bearing: torchvision defaults it to
            # an ImageNet checkpoint and would reach the network for it.
            model = get_model(self.arch, weights=spec, weights_backbone=None)
        except ValueError:
            from torchvision.models import segmentation
            raise ValueError(
                "%r is not a torchvision segmentation model; available: %s"
                % (self.arch, ", ".join(list_models(module=segmentation))))
        finally:
            torch.set_rng_state(rng_state)

        if state_path is not None:
            model.load_state_dict(torch.load(str(state_path),
                                             map_location="cpu"))
        return model.eval().to(device)

    def cadences(self):
        """One forward per frame: backbone and decoder share a single cadence.

        The empty key is a prefix match covering every component the dissector
        emits (``dissect._resolve_cadence``).
        """
        return {"": Cadence.PER_FRAME}

    # -- Graphs ----------------------------------------------------------
    def graphs(self):
        """One engine: backbone and decode head run together, once per frame."""
        return [GraphSpec(
            key=self.graph_key,
            inputs={"image": self.image_shape},
            outputs=["class_logits"],
            component=self.arch,
            cadence=Cadence.PER_FRAME,
            calls_per_decision=1.0,
            precision_sensitive=self.precision_sensitive,
            opset=17,
            notes=("a decoder's upsampling exports as Resize, which is normal "
                   "here and a bug in a ViT -- use the default OpGatePolicy, "
                   "not op_gate.vit_policy()"))]

    def wrappers(self, model):
        """Wrap the dict-returning forward so the graph has one named output."""
        return {self.graph_key: _dense_logits_wrapper(model)}

    def patch(self, model):
        """No model-specific export patch is needed; an explicit no-op.

        Unwrapping the output dict deliberately does *not* happen here: a patch
        changes the model the torch reference is measured against, a wrapper
        changes only what is exported. Reshaping the output is the wrapper's job.
        """
        return model

    # -- Execution -------------------------------------------------------
    def scenarios(self, count, seed=0):
        """``count`` deterministic random NCHW batches.

        **These are not images.** They are standard-normal noise, and they exist
        so this reference runs offline with no dataset. A segmentation gate
        computed on noise is meaningless twice over: the class maps are
        arbitrary, and IoU over arbitrary maps has no relationship to IoU over a
        real scene. **Replace this with a loader over real held-out frames
        before believing any gate result**, and prefer frames from the camera
        that will be deployed.

        Args:
            count: how many scenarios to generate.
            seed: RNG seed; the same seed always yields the same batches.

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
        """One torch forward. Returns the dense class logits as float32 numpy."""
        import torch

        device = next(model.parameters()).device
        tensor = torch.from_numpy(np.ascontiguousarray(scenario,
                                                       dtype=np.float32))
        with torch.no_grad():
            logits = model(tensor.to(device))[OUTPUT_KEY]
        return logits.detach().cpu().numpy().astype(np.float32)

    def run_engines(self, runtimes, scenario):
        """One engine forward. Returns the dense class logits as float32 numpy.

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
            scenario, dtype=np.float32)})["class_logits"], dtype=np.float32)

    # -- Quality ---------------------------------------------------------
    def decision_metrics(self, reference, candidate):
        """Did the *class map* change, and did any class lose its shape?

        Gated:

        * ``pixel_agreement`` -- fraction of pixels whose predicted class is
          unchanged. The headline number, and on its own a misleading one: a
          class occupying 0.4% of the frame can vanish entirely while pixel
          agreement stays above 0.99.
        * ``mean_iou`` -- IoU between the reference and candidate masks,
          averaged over every class present in *either* map. Per-class, so the
          small object that pixel agreement ignores dominates this instead. A
          class the candidate invented counts at IoU 0, which is intended.

        Diagnostics, never gated: ``changed_pixel_fraction`` (the complement of
        ``pixel_agreement`` -- "0.3% of pixels moved" is the sentence a reviewer
        reasons about) and ``mean_abs_logit_error`` (mean
        ``|reference - candidate|`` over the dense logits, which explains a drop
        and never causes one).

        Args:
            reference: dense logits from :meth:`run_reference`, ``(N, C, H, W)``
                or ``(C, H, W)``.
            candidate: dense logits from :meth:`run_engines`, the same shape.

        Returns:
            Mapping of metric name -> float.

        Raises:
            ValueError: on a shape mismatch, or on anything that is not a dense
                class-logit volume.
        """
        ref = _as_dense_logits(reference, "reference")
        cand = _as_dense_logits(candidate, "candidate")
        if ref.shape != cand.shape:
            raise ValueError(
                "reference logits %r and candidate logits %r have different "
                "shapes; the engine is not computing the same thing"
                % (ref.shape, cand.shape))

        ref_map = np.argmax(ref, axis=1)
        cand_map = np.argmax(cand, axis=1)
        agreement = float(np.mean(ref_map == cand_map))
        return {
            "pixel_agreement": agreement,
            "mean_iou": _mean_iou(ref_map, cand_map),
            "changed_pixel_fraction": 1.0 - agreement,
            "mean_abs_logit_error": float(np.mean(np.abs(ref - cand))),
        }

    def gates(self):
        """Gate the two mask metrics; the tensor error stays a diagnostic.

        ``mean_iou`` carries the looser threshold precisely because it is the
        stricter measurement: a boundary pixel moving costs pixel agreement
        almost nothing and costs a thin class's IoU a great deal, so the same
        number would fail every honest FP16 build.
        """
        return {
            "pixel_agreement": (">=", 0.99),
            "mean_iou": (">=", 0.95),
        }


def _dense_logits_wrapper(model):
    """An ``nn.Module`` returning only the dense class logits, positionally.

    The class is defined here rather than at module scope because this module
    must import without torch: a plan is inspected on machines that cannot build.
    """
    from torch import nn

    class _DenseLogits(nn.Module):
        """Unwrap a torchvision segmentation output dict to one tensor."""

        def __init__(self, wrapped):
            super(_DenseLogits, self).__init__()
            self.wrapped = wrapped

        def forward(self, image):
            """Dense class logits for one image, ``(N, C, H, W)``."""
            return self.wrapped(image)[OUTPUT_KEY]

    return _DenseLogits(model).eval()


def _as_dense_logits(values, label):
    """Coerce one decision's dense logits to a float64 ``(N, C, H, W)`` array.

    Raises:
        ValueError: on anything that is not a dense class-logit volume.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4 or array.shape[1] < 2:
        raise ValueError(
            "%s logits have shape %r; expected (classes, H, W) or "
            "(batch, classes, H, W) with at least two classes"
            % (label, np.asarray(values).shape))
    return array


def _mean_iou(ref_map, cand_map):
    """Mean IoU over every class present in either argmax map.

    Args:
        ref_map: reference class indices, any shape.
        cand_map: candidate class indices, the same shape.

    Returns:
        float: mean IoU in ``0..1``; 1.0 for two identical maps. A class absent
        from both maps is skipped rather than scored as a perfect match, so a
        21-class head is not flattered by the 18 classes never in the frame.
    """
    classes = np.union1d(np.unique(ref_map), np.unique(cand_map))
    scores = []
    for class_id in classes:
        in_ref = ref_map == class_id
        in_cand = cand_map == class_id
        union = int(np.count_nonzero(np.logical_or(in_ref, in_cand)))
        if union == 0:
            continue
        scores.append(int(np.count_nonzero(np.logical_and(in_ref, in_cand)))
                      / float(union))
    return float(np.mean(scores)) if scores else 1.0


adapter_mod.register(SemanticSegmentationAdapter.name,
                     SemanticSegmentationAdapter)
