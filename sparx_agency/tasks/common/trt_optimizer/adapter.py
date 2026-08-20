"""The contract a network implements to be optimized by this toolkit.

Everything else in this package is network-agnostic. This module is the seam:
one small object per network that answers the handful of questions the generic
machinery cannot answer for itself -- how to build the model, which parts run
how often, how to slice it into exportable graphs, and, most importantly, **what
counts as the same answer**.

That last question is where most TensorRT projects go wrong. A generic tool can
measure the relative L2 error between two tensors, and that number is nearly
worthless for a robot: published work finds raw action MSE correlates only about
-0.61 (Spearman) with rollout success. What matters is whether the *decision*
changed -- whether a different trajectory was selected, whether the stop flag
flipped, whether the commanded heading moved enough to matter. Only the network's
own author can say that, so :meth:`ModelAdapter.decision_metrics` is abstract and
has no default. An adapter that returns tensor L2 from it has opted out of the
one check that protects the aircraft.

Adapters are registered rather than imported directly, following the repo's
registry convention: the factory closure does the heavy importing, so listing
what is available stays free.
"""
from __future__ import annotations

from typing import Callable, Dict


class ModelAdapter(object):
    """One network's answers to the questions the generic pipeline must ask.

    Subclass it, implement the abstract methods, and register a factory with
    :func:`register`. Nothing here may import torch at module scope -- do it
    inside the method, so a plan can be inspected on a machine with no torch.

    Attributes:
        name: the registry key and the stem of every artifact this produces.
    """

    name = "unnamed"

    # -- Model -----------------------------------------------------------
    def load(self, checkpoint, device="cpu"):
        """Build the reference torch model from a checkpoint.

        This must produce the *same* model the deployed system runs, in eval
        mode. It is the reference every accuracy claim is measured against.

        Raises:
            NotImplementedError: unless overridden.
        """
        raise NotImplementedError

    def cadences(self):
        """Map component name (or name prefix) -> a :class:`..spec.Cadence`.

        Declared cadences are a starting point only: the profiler counts actual
        calls and overrides anything that disagrees. Declare what you believe
        and let the measurement correct you.
        """
        return {}

    # -- Graphs ----------------------------------------------------------
    def graphs(self):
        """The :class:`..spec.GraphSpec` list this network exports.

        One graph per engine. Split at the boundaries where the *cadence*
        changes, not where the source code happens to have a class: a vision
        trunk that runs once and a denoiser that runs twenty times inside the
        same decision must be two engines, so the conditioning can be uploaded
        once and left resident on the device.

        Raises:
            NotImplementedError: unless overridden.
        """
        raise NotImplementedError

    def wrappers(self, model):
        """Map engine key -> an ``nn.Module`` taking that graph's inputs in order.

        These are *export wrappers*, not the original submodules: they flatten
        keyword arguments into positional tensors, precompute anything that
        would otherwise trace as data-dependent, and replace ``-inf`` masks with
        a large finite value. Any such deliberate change must be blessed by the
        parity gate, which compares the wrapper against the original module.

        Raises:
            NotImplementedError: unless overridden.
        """
        raise NotImplementedError

    def patch(self, model):
        """Apply network-specific export patches to ``model``, in place.

        Called before export and before the torch reference is timed, so the
        reference and the exported graph are the same computation. Generic
        patches (SDPA math, MHA fast path, gradient checkpointing) are applied
        by the exporter; this is for the model's own quirks -- baking a
        positional embedding, deleting a no-op classifier-free-guidance branch,
        swapping a custom autograd Function.
        """
        return model

    # -- Execution -------------------------------------------------------
    def scenarios(self, count, seed=0):
        """Representative inputs, as a list of opaque scenario objects.

        Used for parity, calibration and benchmarking. **Use real captures where
        you have them.** Uniform random RGB-D is out of distribution for any
        pretrained vision trunk, and calibrating or gating on it produces
        confident numbers that do not transfer.

        Raises:
            NotImplementedError: unless overridden.
        """
        raise NotImplementedError

    def run_reference(self, model, scenario):
        """Execute one full decision with the torch model. Returns its output.

        This is the BEFORE measurement and the accuracy reference. It must
        include everything the deployed decision includes -- the whole denoise
        loop, the post-processing, the selection -- not just one forward pass.

        Raises:
            NotImplementedError: unless overridden.
        """
        raise NotImplementedError

    def run_engines(self, runtimes, scenario):
        """Execute one full decision with the built engines. Returns its output.

        ``runtimes`` maps engine key -> a loaded ``TRTEngineRunner``. Everything
        stochastic or data-dependent -- the scheduler, the sampling fan-out, the
        ranking, the post-processing -- stays here in numpy, outside the engines,
        so the comparison isolates the converted graphs.

        Raises:
            NotImplementedError: unless overridden.
        """
        raise NotImplementedError

    # -- Quality ---------------------------------------------------------
    def decision_metrics(self, reference, candidate):
        """Compare two decisions and return the metrics that actually matter.

        Not tensor error. The metrics that decide whether this engine may fly:
        did the selected trajectory change, did the stop decision flip, did the
        commanded heading move more than the controller's deadband, did a
        waypoint cross an obstacle. Tensor L2 belongs here too, as a diagnostic,
        but must never be the gated quantity.

        Returns:
            Mapping of metric name -> float.

        Raises:
            NotImplementedError: unless overridden. There is deliberately no
                default -- a generic guess at what "the same answer" means is
                exactly the thing that lets a bad engine through.
        """
        raise NotImplementedError

    def gates(self):
        """Map metric name -> ``(comparison, threshold)`` the metric must satisfy.

        ``comparison`` is ``"<="`` or ``">="``. A metric with no gate is
        reported as a diagnostic and never blocks. Gate the decision metrics,
        not the tensor ones.
        """
        return {}


_FACTORIES = {}  # type: Dict[str, Callable[[], ModelAdapter]]


def register(name, factory):
    """Register an adapter factory under ``name``.

    Args:
        name: registry key, matching the adapter's ``name``.
        factory: zero-argument callable returning a :class:`ModelAdapter`. Do
            the heavy importing inside it so listing stays free.

    Raises:
        ValueError: if ``name`` is already registered to a different factory.
    """
    existing = _FACTORIES.get(name)
    if existing is not None and existing is not factory:
        raise ValueError("adapter %r is already registered" % name)
    _FACTORIES[name] = factory


def create(name):
    """Instantiate a registered adapter.

    Raises:
        KeyError: naming what is available, when ``name`` is unknown.
    """
    if name not in _FACTORIES:
        raise KeyError("no adapter %r; available: %s"
                       % (name, ", ".join(sorted(_FACTORIES)) or "none"))
    return _FACTORIES[name]()


def available():
    """Sorted names of every registered adapter."""
    return sorted(_FACTORIES)


def check(adapter):
    """Validate an adapter's static declarations before a long run starts.

    Catches the mistakes that would otherwise surface an hour into an export:
    a graph whose shapes are not static, a duplicate engine key, a gate naming a
    metric with an unknown comparison.

    Args:
        adapter: the :class:`ModelAdapter` to check.

    Returns:
        The adapter, unchanged.

    Raises:
        ValueError: describing the first problem found.
    """
    graphs = adapter.graphs()
    if not graphs:
        raise ValueError("adapter %r declares no graphs" % adapter.name)
    seen = set()
    for graph in graphs:
        if graph.key in seen:
            raise ValueError("adapter %r declares engine key %r twice"
                             % (adapter.name, graph.key))
        seen.add(graph.key)
        graph.validate()
    for metric, gate in adapter.gates().items():
        if not isinstance(gate, (tuple, list)) or len(gate) != 2:
            raise ValueError("gate for %r must be (comparison, threshold)" % metric)
        if gate[0] not in ("<=", ">="):
            raise ValueError("gate comparison for %r must be '<=' or '>=', got %r"
                             % (metric, gate[0]))
    return adapter


def evaluate_gates(adapter, metrics):
    """Apply an adapter's gates to a metrics mapping.

    Args:
        adapter: the adapter whose :meth:`ModelAdapter.gates` apply.
        metrics: mapping of metric name -> measured value.

    Returns:
        ``(passed, rows)`` where ``rows`` is a list of
        ``(metric, measured, comparison, threshold, ok)`` tuples covering every
        gated metric. A gated metric missing from ``metrics`` is a failure, not
        a skip -- a gate that silently does not run is worse than no gate.
    """
    rows, passed = [], True
    for metric, (comparison, threshold) in sorted(adapter.gates().items()):
        if metric not in metrics:
            rows.append((metric, None, comparison, threshold, False))
            passed = False
            continue
        value = float(metrics[metric])
        ok = value <= threshold if comparison == "<=" else value >= threshold
        rows.append((metric, value, comparison, threshold, bool(ok)))
        passed = passed and bool(ok)
    return passed, rows
