"""Split a batch into slices whose total work stays inside a memory budget.

The rasteriser is vectorised, which means it materialises one array per
intermediate step over the whole batch at once. That is fast until the batch is
a million triangles, at which point the intermediates are gigabytes. Every
vectorised stage therefore walks its input in slices sized by how much work
each item generates -- samples along a segment, cells inside a bounding box --
rather than by a fixed item count, because those weights vary by orders of
magnitude within a single mesh.
"""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np

DEFAULT_WORK_BUDGET = 4_000_000


def iter_budget_slices(
    weights: np.ndarray, budget: int = DEFAULT_WORK_BUDGET
) -> Iterator[Tuple[int, int]]:
    """Yield ``(start, stop)`` slices whose summed weight fits the budget.

    An item heavier than the whole budget is yielded alone rather than skipped:
    the caller still has to draw it.

    Args:
        weights: ``(N,)`` non-negative integer work estimate per item.
        budget: Maximum summed weight per slice. Must be positive.

    Yields:
        Half-open ``(start, stop)`` index pairs covering ``[0, N)`` in order.

    Raises:
        ValueError: If ``budget`` is not positive.
    """
    if budget <= 0:
        raise ValueError("budget must be positive, got %r" % (budget,))
    weights = np.asarray(weights, dtype=np.int64)
    total = weights.shape[0]
    start = 0
    while start < total:
        running = np.cumsum(weights[start:])
        within = int(np.searchsorted(running, budget, side="right"))
        stop = start + max(1, within)
        yield start, min(stop, total)
        start = stop
