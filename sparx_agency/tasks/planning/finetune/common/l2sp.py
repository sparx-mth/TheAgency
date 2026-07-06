"""L2-SP: anti-catastrophic-forgetting regularization toward the pretrained weights.

With a handful of drone flights against 100k+ hours of sim pretraining, a naive
fine-tune overfits and forgets. L2-SP penalizes ``||theta - theta0||^2`` toward the
*starting* (pretrained) weights -- a simplified EWC that needs no Fisher matrix and
no source data (which we do not have). It anchors the fine-tuned model near its
strong prior while still letting it adapt.

Snapshot ``theta0`` at construction, before the first optimizer step. Only the
currently-trainable parameters (and, optionally, a prefix allowlist) are
regularized -- frozen params never move, so penalizing them is wasted compute.

Torch only.
"""
from __future__ import annotations

from typing import Iterable, Optional

import torch


class L2SP:
    """L2 penalty toward a frozen snapshot of the model's initial parameters."""

    def __init__(
        self,
        model: torch.nn.Module,
        weight: float = 1e-3,
        include_prefixes: Optional[Iterable[str]] = None,
    ) -> None:
        """Snapshot the reference weights.

        Args:
            model: The model *as loaded from the pretrained checkpoint* (call before
                training so ``theta0`` is the pretrained value).
            weight: Regularization strength (``lambda_l2sp``).
            include_prefixes: If given, only parameters whose name starts with one
                of these prefixes are regularized. ``None`` -> all trainable params.
        """
        self.weight = float(weight)
        self._prefixes = tuple(include_prefixes) if include_prefixes else None
        self._ref: dict = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if self._prefixes and not name.startswith(self._prefixes):
                continue
            self._ref[name] = p.detach().clone()

    def _selected(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if name in self._ref and p.requires_grad:
                yield name, p

    def penalty(self, model: torch.nn.Module) -> torch.Tensor:
        """Return ``weight * sum ||theta - theta0||^2`` over the selected params."""
        device = next(model.parameters()).device
        total = torch.zeros((), device=device)
        for name, p in self._selected(model):
            ref = self._ref[name].to(p.device)
            total = total + torch.sum((p - ref) ** 2)
        return self.weight * total

    def num_params(self) -> int:
        """Number of parameter tensors currently regularized (diagnostics)."""
        return len(self._ref)
