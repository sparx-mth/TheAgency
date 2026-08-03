"""Exponential moving average of model weights (small-data stabilizer).

FlowNav already trains with ``diffusers.EMAModel`` -- reuse that in its loop. NavDP
has no trainer at all, so this lightweight EMA (no diffusers dependency) is provided
for the NavDP fine-tune loop. Evaluate and checkpoint the EMA weights, not the raw
ones; EMA meaningfully stabilizes fine-tuning on tiny datasets.

Torch only.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch


class ModelEma:
    """Track an EMA of a model's parameters and buffers.

    Usage:
        ema = ModelEma(model, decay=0.999)
        ...
        loss.backward(); optimizer.step(); ema.update(model)
        ...
        ema.copy_to(model)      # or evaluate on a clone with ema.state_dict()
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999,
                 keys: Optional[Iterable[str]] = None) -> None:
        """Snapshot the model's weights as the initial shadow copy.

        Args:
            model: The model to track.
            decay: EMA decay per update.
            keys: Restrict the shadow to these ``state_dict`` names. ``None``
                tracks everything. Passing only the *trainable* names is worth
                it on a small GPU: a frozen weight's average is itself, so
                shadowing NavDP's 91 M frozen parameters costs 360 MB of device
                memory to compute the identity. A partial shadow loads back with
                ``strict=False``, which :meth:`copy_to` already uses.
        """
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self._keys = set(keys) if keys is not None else None
        self._shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
            if self._keys is None or k in self._keys
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Blend the model's current weights into the shadow copy."""
        d = self.decay
        for k, v in model.state_dict().items():
            shadow = self._shadow.get(k)
            if shadow is None:
                continue
            if v.dtype.is_floating_point:
                shadow.mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                shadow.copy_(v)  # ints / buffers: track directly

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module) -> None:
        """Load the EMA weights into ``model`` (in place)."""
        model.load_state_dict(self._shadow, strict=False)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return the EMA weights on CPU (for checkpointing / export).

        Streams each shadow tensor host-side rather than cloning on-device, so
        saving never needs a second GPU copy of the model (important on a small or
        shared GPU).
        """
        return {k: v.detach().to("cpu") for k, v in self._shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> int:
        """Restore a saved shadow, for resuming an interrupted run.

        Copies **in place** into the existing shadow tensors so the EMA keeps
        the device and dtype it was built with -- a saved state dict is always
        on CPU, and assigning it wholesale would silently move the shadow off
        the GPU and make every later update a host round-trip.

        Args:
            state: A dict from :meth:`state_dict`.

        Returns:
            How many shadow tensors were restored.

        Raises:
            KeyError: If the saved state shares no keys with this shadow, which
                means it belongs to a different model or a different freeze
                policy. Silently continuing there would resume a run with an
                EMA that never saw any training.
        """
        restored = 0
        for key, value in state.items():
            shadow = self._shadow.get(key)
            if shadow is None:
                continue
            shadow.copy_(value.to(shadow.device, shadow.dtype))
            restored += 1
        if not restored:
            raise KeyError(
                f"none of the {len(state)} saved EMA tensors match this shadow's "
                f"{len(self._shadow)} keys -- the checkpoint is for a different "
                f"model or a different set of trainable parameters")
        return restored
