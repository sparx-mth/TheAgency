"""The YOLO-World text branch: prompts -> text embeddings (runs only on re-prompt).

This is the piece your redirect is built on: the class list is given once at the
start of a run (or whenever the target changes), so the CLIP text encoder does
*not* run per frame. :class:`TextEmbedder` encodes a prompt list into the exact
text-embedding tensor the head engine expects, and the result is cached -- for a
fixed prompt list you always get the same embeddings, so a run re-uses them.

Kept as the torch / ultralytics path on purpose (it is infrequent and CLIP is a
poor TensorRT/DLA fit). It is used both offline (``export_onnx`` needs an example
embedding to trace the head) and at runtime (``YoloTRTDetector.set_prompts``). The
per-frame path never touches this module, so the frame loop stays torch-free.

``embed`` returns a numpy array in the model's text-embedding space (typically
``[N, 512]``, or ``[1, N, 512]`` on some versions -- whatever ``WorldModel``
produces, so the head engine's ``txt_feats`` input matches exactly).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


class TextEmbedder:
    """Encode open-vocabulary prompts into YOLO-World text embeddings (torch)."""

    def __init__(self, weights: str, device: str = "cpu"):
        """Load the YOLO-World model that owns the CLIP text encoder.

        Args:
            weights: path to the ``.pt`` YOLO-World checkpoint (any of s/m/l/x --
                they share the same CLIP text space, so the embeddings are
                interchangeable across the visual scales).
            device: torch device for the (infrequent) text encode.
        """
        self.weights = str(weights)
        self.device = device
        self._model = None  # lazily loaded ultralytics YOLOWorld

    def _ensure(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLOWorld  # lazy: heavy torch dep
        except Exception as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "TextEmbedder needs 'ultralytics' (pip install ultralytics). "
                "Original error: %r" % (exc,))
        self._model = YOLOWorld(self.weights)
        return self._model

    def embed(self, prompts: Sequence[str]) -> np.ndarray:
        """Return the text-embedding tensor for ``prompts`` as float32 numpy.

        Uses ``WorldModel.set_classes`` + the stored ``txt_feats`` buffer -- the
        most version-stable path -- so the array is byte-for-byte the tensor the
        head engine was traced against.
        """
        import torch

        cleaned = [str(p).strip() for p in prompts if str(p).strip()]
        if not cleaned:
            raise ValueError("TextEmbedder.embed: at least one non-empty prompt.")
        model = self._ensure()
        with torch.no_grad():
            model.set_classes(cleaned)
            txt = model.model.txt_feats
        arr = txt.detach().to("cpu", dtype=torch.float32).numpy()
        return np.ascontiguousarray(arr)


def txt_n_axis(shape: Tuple[int, ...], n: int) -> int:
    """Index of the axis whose length equals the prompt count ``n``.

    The embedding may be ``[N, D]`` or ``[1, N, D]``; the class (``N``) axis is the
    dynamic one for the head engine. Raises if it is ambiguous.
    """
    matches = [i for i, s in enumerate(shape) if s == n]
    if len(matches) != 1:
        raise ValueError(
            "cannot locate the N=%d axis in txt_feats shape %s (matches=%s); "
            "pick an example prompt count that is unique in the shape."
            % (n, shape, matches))
    return matches[0]


def save_embeddings(path: str, prompts: List[str], embeddings: np.ndarray) -> None:
    """Persist ``embeddings`` + their prompt order to an ``.npz`` (offline cache)."""
    np.savez(Path(path), prompts=np.array(prompts, dtype=object),
             embeddings=embeddings.astype(np.float32))


def load_embeddings(path: str) -> Tuple[List[str], np.ndarray]:
    """Load a persisted ``(prompts, embeddings)`` pair written by :func:`save_embeddings`."""
    data = np.load(Path(path), allow_pickle=True)
    return list(data["prompts"]), data["embeddings"].astype(np.float32)
