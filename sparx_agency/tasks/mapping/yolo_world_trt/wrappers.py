"""Split the ultralytics YOLO-World model at the backbone / text-fusion boundary.

Open-set YOLO-World has two data paths with very different runtime profiles:

  * the **backbone** (CSPDarknet: conv / C2f / SPPF) is text-free and depends only
    on the image -- fully static, and the bulk of the FLOPs. It is the part that
    can run on the Jetson **DLA**.
  * the **neck + head** (RepVL-PAN: ``C2fAttn`` / ``ImagePoolingAttn`` /
    ``WorldDetect``) fuses the text embeddings into the image features. Its class
    dimension follows the (runtime, variable) prompt count, so it stays on the GPU.

Rather than bake the prompts (which would kill open-set), we keep the text
embeddings a *runtime input* and cut the graph in two. This module builds two
:class:`torch.nn.Module` wrappers around a loaded ``WorldModel`` that faithfully
replay ultralytics' own ``WorldModel.predict`` routing (the ``.f`` from-index
wiring, the ``self.save`` retained-output list, ``ImagePoolingAttn`` refining the
text features, and ``WorldDetect`` using the *original* text features):

  * :class:`BackboneWrapper` -- ``image -> (feat_a, feat_b, ...)`` static feature
    maps (the tensors the head consumes across the cut).
  * :class:`HeadWrapper` -- ``(feat_a, feat_b, ..., txt_feats) -> raw detections``
    ``[1, 4 + N, num_anchors]`` with ``N`` = number of prompts.

The cut index is discovered generically (first text-aware layer), so it adapts to
the ``s``/``m``/``l``/``x`` scales without hard-coded indices. ``export_onnx.py``
runs a numerical parity gate (full model vs. split) before trusting the cut -- the
one thing that can silently break across ultralytics versions.

Imports torch lazily via the caller; the classes themselves need torch present.
"""
from __future__ import annotations

from typing import List, Tuple

# Class names of the text-aware modules (matched by name to survive ultralytics
# import-path churn). The FIRST such layer marks the backbone/head cut.
TEXT_AWARE = {"C2fAttn", "ImagePoolingAttn", "WorldDetect", "C2fAttnCIB"}
_IMAGE_POOL = "ImagePoolingAttn"
_WORLD_DETECT = "WorldDetect"


def _cls_name(module) -> str:
    return type(module).__name__


def find_cut(layers) -> int:
    """Index of the first text-aware layer -- the backbone/head boundary.

    Everything before it is text-free (and thus DLA-eligible); everything from it
    onward fuses text and stays on the GPU.
    """
    for i, m in enumerate(layers):
        if _cls_name(m) in TEXT_AWARE:
            return i
    raise RuntimeError(
        "No text-aware layer (C2fAttn / ImagePoolingAttn / WorldDetect) found -- "
        "is this really a YOLO-World model? Class list: %s"
        % [_cls_name(m) for m in layers])


def backbone_output_indices(layers, cut: int) -> List[int]:
    """Backbone layer indices whose outputs the head reads across the cut.

    That is ``cut - 1`` (the ``-1`` feed into the first head layer) plus every
    backbone index referenced by a head layer's ``.f`` (from-index) wiring.
    """
    needed = {cut - 1}
    for m in layers[cut:]:
        f = m.f
        refs = [f] if isinstance(f, int) else list(f)
        for j in refs:
            if j != -1 and j < cut:
                needed.add(j)
    return sorted(needed)


class BackboneWrapper:
    """``image -> tuple(feature maps)`` for the text-free backbone (see module doc).

    Constructed via :func:`build_split` so it shares the resolved cut with the head.
    Subclasses ``torch.nn.Module`` at construction time (torch is imported there).
    """

    def __new__(cls, world_model, cut, out_indices):
        import torch.nn as nn

        class _Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = world_model.model
                self.cut = cut
                # `.save` (indices to cache) lives on the WorldModel, not its
                # inner nn.Sequential (`world_model.model`).
                self.save = set(world_model.save) | set(out_indices)
                self.out_indices = out_indices

            def forward(self, x):
                y = []
                for m in self.layers[:self.cut]:
                    if m.f != -1:
                        x = y[m.f] if isinstance(m.f, int) \
                            else [x if j == -1 else y[j] for j in m.f]
                    x = m(x)
                    y.append(x if m.i in self.save else None)
                return tuple(y[i] for i in self.out_indices)

        return _Backbone()


class HeadWrapper:
    """``(*feature maps, txt_feats) -> raw detections`` for the text-fused head.

    Replays ``WorldModel.predict`` for the layers at/after the cut: keeps the
    original text features for ``WorldDetect`` while letting ``ImagePoolingAttn``
    refine a working copy for the ``C2fAttn`` blocks.
    """

    def __new__(cls, world_model, cut, out_indices):
        import torch.nn as nn

        class _Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = world_model.model
                self.cut = cut
                self.total = len(world_model.model)
                self.save = set(world_model.save)   # on the WorldModel, not Sequential
                self.out_indices = out_indices

            def forward(self, feats: Tuple, txt_feats):
                y = [None] * self.total
                for slot, idx in enumerate(self.out_indices):
                    y[idx] = feats[slot]
                x = y[self.cut - 1]
                ori_txt = txt_feats
                cur_txt = txt_feats
                for m in self.layers[self.cut:]:
                    if m.f != -1:
                        x = y[m.f] if isinstance(m.f, int) \
                            else [x if j == -1 else y[j] for j in m.f]
                    name = _cls_name(m)
                    if name == _IMAGE_POOL:
                        cur_txt = m(x, cur_txt)          # refine text with image
                        y[m.i] = x if m.i in self.save else None
                        continue
                    if name == _WORLD_DETECT:
                        x = m(x, ori_txt)                # head uses ORIGINAL text
                    elif name in TEXT_AWARE:
                        x = m(x, cur_txt)                # C2fAttn: refined text
                    else:
                        x = m(x)
                    y[m.i] = x if m.i in self.save else None
                return x

        return _Head()


def build_split(world_model):
    """Return ``(backbone, head, out_indices, cut)`` wrappers for ``world_model``.

    Args:
        world_model: a loaded ultralytics ``WorldModel`` (``YOLOWorld(...).model``).

    Returns:
        ``(BackboneWrapper module, HeadWrapper module, out_indices, cut)``.
    """
    layers = world_model.model
    cut = find_cut(layers)
    out_indices = backbone_output_indices(layers, cut)
    backbone = BackboneWrapper(world_model, cut, out_indices)
    head = HeadWrapper(world_model, cut, out_indices)
    return backbone, head, out_indices, cut
