"""What of NavDP is trained, what is frozen, and how the frozen part is skipped.

NavDP is 135.7 M parameters, and roughly 99 % of a forward pass is spent in two
frozen DINOv2 ViT-S trunks: eight passes over the RGB memory plus one over the
depth frame. If those trunks never move, their output for a given frame never
changes -- so it can be computed once, cached, and the training loop never runs a
ViT again. That is the difference between a 1.5 s optimiser step and a 0.05 s
one on an 8 GB laptop GPU, which is the difference between "a small experiment"
and "a real training run".

:meth:`WorldGoalNavDP.encode_tokens` is the cached-path encoder: it takes the
per-frame patch tokens and resumes NavDP's own forward at the Q-Former, which is
trainable. It is a line-for-line continuation of ``NavDP_RGBD_Backbone.forward``
(and of the ONNX ``EncoderWrapper``), so the cached path and the live path
produce the same numbers.

Trained by default (~44.5 M of 135.7 M):

* the **Q-Former** -- 128 learned queries that cross-attend the 2304 patch
  tokens and compress them into the 128-token scene summary. This is where "what
  matters in this picture" is decided, so it is the layer that must adapt.
* the **fusion decoder** -- 16 transformer layers, 37.9 M parameters, shared by
  the diffusion denoiser and the critic. This is the policy itself.
* the **point-goal encoder** and the action/critic heads.

Frozen always: the **RGB DINOv2 trunk**. It is a general visual representation
trained on far more imagery than this dataset contains, and it is the single
thing keeping the fine-tune from collapsing onto one office building. Frozen by
default but unfreezable: the **depth trunk**, which is the viewpoint-sensitive
path and the right thing to adapt in a second, low-learning-rate stage.

Torch + the external NavDP repo: runs in the ``navdp`` conda env.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from sparx_agency.tasks.planning.vlas.navdp.finetune.finetune_model import (
    DEPTH_TRUNK_PREFIX, NavDPFinetune, NavDPFinetuneConfig,
)

MEMORY_SIZE = 8
PATCH_TOKENS = 256          # (224 / 14) ** 2
TOKEN_DIM = 384
QUERY_TOKENS = MEMORY_SIZE * 16


@dataclass(frozen=True)
class WorldGoalModelConfig(NavDPFinetuneConfig):
    """Freeze policy, plus the one extra capacity knob.

    Attributes:
        train_decoder_last_n: Train only the last N of the 16 decoder layers,
            leaving the earlier ones frozen. ``None`` trains all of them. With a
            single-building dataset the 37.9 M-parameter decoder is the main
            overfitting risk, and this is the cheapest way to spend less of it
            without adding a PEFT dependency.
    """

    train_decoder_last_n: Optional[int] = None


class WorldGoalNavDP(NavDPFinetune):
    """NavDP with a cached-feature encoder path and per-layer decoder freezing."""

    def __init__(self, ckpt_path: str, navdp_repo: Optional[str] = None,
                 device: str = "cuda",
                 config: Optional[WorldGoalModelConfig] = None) -> None:
        super().__init__(ckpt_path, navdp_repo, device=device,
                         config=config or WorldGoalModelConfig())
        self.backbone = self.policy.rgbd_encoder
        self._freeze_decoder_prefix()

    # ------------------------------------------------------------------ freeze
    def _freeze_decoder_prefix(self) -> None:
        keep = getattr(self.config, "train_decoder_last_n", None)
        if keep is None:
            return
        total = len(self.policy.decoder.layers)
        if not 0 < keep <= total:
            raise ValueError(f"train_decoder_last_n must be in 1..{total}, got {keep}")
        for index in range(total - keep):
            for parameter in self.policy.decoder.layers[index].parameters():
                parameter.requires_grad_(False)

    @property
    def depth_encoder_trainable(self) -> bool:
        """True when the depth trunk is being trained -- which invalidates a cache."""
        return any(p.requires_grad for n, p in self.policy.named_parameters()
                   if n.startswith(DEPTH_TRUNK_PREFIX))

    def param_counts(self) -> Dict[str, int]:
        """Total / trainable / frozen parameter counts, for the run log."""
        total = sum(p.numel() for p in self.policy.parameters())
        trainable = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    # ---------------------------------------------------------------- encoders
    @torch.no_grad()
    def tokenize_rgb(self, images: torch.Tensor) -> torch.Tensor:
        """``(N, 3, 224, 224)`` in [0, 1] -> ``(N, 256, 384)`` frozen patch tokens.

        The ImageNet normalisation is applied here, positionally, exactly as the
        backbone does it -- see :mod:`.preprocess` on why the channels are BGR.
        """
        mean = self.encoder.mean.to(images.dtype)
        std = self.encoder.std.to(images.dtype)
        return self.encoder.rgb_model.get_intermediate_layers((images - mean) / std)[0]

    @torch.no_grad()
    def tokenize_depth(self, depth: torch.Tensor) -> torch.Tensor:
        """``(N, 1, 224, 224)`` metres -> ``(N, 256, 384)``. Depth is not normalised."""
        return self.encoder.depth_model.get_intermediate_layers(
            depth.repeat(1, 3, 1, 1))[0]

    def encode_tokens(self, rgb_tokens: torch.Tensor,
                      depth_tokens: torch.Tensor) -> torch.Tensor:
        """Resume NavDP's encoder at the Q-Former, from cached patch tokens.

        Args:
            rgb_tokens: ``(B, 8, 256, 384)`` -- the memory, oldest first.
            depth_tokens: ``(B, 256, 384)`` -- the current frame only, as NavDP
                only ever encodes one depth image.

        Returns:
            ``(B, 128, 384)`` scene embedding, identical to what
            :meth:`encode` produces from raw pixels.
        """
        batch = rgb_tokens.shape[0]
        rgb = rgb_tokens.reshape(batch, MEMORY_SIZE * PATCH_TOKENS, TOKEN_DIM)
        tokens = torch.cat([rgb, depth_tokens], dim=1)          # (B, 2304, 384)
        tokens = tokens + self.backbone.former_pe(tokens)
        query = self.backbone.former_query(
            tokens.new_zeros((batch, QUERY_TOKENS, TOKEN_DIM)))
        return self.backbone.project_layer(self.backbone.former_net(query, tokens))

    # ------------------------------------------------------------ checkpointing
    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """Only the parameters this run can change, on CPU.

        A full NavDP state dict is 543 MB; the trainable subset is ~178 MB, and
        milestone checkpoints are worth keeping only if keeping five of them is
        not a gigabyte decision. Loading is ``strict=False`` onto a freshly built
        policy, so the frozen weights always come from the pretrained checkpoint.
        """
        trainable = {name for name, p in self.policy.named_parameters() if p.requires_grad}
        return {name: value.detach().to("cpu")
                for name, value in self.policy.state_dict().items() if name in trainable}

    def load_trainable(self, state: Dict[str, torch.Tensor], strict: bool = True) -> int:
        """Load a :meth:`trainable_state_dict` back, verifying it changed something.

        Args:
            state: The saved tensors. Extra keys (e.g. a full EMA state dict) are
                accepted; only names the policy knows are used.
            strict: Raise if no parameter actually changed. ``load_state_dict``
                with ``strict=False`` silently succeeds on a completely
                mismatched dict, which would make a "trained" evaluation arm
                identical to the baseline and report a null result as a real one.

        Returns:
            How many parameter tensors changed value.
        """
        own = self.policy.state_dict()
        before = {k: own[k].detach().clone() for k in state if k in own}
        self.policy.load_state_dict(
            {k: v for k, v in state.items() if k in own}, strict=False)
        after = self.policy.state_dict()
        changed = sum(1 for k, v in before.items() if not torch.equal(v, after[k]))
        if strict and changed == 0:
            raise RuntimeError(
                f"loading {len(state)} tensors changed nothing -- the checkpoint does "
                f"not match this model, and evaluating it would silently report the "
                f"pretrained baseline as the fine-tuned result")
        return changed
