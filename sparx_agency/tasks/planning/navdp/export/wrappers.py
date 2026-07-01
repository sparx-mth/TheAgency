"""Export-friendly nn.Module wrappers around NavDP's point-goal submodules.

The full ``NavDP_Policy`` is not exportable as one graph (numpy ingestion, a
stochastic diffusion loop, the diffusers scheduler, data-dependent ranking). We
export only the three heavy, deterministic transformer passes as static-shape
ONNX graphs and keep everything else in numpy (``core.planning.navdp.trt``).

Three wrappers, each holding references to the *trained* submodules (weights
shared with the loaded ``NavDP_Policy``):

  * :class:`EncoderWrapper`  -- RGB-D encoder, 9 ViT passes (8 RGB memory frames
    + 1 current depth frame) + the Q-Former. Input is canonical NCHW already in
    [0,1]; ImageNet mean/std are applied *positionally to the BGR* channels (the
    server feeds BGR -- see ``cv2.cvtColor(RGB2BGR)`` before ``process_image``);
    depth is tiled 1->3 channels and NOT normalized.
  * :class:`DenoiseStepWrapper` -- one diffusion denoise step (causal self-attn).
  * :class:`CriticWrapper`      -- the critic (cross-attn cond mask, no causal).

Two export-time parity fixes baked in here:
  * **Positional embedding pre-bake** (:func:`bake_pos_embed`): DINOv2 always
    bicubic-interpolates its 518px pos-embed down to the 224px grid; we compute
    that once in eager fp32 and freeze it as a buffer, deleting the ``Resize``
    op (the single biggest parity risk) from the graph.
  * **Finite masks**: the reference uses ``-inf`` additive attention masks; in
    FP16/TRT ``-inf`` risks NaN, so we bake ``-1e4`` (``exp(-1e4)==0`` -> softmax
    identical). This substitution is blessed by ``validate_parity``.

These modules import torch and the external NavDP model and therefore live under
``tasks/`` -- never importable from the Python-3.8 ROS-free ``core``.
"""
from __future__ import annotations

import types

import torch
import torch.nn as nn

# Large finite stand-in for -inf in additive attention masks (FP16-safe: 65504).
NEG_MASK = -1.0e4


def bake_pos_embed(vit):
    """Freeze DINOv2's interpolated 224px positional embedding as a buffer.

    Computes ``interpolate_pos_encoding`` once at the fixed 224x224 geometry
    (which the model runs every forward anyway, since its pos-embed is sized for
    518px) and overrides the method on THIS instance to return the frozen result,
    so the exported graph contains no bicubic ``Resize``.

    Args:
        vit: a ``DinoVisionTransformer`` trunk (``rgb_model`` / ``depth_model``).
    """
    dev = next(vit.parameters()).device
    dim = vit.pos_embed.shape[-1]
    with torch.no_grad():
        # interpolate_pos_encoding only reads x.shape[1] (=npatch+1) and x.shape[-1];
        # 257 = 256 patches (224/14)^2 + 1 cls -> triggers the 37x37 -> 16x16 path.
        dummy = torch.zeros(1, 257, dim, device=dev)
        baked = vit.interpolate_pos_encoding(dummy, 224, 224).detach().clone()
    vit.register_buffer("_baked_pos_embed", baked)

    def _baked(self, x, w, h):
        return self._baked_pos_embed

    vit.interpolate_pos_encoding = types.MethodType(_baked, vit)


class EncoderWrapper(nn.Module):
    """RGB-D encoder: ``(images (1,8,3,224,224), depth (1,1,224,224)) -> (1,128,384)``."""

    def __init__(self, rgbd_backbone):
        super().__init__()
        self.rgb_model = rgbd_backbone.rgb_model
        self.depth_model = rgbd_backbone.depth_model
        self.former_pe = rgbd_backbone.former_pe
        self.former_query = rgbd_backbone.former_query
        self.former_net = rgbd_backbone.former_net
        self.project_layer = rgbd_backbone.project_layer
        self.memory_size = rgbd_backbone.memory_size
        mean = rgbd_backbone.preprocess_mean.reshape(1, 3, 1, 1).clone()
        std = rgbd_backbone.preprocess_std.reshape(1, 3, 1, 1).clone()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        bake_pos_embed(self.rgb_model)
        bake_pos_embed(self.depth_model)
        self.eval()

    def forward(self, images, depth):
        b = images.shape[0]
        imgs = images.reshape(b * self.memory_size, 3, 224, 224)
        imgs = (imgs - self.mean) / self.std                    # BGR-positional
        img_tok = self.rgb_model.get_intermediate_layers(imgs)[0]      # (8,256,384)
        img_tok = img_tok.reshape(b, self.memory_size * 256, -1)       # (1,2048,384)
        d3 = torch.cat([depth, depth, depth], dim=1)                   # (1,3,224,224)
        dep_tok = self.depth_model.get_intermediate_layers(d3)[0]      # (1,256,384)
        former_token = torch.cat([img_tok, dep_tok], dim=1)           # (1,2304,384)
        former_token = former_token + self.former_pe(former_token)
        query = self.former_query(former_token.new_zeros((b, self.memory_size * 16, former_token.shape[-1])))
        memory_token = self.former_net(query, former_token)
        return self.project_layer(memory_token)                       # (1,128,384)


class DenoiseStepWrapper(nn.Module):
    """One diffusion denoise step.

    ``(last_actions (N,24,3), time_token (N,1,384), goal_embed (N,1,384),
    rgbd_embed (N,128,384)) -> noise_pred (N,24,3)``. Uses a baked causal
    ``tgt_mask`` (finite -1e4); no memory mask.

    Args:
        policy: a loaded ``NavDP_Policy``.
        causal_hint: when True, also pass ``tgt_is_causal=True`` to the decoder.
            This is the one concrete optimization from the alternative export
            script: it hints PyTorch's self-attention that the mask is causal so
            it can take a faster SDPA kernel. The explicit finite ``tgt_mask`` is
            still passed (correctness), unlike the alt script which paired the
            hint with an ``-inf`` mask (an FP16 NaN hazard). The self-attention is
            only 24 tokens, so any speedup is small -- this variant exists so it
            can be measured, not assumed.
    """

    def __init__(self, policy, causal_hint=False):
        super().__init__()
        self.input_embed = policy.input_embed
        self.cond_pos_embed = policy.cond_pos_embed
        self.out_pos_embed = policy.out_pos_embed
        self.decoder = policy.decoder
        self.layernorm = policy.layernorm
        self.action_head = policy.action_head
        self.causal_hint = bool(causal_hint)
        p = policy.predict_size
        causal = torch.triu(torch.ones(p, p), diagonal=1) > 0    # True above diagonal
        self.register_buffer("tgt_mask", torch.where(
            causal, torch.full((p, p), NEG_MASK), torch.zeros(p, p)))
        self.eval()

    def forward(self, last_actions, time_token, goal_embed, rgbd_embed):
        action_embeds = self.input_embed(last_actions)
        cond = torch.cat([time_token, goal_embed, goal_embed, goal_embed, rgbd_embed], dim=1)
        cond_embedding = cond + self.cond_pos_embed(cond)
        input_embedding = action_embeds + self.out_pos_embed(action_embeds)
        extra = {"tgt_is_causal": True} if self.causal_hint else {}
        out = self.decoder(input_embedding, cond_embedding, tgt_mask=self.tgt_mask, **extra)
        return self.action_head(self.layernorm(out))


class CriticWrapper(nn.Module):
    """Critic: ``(predict_trajectory (N,24,3), rgbd_embed (N,128,384)) -> (N,1)``.

    Uses a baked cross-attention ``memory_mask`` (first 4 conditioning columns
    -1e4, goal-agnostic) and no causal mask -- bidirectional over the 24
    trajectory tokens.
    """

    def __init__(self, policy):
        super().__init__()
        self.input_embed = policy.input_embed
        self.cond_pos_embed = policy.cond_pos_embed
        self.out_pos_embed = policy.out_pos_embed
        self.decoder = policy.decoder
        self.layernorm = policy.layernorm
        self.critic_head = policy.critic_head
        p, cols = policy.predict_size, 4 + policy.memory_size * 16
        mask = torch.zeros(p, cols)
        mask[:, 0:4] = NEG_MASK
        self.register_buffer("cond_critic_mask", mask)
        self.eval()

    def forward(self, predict_trajectory, rgbd_embed):
        nogoal = torch.zeros_like(rgbd_embed[:, 0:1])
        action_embeddings = self.input_embed(predict_trajectory)
        action_embeddings = action_embeddings + self.out_pos_embed(action_embeddings)
        cond = torch.cat([nogoal, nogoal, nogoal, nogoal, rgbd_embed], dim=1)
        cond_embeddings = cond + self.cond_pos_embed(cond)
        out = self.decoder(action_embeddings, cond_embeddings,
                           memory_mask=self.cond_critic_mask)
        out = self.layernorm(out)
        return self.critic_head(out.mean(dim=1))                      # (N,1)
