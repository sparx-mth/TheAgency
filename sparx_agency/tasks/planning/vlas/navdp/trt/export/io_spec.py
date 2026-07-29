"""Static IO contract (names + shapes) for the three NavDP engines.

Single source of truth shared by the exporter, the parity validator, the INT8
calibrator and the benchmark. Tensor *names* are imported from the core runtime
(``core.planning.vlas.navdp.trt.policy``) so the engine the runtime feeds and the
engine the builder exports can never drift apart. Shapes are fully static
(single drone, ``sample_num=16``) -- see the design notes on why no axis is
dynamic.

This module is numpy-only (no torch), so it is importable in any environment.
"""
from __future__ import annotations

from sparx_agency.core.planning.vlas.navdp.trt.policy import (
    CRI_IN_RGBD, CRI_IN_TRAJ, CRI_OUT, DEN_IN_ACTIONS, DEN_IN_GOAL, DEN_IN_RGBD,
    DEN_IN_TIME, DEN_OUT, ENC_IN_DEPTH, ENC_IN_IMAGES, ENC_OUT,
)

N = 16          # sample_num
MEM = 8         # memory_size (RGB frames)
PREDICT = 24    # predict_size
TOK = 384       # token_dim
MEM_TOK = 128   # memory tokens out of the encoder (memory_size * 16)
IMG = 224       # image_size

ENCODER = "navdp_encoder"
DENOISE = "navdp_denoise"
DENOISE_CAUSAL = "navdp_denoise_causal"   # variant: tgt_is_causal=True (A/B test)
CRITIC = "navdp_critic"

_DENOISE_IO = (
    [DEN_IN_ACTIONS, DEN_IN_TIME, DEN_IN_GOAL, DEN_IN_RGBD], [DEN_OUT],
    {DEN_IN_ACTIONS: (N, PREDICT, 3), DEN_IN_TIME: (N, 1, TOK),
     DEN_IN_GOAL: (N, 1, TOK), DEN_IN_RGBD: (N, MEM_TOK, TOK),
     DEN_OUT: (N, PREDICT, 3)},
)

# name -> (input_names, output_names, {name: shape})
SPECS = {
    ENCODER: (
        [ENC_IN_IMAGES, ENC_IN_DEPTH], [ENC_OUT],
        {ENC_IN_IMAGES: (1, MEM, 3, IMG, IMG), ENC_IN_DEPTH: (1, 1, IMG, IMG),
         ENC_OUT: (1, MEM_TOK, TOK)},
    ),
    DENOISE: _DENOISE_IO,
    DENOISE_CAUSAL: _DENOISE_IO,           # identical IO, only the export hint differs
    CRITIC: (
        [CRI_IN_TRAJ, CRI_IN_RGBD], [CRI_OUT],
        {CRI_IN_TRAJ: (N, PREDICT, 3), CRI_IN_RGBD: (N, MEM_TOK, TOK),
         CRI_OUT: (N, 1)},
    ),
}


def input_names(engine):
    """Ordered input tensor names for an engine key."""
    return list(SPECS[engine][0])


def output_names(engine):
    """Ordered output tensor names for an engine key."""
    return list(SPECS[engine][1])


def shapes(engine):
    """Mapping of tensor name -> static shape for an engine key."""
    return dict(SPECS[engine][2])
