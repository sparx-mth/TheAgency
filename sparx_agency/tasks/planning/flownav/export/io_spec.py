"""Static IO contract (names + shapes) for the three FlowNav engines.

Single source of truth shared by the exporter, the parity validator, the engine
builder and the benchmark. Tensor *names* are imported from the core runtime
(``core.planning.flownav.trt.policy``) so the engine the runtime feeds and the
engine the builder exports can never drift apart. Shapes are fully static
(single drone for the encoder/dist; ``N = num_samples`` for the velocity field).

Geometry comes from FlowNav's config (``flownav.yaml`` / ``data_config.yaml``):
``context_size=3`` -> ``OBS_CH = 3*(3+1) = 12``; ``image_size=[96,96]``;
``len_traj_pred=8``; ``action_dim=2``; ``encoding_size=256``.

The velocity-field ``timestep`` is the continuous flow-matching time ``t in
[0,1]``; it is fed as a rank-1 ``(1,)`` tensor and broadcast to ``N`` inside the
network. The number of Euler steps ``K`` is a *runtime* knob (same engine, more
or fewer calls), NOT a shape -- so it is not encoded here.

This module is numpy-free / torch-free, so it is importable in any environment.
"""
from __future__ import annotations

from sparx_agency.core.planning.flownav.trt.policy import (
    DIST_IN_COND, DIST_OUT, ENC_IN_GOAL, ENC_IN_OBS, ENC_OUT,
    VF_IN_COND, VF_IN_SAMPLE, VF_IN_TIME, VF_OUT,
)

N = 8           # num_samples (the velocity-field engine is built static at this N)
IMG = 96        # image_size (square)
CTX = 3         # context_size
OBS_CH = 3 * (CTX + 1)   # obs stack channels = 12
HORIZON = 8     # len_traj_pred
ACT_DIM = 2     # action_dim (dx, dy)
COND = 256      # encoding_size (global conditioning dim)

ENCODER = "flownav_encoder"
VFIELD = "flownav_vfield"
DIST = "flownav_dist"

# name -> (input_names, output_names, {name: static_shape})
SPECS = {
    ENCODER: (
        [ENC_IN_OBS, ENC_IN_GOAL], [ENC_OUT],
        {ENC_IN_OBS: (1, OBS_CH, IMG, IMG), ENC_IN_GOAL: (1, 3, IMG, IMG),
         ENC_OUT: (1, COND)},
    ),
    VFIELD: (
        [VF_IN_SAMPLE, VF_IN_TIME, VF_IN_COND], [VF_OUT],
        {VF_IN_SAMPLE: (N, HORIZON, ACT_DIM), VF_IN_TIME: (1,),
         VF_IN_COND: (N, COND), VF_OUT: (N, HORIZON, ACT_DIM)},
    ),
    DIST: (
        [DIST_IN_COND], [DIST_OUT],
        {DIST_IN_COND: (1, COND), DIST_OUT: (1, 1)},
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
