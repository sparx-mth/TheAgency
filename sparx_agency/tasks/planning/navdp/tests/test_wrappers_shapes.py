"""Integration smoke: the export wrappers produce the contracted shapes.

Requires torch, the external NavDP repo (``NAVDP_REPO``) and the checkpoint
(``NAVDP_CKPT``); skipped otherwise (so it does not run on the ROS-free core CI).
The authoritative numeric proof is ``export/validate_parity.py``; this just guards
the IO contract the engines are built against.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None
_REPO = os.environ.get("NAVDP_REPO")
_CKPT = os.environ.get("NAVDP_CKPT")
pytestmark = pytest.mark.skipif(
    not (_HAS_TORCH and _REPO and _CKPT),
    reason="needs torch + NAVDP_REPO + NAVDP_CKPT")


def test_wrapper_output_shapes():
    import torch

    from sparx_agency.tasks.planning.navdp.export import io_spec
    from sparx_agency.tasks.planning.navdp.export.build_policy import build_navdp_policy
    from sparx_agency.tasks.planning.navdp.export.wrappers import (
        CriticWrapper, DenoiseStepWrapper, EncoderWrapper,
    )

    policy = build_navdp_policy(_CKPT, navdp_repo=_REPO, device="cpu")
    enc, den, cri = (EncoderWrapper(policy.rgbd_encoder),
                     DenoiseStepWrapper(policy), CriticWrapper(policy))

    def rt(name, eng):
        return torch.randn(*io_spec.shapes(eng)[name])

    with torch.no_grad():
        e = enc(rt(io_spec.SPECS[io_spec.ENCODER][0][0], io_spec.ENCODER),
                rt(io_spec.SPECS[io_spec.ENCODER][0][1], io_spec.ENCODER))
        assert tuple(e.shape) == (1, io_spec.MEM_TOK, io_spec.TOK)
        n = den(*[rt(x, io_spec.DENOISE) for x in io_spec.input_names(io_spec.DENOISE)])
        assert tuple(n.shape) == (io_spec.N, io_spec.PREDICT, 3)
        c = cri(*[rt(x, io_spec.CRITIC) for x in io_spec.input_names(io_spec.CRITIC)])
        assert tuple(c.shape) == (io_spec.N, 1)
