"""Tests for the differentiable ESDF hinge penalty (torch; skipped without torch)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sparx_agency.tasks.planning.finetune.common.esdf_penalty import (  # noqa: E402
    EsdfHingePenalty,
    EsdfPenaltyConfig,
    sample_sdf,
)


def _flat_sdf(value, h=20, w=20):
    return torch.full((1, 1, h, w), float(value))


def test_sample_constant_field():
    grid = _flat_sdf(1.5)
    wp = torch.tensor([[[1.0, 0.0], [2.0, 0.5]]])  # (1,2,2)
    d = sample_sdf(grid, wp, resolution_m=0.1, origin_x=0.0, origin_y=-1.0)
    assert torch.allclose(d, torch.full_like(d, 1.5), atol=1e-4)


def test_hinge_zero_when_clear():
    grid = _flat_sdf(2.0)                         # everywhere 2 m clear
    pen = EsdfHingePenalty(EsdfPenaltyConfig(margin_m=0.35, weight=1.0))
    wp = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
    out = pen(wp, grid, 0.1, 0.0, -1.0)
    assert float(out) == pytest.approx(0.0, abs=1e-6)


def test_hinge_positive_and_differentiable_inside_margin():
    grid = _flat_sdf(0.1)                         # 0.1 m clearance < margin
    pen = EsdfHingePenalty(EsdfPenaltyConfig(margin_m=0.35, weight=1.0))
    wp = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]], requires_grad=True)
    out = pen(wp, grid, 0.1, 0.0, -1.0)
    assert float(out) > 0.0
    out.backward()
    assert wp.grad is not None and torch.isfinite(wp.grad).all()


def test_gradient_pushes_toward_higher_clearance():
    # SDF increases with the +left (y) direction: a waypoint gets pushed +y.
    h = w = 40
    ys = torch.linspace(-1.0, 1.0, h).view(h, 1).expand(h, w)  # clearance grows with row=y
    grid = ys[None, None].clone()
    pen = EsdfHingePenalty(EsdfPenaltyConfig(margin_m=0.9, weight=1.0, clamp_grad_m=None))
    wp = torch.tensor([[[2.0, 0.0]]], requires_grad=True)      # at y=0, clearance~0
    out = pen(wp, grid, resolution_m=0.1, origin_x=0.0, origin_y=-2.0)
    out.backward()
    # descending the loss means moving +y (toward more clearance): grad_y < 0
    assert wp.grad[0, 0, 1] < 0
