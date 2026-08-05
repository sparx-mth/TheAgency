"""Tests for the NavDP / FlowNav loss modules (torch; skipped without torch)."""
import pytest

torch = pytest.importorskip("torch")

from sparx_agency.tasks.planning.vlas.flownav.finetune.loss import (  # noqa: E402
    FlowNavLoss,
    FlowNavLossConfig,
    action_reduce,
)
from sparx_agency.tasks.planning.vlas.navdp.finetune.loss import (  # noqa: E402
    NavDPLoss,
    NavDPLossConfig,
)


def test_action_reduce_masks_batch():
    loss = torch.tensor([[1.0, 3.0], [10.0, 10.0]])   # per-sample means: 2, 10
    mask = torch.tensor([1.0, 0.0])
    r = action_reduce(loss, mask)
    # weighted toward the unmasked sample (2), not 10
    assert float(r) < 5.0


def test_flownav_bc_loss_components():
    fl = FlowNavLoss(FlowNavLossConfig(esdf=None))
    b, t = 4, 8
    vt = torch.zeros(b, t, 2)
    ut = torch.ones(b, t, 2)
    out = fl.bc_loss(vt, ut, action_mask=torch.ones(b),
                     dist_pred=torch.zeros(b), distance=torch.ones(b),
                     goal_mask=torch.zeros(b))
    assert out["flow"] > 0 and "bc" in out
    # flow dominates (alpha tiny)
    assert float(out["bc"]) == pytest.approx(float((1 - fl.config.alpha) * out["flow"] + fl.config.alpha * out["dist"]), rel=1e-4)


def test_navdp_diffusion_loss_zero_when_exact():
    nl = NavDPLoss(NavDPLossConfig(esdf=None))
    x = torch.randn(2, 24, 3)
    assert float(nl.diffusion_loss(x, x)) == pytest.approx(0.0, abs=1e-7)


def test_navdp_critic_target_shapes_and_sign():
    nl = NavDPLoss(NavDPLossConfig(esdf=None, d_safe_m=0.5, progress_alpha=0.1))
    b = 3
    action = torch.zeros(b, 24, 3)
    action[:, :, 0] = 0.4          # forward steps (4x deltas -> 0.1 m/step)
    # SDF everywhere 1.0 (safe) -> no collisions, flat progress -> V ~ 0
    sdf = torch.full((b, 1, 30, 30), 1.0)
    v = nl.critic_target_from_sdf(action, sdf, 0.1, 0.0, -1.5)
    assert v.shape == (b,)
    assert torch.all(v >= -0.01)   # no collisions -> non-negative
