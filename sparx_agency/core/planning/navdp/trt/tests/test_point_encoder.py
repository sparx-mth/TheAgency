"""NavDPPointEncoder is a plain affine map; verify shapes and values."""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.core.planning.navdp.trt.point_encoder import NavDPPointEncoder


def test_affine_map_matches_manual():
    rng = np.random.RandomState(1)
    w = rng.randn(384, 3).astype(np.float32)
    b = rng.randn(384).astype(np.float32)
    enc = NavDPPointEncoder(w, b)
    assert enc.token_dim == 384

    goal = np.array([[2.0, -1.0, 0.0]], np.float32)
    out = enc(goal)
    assert out.shape == (1, 384)
    np.testing.assert_allclose(out, goal @ w.T + b, rtol=1e-6, atol=1e-6)


def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        NavDPPointEncoder(np.zeros((384, 2), np.float32), np.zeros(384, np.float32))
    with pytest.raises(ValueError):
        NavDPPointEncoder(np.zeros((384, 3), np.float32), np.zeros(10, np.float32))
    enc = NavDPPointEncoder(np.zeros((384, 3), np.float32), np.zeros(384, np.float32))
    with pytest.raises(ValueError):
        enc(np.zeros((1, 2), np.float32))
