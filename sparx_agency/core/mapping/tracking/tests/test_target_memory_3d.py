"""The fused 3-D target memory: anchor, precision weighting, spread and validation."""
from __future__ import annotations

import math

import pytest

from sparx_agency.core.mapping.tracking.target_memory_3d import TargetMemory3D



# -- TargetMemory3D ------------------------------------------------------------------------------
def test_memory_starts_at_the_first_centroid_and_keeps_it_as_the_anchor():
    memory = TargetMemory3D((3.0, 0.5, 0.4), 3.0, step=7)
    assert memory.xyz == (3.0, 0.5, 0.4) and memory.anchor == (3.0, 0.5, 0.4)
    assert memory.n == 1 and memory.first_step == memory.last_step == 7 and memory.spread_m == 0.0
    memory.update((3.4, 0.5, 0.4), 3.4, step=8)
    assert memory.anchor == (3.0, 0.5, 0.4), "the association anchor never moves"
    assert 3.0 < memory.xyz[0] < 3.4 and memory.last_step == 8 and memory.n == 2


def test_a_near_frame_outweighs_a_far_one():
    """sigma = 0.03 + 0.03 r: the frame from one metre carries nine times the weight of the one from four."""
    memory = TargetMemory3D((4.0, 0.0, 0.5), 4.0, step=0)
    memory.update((3.5, 0.0, 0.5), 1.0, step=1)
    assert memory.xyz[0] == pytest.approx(3.5 + 0.5 * (0.06 ** 2) / (0.06 ** 2 + 0.15 ** 2), abs=1e-9)
    assert memory.xyz[0] < 3.6


def test_precision_improves_with_every_frame_and_spread_reports_disagreement():
    memory = TargetMemory3D((2.0, 0.0, 0.5), 2.0, step=0)
    sigmas = [memory.sigma_m]
    for step, x in enumerate((2.1, 1.9, 2.05, 1.95), start=1):
        memory.update((x, 0.0, 0.5), 2.0, step)
        sigmas.append(memory.sigma_m)
    assert all(later < earlier for earlier, later in zip(sigmas, sigmas[1:]))
    assert sigmas[-1] == pytest.approx(0.09 / math.sqrt(5), rel=1e-9)
    assert 0.05 < memory.spread_m < 0.1
    assert memory.planar_range((0.0, 0.0)) == pytest.approx(memory.xyz[0])
    assert memory.shift_from_anchor_m() == pytest.approx(abs(memory.xyz[0] - 2.0))
    diag = memory.diagnostics()
    assert diag["frames"] == 5 and diag["sigma_m"] == pytest.approx(sigmas[-1], abs=1e-4)


def test_identical_frames_do_not_produce_a_negative_spread():
    memory = TargetMemory3D((0.7, 0.0, 0.4), 0.7, step=0)
    for step in range(1, 6):
        memory.update((0.7, 0.0, 0.4), 0.7, step)
    assert memory.spread_m == 0.0 and memory.diagnostics()["spread_m"] == 0.0


@pytest.mark.parametrize("xyz", [(float("nan"), 0, 0), (1, 2), (float("inf"), 0, 0)])
def test_memory_refuses_a_non_finite_centroid(xyz):
    with pytest.raises(ValueError):
        TargetMemory3D(xyz, 1.0, step=0)
    memory = TargetMemory3D((1.0, 0.0, 0.0), 1.0, step=0)
    with pytest.raises(ValueError):
        memory.update(xyz, 1.0, step=1)


def test_memory_noise_model_is_validated():
    with pytest.raises(ValueError):
        TargetMemory3D((1.0, 0.0, 0.0), 1.0, step=0, noise_floor_m=0.0)
    with pytest.raises(ValueError):
        TargetMemory3D((1.0, 0.0, 0.0), 1.0, step=0, noise_per_m=-0.1)


