"""NumpyDDPMScheduler must match diffusers 0.33.1 step-for-step.

Replays committed golden vectors (captured from the real diffusers scheduler by
``tasks/planning/vlas/navdp/trt/export/gen_scheduler_golden.py`` with injected variance
noise) through the numpy scheduler and asserts each reverse step matches. Runs
torch-free.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sparx_agency.core.planning.vlas.navdp.trt.scheduler import NumpyDDPMScheduler

GOLDEN = Path(__file__).with_name("scheduler_golden.npz")


@pytest.mark.skipif(not GOLDEN.exists(), reason="scheduler_golden.npz not generated")
def test_step_matches_diffusers_golden():
    g = np.load(GOLDEN)
    sched = NumpyDDPMScheduler(g["alphas_cumprod"])

    # timesteps order must match what NavDP iterates: T-1 .. 0.
    assert list(sched.timesteps) == list(g["timesteps"])

    for i, k in enumerate(g["timesteps"]):
        prev = sched.step(g["model_outputs"][i], int(k), g["samples_in"][i],
                          variance_noise=g["variance_noises"][i])
        ref = g["prev_samples"][i]
        # float32 numpy vs float32 torch: tight but not bitwise.
        np.testing.assert_allclose(prev, ref, rtol=1e-4, atol=1e-5,
                                   err_msg="mismatch at timestep %d" % int(k))


@pytest.mark.skipif(not GOLDEN.exists(), reason="scheduler_golden.npz not generated")
def test_last_step_adds_no_noise():
    # At t == 0 diffusers adds no variance noise; passing garbage noise must not
    # change the result.
    g = np.load(GOLDEN)
    sched = NumpyDDPMScheduler(g["alphas_cumprod"])
    i = list(g["timesteps"]).index(0)
    a = sched.step(g["model_outputs"][i], 0, g["samples_in"][i],
                   variance_noise=np.full_like(g["variance_noises"][i], 1e6))
    b = sched.step(g["model_outputs"][i], 0, g["samples_in"][i], variance_noise=None)
    np.testing.assert_array_equal(a, b)
