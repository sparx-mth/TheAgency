"""Generate golden DDPM step vectors from diffusers 0.33.1 for the core test.

The core ``NumpyDDPMScheduler`` must match NavDP's ``diffusers.DDPMScheduler``
exactly, but ``core`` tests must run without torch/diffusers. So we capture a set
of golden ``(model_output, sample_in, variance_noise) -> prev_sample`` tuples
from the real diffusers scheduler here (dev/host env) and commit them as an
``.npz`` next to the test; the torch-free test replays them through the numpy
scheduler and asserts equality.

The scheduler's internal variance-noise draw is monkeypatched to consume our
pre-generated noise so both sides see identical randomness (the only way the
comparison is meaningful).

Run (in the navdp conda env, which has diffusers):
    python -m sparx_agency.tasks.planning.vlas.navdp.trt.export.gen_scheduler_golden \
        --out sparx_agency/core/planning/vlas/navdp/trt/tests/scheduler_golden.npz
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import diffusers.schedulers.scheduling_ddpm as ddpm_mod

N, PREDICT = 16, 24


def generate(out_path: str, seed: int = 0) -> None:
    """Build the diffusers scheduler, run a denoise loop, and dump golden vectors."""
    rng = np.random.RandomState(seed)
    sched = DDPMScheduler(num_train_timesteps=10, beta_schedule="squaredcos_cap_v2",
                          clip_sample=True, prediction_type="epsilon")
    sched.set_timesteps(10)
    alphas_cumprod = sched.alphas_cumprod.cpu().numpy().astype(np.float32)
    timesteps = [int(t) for t in sched.timesteps]

    # Make diffusers' variance noise deterministic and known to us.
    noise_queue = []
    original = ddpm_mod.randn_tensor

    def fake_randn_tensor(shape, generator=None, device=None, dtype=None, **kw):
        arr = noise_queue.pop(0)
        return torch.as_tensor(arr, dtype=dtype or torch.float32,
                               device=device or "cpu")

    ddpm_mod.randn_tensor = fake_randn_tensor
    try:
        naction = rng.randn(N, PREDICT, 3).astype(np.float32)
        mos, sins, vns, prevs = [], [], [], []
        for k in timesteps:
            mo = rng.randn(N, PREDICT, 3).astype(np.float32)
            vn = rng.randn(N, PREDICT, 3).astype(np.float32)
            if k > 0:                       # diffusers only draws noise for t>0
                noise_queue.append(vn)
            sin = naction.copy()
            out = sched.step(model_output=torch.as_tensor(mo), timestep=k,
                             sample=torch.as_tensor(naction))
            prev = out.prev_sample.cpu().numpy().astype(np.float32)
            mos.append(mo); sins.append(sin); vns.append(vn); prevs.append(prev)
            naction = prev
    finally:
        ddpm_mod.randn_tensor = original

    np.savez(
        out_path,
        alphas_cumprod=alphas_cumprod,
        timesteps=np.asarray(timesteps, np.int64),
        model_outputs=np.asarray(mos, np.float32),
        samples_in=np.asarray(sins, np.float32),
        variance_noises=np.asarray(vns, np.float32),
        prev_samples=np.asarray(prevs, np.float32),
    )
    print("wrote", out_path, "steps:", timesteps)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    generate(args.out, seed=args.seed)


if __name__ == "__main__":
    main()
