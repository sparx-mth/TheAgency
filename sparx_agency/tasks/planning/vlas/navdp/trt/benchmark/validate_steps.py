"""Measure a reduced-step DDIM sampler against the 10-step DDPM baseline.

The Tier-1 speed lever (fewer denoise steps) cannot be blessed by ``bench.py``'s
parity gate (that compares TRT vs a 10-step torch reference, so a sampler change
is invisible). This tool runs several sampler configs on the EXACT same engines
and reports the speedup together with how far the EXECUTED decision drifts.

Crucially it reports the drift against CONTROLS, because a raw "flip vs 10-step
DDPM" number is uninterpretable on its own -- on out-of-distribution inputs the
16 candidate critics are near-tied, so the argmax is near-random and flips under
ANY perturbation. The controls separate that noise from a real step-count effect:

  * flip FLOOR   : ddpm/T vs ddpm/T with DIFFERENT variance noise -- the inherent
                   stochastic floor. If this is already high, the inputs are
                   uninformative (use --frames with real footage) and the other
                   flip numbers mean nothing.
  * flip SAMPLER : ddpm/T vs ddim/T (full steps) -- the DDPM->DDIM change alone.
  * flip STEPS   : ddim/T vs ddim/K -- the PURE step-count effect (the thing you
                   actually control). This is the number that says whether K is
                   safe, net of sampler type and input noise.
  * flip TOTAL   : ddpm/T vs ddim/K -- what you'd fly vs what you ship today.

Torch-free -- only ``NavDPTRTPolicy`` (numpy + engines). Random inputs are a
smoke test; pass ``--frames`` (an .npz of real RGB-D) for a trustworthy answer.

Run on target:
    python -m sparx_agency.tasks.planning.vlas.navdp.trt.benchmark.validate_steps \
        --engine-dir .../engines/orin_sm87 --steps 4 --frames real_rgbd.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from sparx_agency.core.planning.vlas.navdp.trt.errors import NavDPError
from sparx_agency.core.planning.vlas.navdp.trt.policy import NavDPTRTPolicy
from sparx_agency.tasks.planning.vlas.navdp.trt.export import io_spec


def _load_scenarios(frames, num, num_train, seed):
    """Yield ``(goal, images, depth, init, var_a, var_b)`` scenarios.

    Real RGB-D from ``frames`` (npz with ``images (K,8,224,224,3)``,
    ``depth (K,224,224,1)``, optional ``goals (K,3)``) if given, else uniform
    random (a smoke test -- OOD for the ViT, so the flip floor will be high).
    ``var_a``/``var_b`` are two independent DDPM variance-noise sets (for the
    stochastic floor); ``init`` is shared across all configs.
    """
    rng = np.random.RandomState(seed)

    def _noise():
        return (rng.randn(io_spec.N, io_spec.PREDICT, 3).astype(np.float32),
                rng.randn(num_train, io_spec.N, io_spec.PREDICT, 3).astype(np.float32),
                rng.randn(num_train, io_spec.N, io_spec.PREDICT, 3).astype(np.float32))

    def _goal():
        return np.array([[rng.uniform(0.5, 8.0), rng.uniform(-3, 3), 0.0]], np.float32)

    out = []
    if frames:
        data = np.load(frames)
        imgs = np.asarray(data["images"], np.float32)
        deps = np.asarray(data["depth"], np.float32)
        goals = np.asarray(data["goals"], np.float32) if "goals" in data else None
        k = imgs.shape[0]
        if k == 0:
            raise NavDPError("--frames %s has no frames" % frames)
        for i in range(num):
            j = i % k
            img = imgs[j][None] if imgs.ndim == 4 else imgs[j:j + 1]
            dep = deps[j][None] if deps.ndim == 3 else deps[j:j + 1]
            goal = (goals[j % goals.shape[0]].reshape(1, 3).astype(np.float32)
                    if goals is not None else _goal())
            out.append((goal, img, dep) + _noise())
        return out
    for _ in range(num):
        img = rng.rand(1, io_spec.MEM, io_spec.IMG, io_spec.IMG, 3).astype(np.float32)
        dep = (rng.rand(1, io_spec.IMG, io_spec.IMG, 1) * 5.0).astype(np.float32)
        out.append((_goal(), img, dep) + _noise())
    return out


def _chosen(result):
    """(executed trajectory, executed index, critic.max, chosen-sample zeroed?)."""
    all_traj, critic, positive, _ = result
    idx = int(np.argsort(-critic[0])[0])
    length = float(np.linalg.norm(all_traj[0, idx, -1, 0:2]))
    return positive[0, 0], idx, float(critic[0].max()), length < 0.5


def _metrics(results_a, results_b, stop_threshold):
    """(argmax_flip, stop_match, zero_match, chosen_L2) between two result lists."""
    flips = stop_mm = zero_mm = 0
    l2s = []
    for ra, rb in zip(results_a, results_b):
        ea, ia, ca, za = _chosen(ra)
        eb, ib, cb, zb = _chosen(rb)
        flips += int(ia != ib)
        stop_mm += int((ca < stop_threshold) != (cb < stop_threshold))
        zero_mm += int(za != zb)
        l2s.append(float(np.linalg.norm(ea - eb)))
    n = len(results_a)
    return flips / n, 1 - stop_mm / n, 1 - zero_mm / n, float(np.mean(l2s))


def _fps(policy, scenario, warmup=3, iters=15):
    """End-to-end Hz over repeated inference on one scenario (production noise)."""
    goal, img, dep = scenario[0], scenario[1], scenario[2]
    init = scenario[3]
    for _ in range(warmup):
        policy.predict_pointgoal_action(goal, img, dep, init_noise=init)
    t0 = time.perf_counter()
    for _ in range(iters):
        policy.predict_pointgoal_action(goal, img, dep, init_noise=init)
    return iters / (time.perf_counter() - t0)


def _run(policy, scenarios, use_variance):
    """Run every scenario; feed DDPM variance noise (set A) only when requested."""
    res = []
    for (g, im, dp, ini, va, _vb) in scenarios:
        kw = {"init_noise": ini}
        if use_variance:
            kw["variance_noises"] = va
        res.append(policy.predict_pointgoal_action(g, im, dp, **kw))
    return res


def frozen_gold_check(policy, k, num_scenarios=16, frames=None, seed=0,
                      stop_threshold=-999.0):
    """Compare a K-step DDIM config to the 10-step DDPM 'frozen gold' on ``policy``.

    Runs both on the SAME engines and returns the drift metrics. RESTORES the
    policy's incoming sampler before returning, so it is safe to call at server
    startup as a pre-flight gate. On random inputs ``chosen_l2`` is the robust
    signal (it catches a catastrophically low K even when argmax-flip is noise);
    ``argmax_flip`` is only trustworthy with real ``frames``.

    Returns a dict: ``argmax_flip``, ``stop_match``, ``zero_match``, ``chosen_l2``,
    ``num_train``, ``k``, ``scenarios``.
    """
    t = len(policy._alphas_cumprod)
    scenarios = _load_scenarios(frames, num_scenarios, t, seed)
    incoming = (policy.sampler, policy.num_inference_steps)
    try:
        policy.configure_sampler("ddpm")
        gold = _run(policy, scenarios, use_variance=True)
        policy.configure_sampler("ddim", k)
        test = _run(policy, scenarios, use_variance=False)
    finally:
        policy.configure_sampler(*incoming)                 # restore serving config
    flip, stop_m, zero_m, l2 = _metrics(gold, test, stop_threshold)
    return {"argmax_flip": flip, "stop_match": stop_m, "zero_match": zero_m,
            "chosen_l2": l2, "num_train": t, "k": k, "scenarios": len(scenarios)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--head-params", default=None)
    ap.add_argument("--steps", type=int, default=4, help="DDIM inference steps K")
    ap.add_argument("--frames", default=None,
                    help="npz of real {images,depth[,goals]} -- random smoke test if omitted")
    ap.add_argument("--num-scenarios", type=int, default=32)
    ap.add_argument("--stop-threshold", type=float, default=-999.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    engine_dir = Path(args.engine_dir)
    npz = Path(args.head_params) if args.head_params else engine_dir / "navdp_head_params.npz"
    if not npz.exists():
        raise NavDPError("no head_params npz in %s; pass --head-params." % engine_dir)
    if args.frames is None:
        print("[warn] no --frames: RANDOM inputs (OOD for the ViT) -> the flip FLOOR "
              "will be high and the flip numbers uninterpretable. Pass --frames with "
              "real RGB-D for a trustworthy answer; treat this run as a smoke test.")

    policy = NavDPTRTPolicy(engine_dir, npz)                 # ddpm / full steps
    t = len(policy.scheduler.timesteps)
    st = args.stop_threshold
    scenarios = _load_scenarios(args.frames, args.num_scenarios, t, args.seed)

    policy.configure_sampler("ddpm")
    ddpm_a = _run(policy, scenarios, use_variance=True)       # variance set A
    hz_ddpm = _fps(policy, scenarios[0])
    # second DDPM draw: same init, DIFFERENT variance -> stochastic floor.
    ddpm_b = [policy.predict_pointgoal_action(g, im, dp, init_noise=ini, variance_noises=vb)
              for (g, im, dp, ini, _va, vb) in scenarios]

    policy.configure_sampler("ddim", t)
    ddim_full = _run(policy, scenarios, use_variance=False)   # deterministic, full steps

    policy.configure_sampler("ddim", args.steps)
    ddim_k = _run(policy, scenarios, use_variance=False)
    hz_k = _fps(policy, scenarios[0])

    floor = _metrics(ddpm_a, ddpm_b, st)
    sampler = _metrics(ddpm_a, ddim_full, st)
    steps = _metrics(ddim_full, ddim_k, st)
    total = _metrics(ddpm_a, ddim_k, st)
    n = len(scenarios)

    print("[validate-steps] T=%d  K=%d  scenarios=%d  inputs=%s"
          % (t, args.steps, n, "real" if args.frames else "RANDOM(smoke)"))
    print("  speed        : %.2f Hz -> %.2f Hz  (%.2fx)" % (hz_ddpm, hz_k, hz_k / hz_ddpm))
    print("  flip FLOOR   : %.3f   ddpm/%d vs ddpm/%d (diff noise) -- if high, inputs are uninformative"
          % (floor[0], t, t))
    print("  flip SAMPLER : %.3f   ddpm/%d vs ddim/%d (DDPM->DDIM alone)" % (sampler[0], t, t))
    print("  flip STEPS   : %.3f   ddim/%d vs ddim/%d (PURE step-count effect -- the key number)"
          % (steps[0], t, args.steps))
    print("  flip TOTAL   : %.3f   ddpm/%d vs ddim/%d (what you'd fly vs today)"
          % (total[0], t, args.steps))
    print("  chosen_L2    : floor %.3f | steps %.3f | total %.3f" % (floor[3], steps[3], total[3]))
    print("  stop/zero    : floor %.3f/%.3f | total %.3f/%.3f"
          % (floor[1], floor[2], total[1], total[2]))


if __name__ == "__main__":
    main()
