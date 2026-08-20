"""The headline A/B: upstream as shipped, the free levers, and the engines.

Three configurations of one System-1 decision, **interleaved inside a single
process**, because the GPU clock on this machine is not lockable: two sequential
runs can differ by boost state alone and show a "speedup" that is pure clock.

============  =========================================================
label         what it is
============  =========================================================
``shipped``   upstream exactly as released: torch **bfloat16** (what
              ``InternVLAN1ForCausalLM.from_pretrained`` loads), DiT batch
              **64** because ``generate_traj`` builds a null-conditioned
              branch, 10 flow-matching Euler steps.
``levers``    the same torch model with the two behaviour-free changes:
              the no-op classifier-free-guidance branch dropped (batch 32,
              algebraically identical at ``guidance_scale = 1.0``) and the
              denoise step replayed from a CUDA graph.
``tensorrt``  the built engines at the precision ``selected.json`` blessed,
              with the Euler loop in numpy.
============  =========================================================

``shipped`` is the only honest denominator for a headline speedup: a comparison
against the FP32, batch-32 reference the plan stage happens to use would credit
TensorRT with the levers' win as well as its own.

Usage::

    python -m sparx_agency.tasks.planning.vlas.internvla_n1.trt.benchmark.end_to_end \\
        --ckpt <InternVLA-N1-DualVLN> --engine-dir <.../engines/<target_tag>>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparx_agency.tasks.common.trt_optimizer.bench import latency
from sparx_agency.tasks.common.trt_optimizer.engine import selection
from sparx_agency.tasks.common.trt_optimizer.engine.runner import EngineRunner
from sparx_agency.tasks.planning.vlas.internvla_n1.trt import adapter as ia
from sparx_agency.tasks.planning.vlas.internvla_n1.trt import model as model_mod

#: Upstream's DiT batch: 32 candidates doubled by the null CFG branch.
SHIPPED_BATCH = 2 * model_mod.NUM_SAMPLE_TRAJS


def shipped_call(model, scenario, steps):
    """One System-1 decision exactly as ``generate_traj`` performs it.

    Args:
        model: a patched :class:`..model.System1` in bfloat16.
        scenario: one item from ``adapter.scenarios``.
        steps: flow-matching Euler steps.

    Returns:
        ``(32, 32, 3)`` candidate trajectory deltas.
    """
    import torch

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    images = torch.from_numpy(scenario["images"]).to(device, dtype)
    latents_in = torch.from_numpy(scenario["traj_latents"]).to(device, dtype)
    sample = torch.from_numpy(scenario["noise"]).to(device, dtype)

    with torch.no_grad():
        condition = model.condition(model.vision(images), latents_in)
        # The null branch upstream computes and then cancels at scale 1.0.
        condition = torch.cat([torch.zeros_like(condition), condition], 0)
        condition = condition.repeat_interleave(model_mod.NUM_SAMPLE_TRAJS, dim=0)
        for sigma, sigma_next, timestep in ia.euler_schedule(steps):
            doubled = sample.repeat(2, 1, 1)
            step = torch.full((doubled.shape[0],), float(timestep),
                              device=device, dtype=dtype)
            prediction = model.denoise_step(doubled, step, condition)
            uncond, guided = prediction.chunk(2)
            velocity = uncond + 1.0 * (guided - uncond)
            sample = (sample.float() + (sigma_next - sigma) * velocity.float()).to(dtype)
    return sample.float().cpu().numpy()


def build_lever_call(model, scenario, steps, batch):
    """One decision with the CFG branch dropped and the step in a CUDA graph.

    The graph is captured once against static buffers and replayed, so the
    hundreds of kernel launches a 12-block transformer costs become one. Capture
    happens here rather than inside the timed function.

    Args:
        model: a patched :class:`..model.System1` in bfloat16.
        scenario: the scenario whose shapes the capture is built for.
        steps: flow-matching Euler steps.
        batch: trajectory candidates.

    Returns:
        A zero-argument callable performing one decision.
    """
    import torch

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    images = torch.from_numpy(scenario["images"]).to(device, dtype)
    latents_in = torch.from_numpy(scenario["traj_latents"]).to(device, dtype)
    noise = torch.from_numpy(scenario["noise"]).to(device, dtype)

    with torch.no_grad():
        condition = model.condition(model.vision(images), latents_in)
        static_condition = condition.repeat_interleave(batch, dim=0).contiguous()
        static_sample = noise.clone()
        static_step = torch.full((batch,), 1000.0, device=device, dtype=dtype)

        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):
                model.denoise_step(static_sample, static_step, static_condition)
        torch.cuda.current_stream().wait_stream(warm)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_velocity = model.denoise_step(static_sample, static_step,
                                                 static_condition)

    def call():
        with torch.no_grad():
            cond = model.condition(model.vision(images), latents_in)
            static_condition.copy_(cond.repeat_interleave(batch, dim=0))
            static_sample.copy_(noise)
            for sigma, sigma_next, timestep in ia.euler_schedule(steps):
                static_step.fill_(float(timestep))
                graph.replay()
                static_sample.copy_(
                    (static_sample.float()
                     + (sigma_next - sigma) * static_velocity.float()).to(dtype))
        return static_sample.float().cpu().numpy()

    return call


def main(argv=None):
    """Run the interleaved A/B/C and print the table."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import torch

    adapter = ia.InternVLAN1System1Adapter()
    scenario = adapter.scenarios(1)[0]

    # bfloat16, because that is what the deployed server loads.
    model = adapter.patch(model_mod.load_system1(args.ckpt, device="cuda",
                                                 dtype=torch.bfloat16))
    chosen = selection.read(args.engine_dir)
    runtimes = dict((key, EngineRunner(Path(args.engine_dir) / name))
                    for key, name in chosen["engines"].items())

    calls = {
        "shipped (torch bf16, DiT batch 64, CFG computed)":
            lambda: shipped_call(model, scenario, adapter.steps),
        "levers  (CFG dropped, batch 32, CUDA graph)":
            build_lever_call(model, scenario, adapter.steps, adapter.batch),
        "tensorrt (%s, batch 32)" % chosen["precision"]:
            lambda: adapter.run_engines(runtimes, scenario),
    }

    sync = latency.cuda_sync()
    samples = dict((label, []) for label in calls)
    for label, call in calls.items():          # warm every path before timing
        latency.measure(call, warmup=5, iters=3, sync=sync)
    for _ in range(int(args.rounds)):          # interleave: one round each
        for label, call in calls.items():
            samples[label].extend(
                latency.measure(call, warmup=0, iters=1, sync=sync).samples_ms)

    stats = dict((label, latency.LatencyStats(
        mean_ms=sum(v) / len(v),
        p50_ms=latency.percentile(v, 50), p90_ms=latency.percentile(v, 90),
        p99_ms=latency.percentile(v, 99), min_ms=min(v), max_ms=max(v),
        std_ms=0.0, iters=len(v), warmup=5, samples_ms=v))
        for label, v in samples.items())

    base = list(stats.values())[0]
    print("\ninterleaved, %d rounds each, one process, clocks NOT lockable\n"
          % args.rounds)
    print("%-50s %9s %9s %9s %9s %9s" % ("configuration", "p50 ms", "p90 ms",
                                         "p99 ms", "Hz", "speedup"))
    for label, st in stats.items():
        print("%-50s %9.2f %9.2f %9.2f %9.2f %8.2fx"
              % (label, st.p50_ms, st.p90_ms, st.p99_ms, st.hz,
                 base.p50_ms / st.p50_ms))
    for label, st in stats.items():
        drift = latency.drift_check(st.samples_ms)
        ok = drift.ok if hasattr(drift, "ok") else drift
        if not ok:
            print("  WARNING: %s drifted across the run; the mean flatters it" % label)
    if args.out:
        Path(args.out).write_text(json.dumps(
            dict((k, v.as_dict()) for k, v in stats.items()), indent=2))
        print("\n[written]", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
