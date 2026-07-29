"""Build the FlowNav TensorRT engines with a fully explicit configuration.

Nothing is left to TensorRT defaults: the workspace memory pool, optimization
level, precision flags, a persisted per-target timing cache, and the strongly-
vs-weakly-typed dispatch are all set from the :class:`HardwareProfile` and
:class:`NetworkPolicy`. FP16 is the validated default ("FP16-first"); INT8 is
deliberately NOT wired for FlowNav yet (FlowNav explicitly trades accuracy for
speed via *fewer Euler steps* K, so over-quantizing on top risks corrupting the
trajectory -- the user's "don't optimize too strongly" requirement).

Engines are written to ``<out-dir>/<target_tag>/<name>.<precision>.engine`` with
a sibling ``.json`` recording the TensorRT version, target SM, precision and IO,
so the runtime can version-lock at load (engines are NOT portable across GPUs or
TensorRT builds). MUST be run with the SAME python ``tensorrt`` the runtime
imports, on the target device.

Run (target device, TRT venv; PYTHONPATH = repo root):
    python -m sparx_agency.tasks.planning.vlas.flownav.trt.engine.build_engine \
        --onnx-dir .../engines/onnx --precision fp16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from sparx_agency.tasks.planning.vlas.flownav.trt.engine.inspect_onnx import inspect
from sparx_agency.tasks.planning.vlas.flownav.trt.export import io_spec
from sparx_agency.tasks.planning.vlas.common.hardware.detect import detect

ENGINE_KEYS = (io_spec.ENCODER, io_spec.VFIELD, io_spec.DIST)
HEAD_PARAMS_NPZ = "flownav_head_params.npz"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _copy_head_params(onnx_dir, out_dir):
    """Copy the exported head-params npz next to the engines (consumers look here)."""
    src = Path(onnx_dir) / HEAD_PARAMS_NPZ
    if src.exists():
        shutil.copy2(src, Path(out_dir) / src.name)


def _load_timing_cache(config, trt, cache_path):
    """Load (or create) a persisted timing cache and attach it to the config."""
    existing = cache_path.read_bytes() if cache_path.exists() else b""
    cache = config.create_timing_cache(existing)
    config.set_timing_cache(cache, ignore_mismatch=False)
    return cache


def _parse(network, trt, onnx_path):
    """Parse an ONNX graph into ``network`` or raise with the parser errors."""
    parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))
    if not parser.parse(Path(onnx_path).read_bytes()):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed for %s: %s" % (onnx_path, errs))


def _build_weak_typed(builder, trt, onnx_path, config, policy):
    """TRT<=10 path: weakly-typed network + BuilderFlag.FP16 (the Orin path)."""
    network = builder.create_network(0)            # explicit batch, weak typing
    _parse(network, trt, onnx_path)
    if policy.use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    return network


def _build_strongly_typed(builder, trt, onnx_path, out_dir, policy, engine_key):
    """TRT>=11 path: strongly-typed network; FP16 comes from an FP16-converted ONNX."""
    parse_path = onnx_path
    if policy.use_fp16 and not policy.force_fp32_strong:
        from sparx_agency.tasks.planning.vlas.common.engine.fp16_onnx import to_fp16_onnx
        parse_path = to_fp16_onnx(onnx_path, out_dir / (engine_key + ".fp16.onnx"))
    elif policy.force_fp32_strong:
        print("[note] %s built FP32 (strongly-typed): deep encoder kept high precision"
              % engine_key)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flag)
    _parse(network, trt, parse_path)
    return network


def build_one(engine_key, onnx_path, out_dir, profile, precision):
    """Parse one ONNX graph and build + serialize its engine. Returns the path.

    Dispatches on the TensorRT generation: TRT<=10 uses weakly-typed networks with
    ``BuilderFlag.FP16``; TRT>=11 removed weak typing, so it builds a strongly-typed
    network from an FP16-converted ONNX.
    """
    import tensorrt as trt

    policy = inspect(onnx_path, profile, precision=precision)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 profile.recommended_workspace_bytes)
    try:
        config.builder_optimization_level = policy.builder_optimization_level
    except AttributeError:
        pass

    weak_typed = hasattr(trt.BuilderFlag, "FP16")    # removed in TRT 11
    if weak_typed:
        network = _build_weak_typed(builder, trt, onnx_path, config, policy)
    else:
        network = _build_strongly_typed(builder, trt, onnx_path, out_dir, policy,
                                        engine_key)

    cache_path = out_dir / ("timing_%s.cache" % profile.target_tag)
    timing = _load_timing_cache(config, trt, cache_path)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None for %s" % engine_key)

    engine_path = out_dir / ("%s.%s.engine" % (engine_key, precision))
    engine_path.write_bytes(bytes(serialized))
    cache_path.write_bytes(bytes(timing.serialize()))
    _write_manifest(engine_path, engine_key, onnx_path, profile, precision, trt)
    return engine_path


def _write_manifest(engine_path, engine_key, onnx_path, profile, precision, trt):
    """Write the sibling ``.json`` the runtime version-locks against."""
    meta = {
        "engine_key": engine_key,
        "precision": precision,
        "trt_version": str(trt.__version__),
        "sm": profile.sm,
        "target_tag": profile.target_tag,
        "gpu_name": profile.gpu_name,
        "onnx_sha256": _sha256(onnx_path),
        "inputs": io_spec.input_names(engine_key),
        "outputs": io_spec.output_names(engine_key),
        "shapes": {k: list(v) for k, v in io_spec.shapes(engine_key).items()},
    }
    Path(str(engine_path) + ".json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--out-dir", default=None, help="default: <onnx-dir>/../<target_tag>")
    ap.add_argument("--precision", choices=["fp16"], default="fp16",
                    help="fp16 only; INT8 is intentionally not wired for FlowNav "
                         "(don't over-quantize on top of low-K flow matching)")
    args = ap.parse_args()

    profile = detect()
    onnx_dir = Path(args.onnx_dir)
    out_dir = Path(args.out_dir) if args.out_dir else \
        onnx_dir.parent / profile.target_tag
    print("[hw] %s sm=%s workspace=%.1fGiB opt-target=%s"
          % (profile.gpu_name, profile.sm,
             profile.recommended_workspace_bytes / (1 << 30), profile.target_tag))

    # Build every flownav_*.onnx present, skipping any without an IO spec.
    keys = sorted(p.stem for p in onnx_dir.glob("flownav_*.onnx"))
    keys = [k for k in keys if k in io_spec.SPECS] or list(ENGINE_KEYS)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in keys:
        path = build_one(key, onnx_dir / (key + ".onnx"), out_dir, profile, args.precision)
        print("[ok] built", path.name)
    _copy_head_params(onnx_dir, out_dir)
    print("[done] engines in", out_dir)


if __name__ == "__main__":
    main()
