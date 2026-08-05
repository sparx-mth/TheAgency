"""Build the NavDP TensorRT 10.x engines with a fully explicit configuration.

Nothing is left to TensorRT defaults: the workspace memory pool, optimization
level, precision flags, per-layer precision pins (under INT8), a persisted
per-target timing cache, and tactic selection are all set from the
:class:`HardwareProfile` and :class:`NetworkPolicy`. FP16 is the default
("FP16-first"); INT8 is opt-in and requires a calibration ``.npz`` -- and it must
still pass the on-target accuracy gate before the benchmark will select it.

Engines are written to ``<out-dir>/<target_tag>/<name>.<precision>.engine`` with
a sibling ``.json`` recording the TensorRT version, target SM, precision and IO,
so the runtime can version-lock at load (engines are NOT portable across GPUs or
TensorRT builds). MUST be run with the SAME python ``tensorrt`` the server
imports, on the target device.

Run (target device, TRT venv; PYTHONPATH = repo root):
    python -m sparx_agency.tasks.planning.vlas.navdp.trt.engine.build_engine \
        --onnx-dir .../engines/onnx --precision fp16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from contextlib import contextmanager
from pathlib import Path

from sparx_agency.tasks.planning.vlas.navdp.trt.engine.inspect_onnx import inspect
from sparx_agency.tasks.planning.vlas.navdp.trt.export import io_spec
from sparx_agency.tasks.planning.vlas.common.hardware.detect import detect

ENGINE_KEYS = (io_spec.ENCODER, io_spec.DENOISE, io_spec.CRITIC)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def _cuda_context(needed):
    """Hold a CUDA primary context if ``needed`` (the INT8 calibrator allocates
    device memory via pycuda, which requires a current context). FP16 builds let
    TensorRT manage its own context, so no pycuda is touched."""
    if not needed:
        yield
        return
    import pycuda.driver as cuda
    cuda.init()
    ctx = cuda.Device(0).retain_primary_context()
    ctx.push()
    try:
        yield
    finally:
        ctx.pop()


def _copy_head_params(onnx_dir, out_dir):
    """Copy the exported head-params npz next to the engines (consumers look here)."""
    src = Path(onnx_dir) / "navdp_head_params.npz"
    if src.exists():
        shutil.copy2(src, Path(out_dir) / src.name)


def _load_timing_cache(config, trt, cache_path):
    """Load (or create) a persisted timing cache and attach it to the config."""
    existing = cache_path.read_bytes() if cache_path.exists() else b""
    cache = config.create_timing_cache(existing)
    config.set_timing_cache(cache, ignore_mismatch=False)
    return cache


def _pin_precision(network, trt, policy):
    """Pin sensitive FLOAT layers to FP32 under INT8 (no-op for pure FP16).

    Only float-valued layers are pinned. Shape / constant / integer layers are
    skipped: their outputs must stay INT32/INT64 (``IShapeLayer``, shape math, and
    the Int64 constants baked into the ``pos_embed`` / ``former`` subgraphs), and a
    name-substring match (e.g. ``pos_embed`` hitting ``.../Constant_2_output_0``)
    would otherwise force FP32 on them -- an illegal TensorRT API use that fails
    the whole INT8 build ("IShapeLayer can only run in precision Int64" /
    "cannot use precision Float with weights of type Int64"). Skipping them does
    not weaken the safeguard: those layers never did float math to begin with.
    """
    if not policy.use_int8:
        return
    int_dtypes = {getattr(trt, n) for n in ("int32", "int64", "bool") if hasattr(trt, n)}
    skip_types = {getattr(trt.LayerType, n) for n in ("SHAPE", "CONSTANT")
                  if hasattr(trt.LayerType, n)}
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if not any(kw in layer.name.lower() for kw in policy.fp_keep_keywords):
            continue
        if layer.type in skip_types:
            continue
        outs = [layer.get_output(o) for o in range(layer.num_outputs)]
        if any(t is not None and t.dtype in int_dtypes for t in outs):
            continue                       # integer/shape output -> must not be FP32
        layer.precision = trt.float32
        for o, t in enumerate(outs):
            if t is not None:
                layer.set_output_type(o, trt.float32)


def _build_weak_typed(builder, trt, onnx_path, config, policy, calibrator, engine_key):
    """TRT<=10 path: weakly-typed network + BuilderFlag.FP16/INT8 (the Orin path)."""
    network = builder.create_network(0)            # explicit batch, weak typing
    _parse(network, trt, onnx_path)
    if policy.use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if policy.use_int8:
        if calibrator is None:
            raise RuntimeError("INT8 requested for %s but no calibrator given" % engine_key)
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        config.int8_calibrator = calibrator
        _pin_precision(network, trt, policy)
    return network


def _build_strongly_typed(builder, trt, onnx_path, out_dir, policy, precision, engine_key):
    """TRT>=11 path: strongly-typed network; FP16 comes from an FP16-converted ONNX."""
    if policy.use_int8:
        raise RuntimeError(
            "INT8 on TensorRT %s needs a Q/DQ ONNX (the calibrator path was removed "
            "in TRT 11). Build INT8 on the Orin's TRT-10 stack, or export a "
            "quantized ONNX." % trt.__version__)
    parse_path = onnx_path
    if policy.use_fp16 and not policy.force_fp32_strong:
        from sparx_agency.tasks.planning.vlas.common.engine.fp16_onnx import to_fp16_onnx
        parse_path = to_fp16_onnx(onnx_path, out_dir / (engine_key + ".fp16.onnx"))
    elif policy.force_fp32_strong:
        print("[note] %s built FP32 (strongly-typed): deep ViT drifts in forced FP16"
              % engine_key)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flag)
    _parse(network, trt, parse_path)
    return network


def _parse(network, trt, onnx_path):
    """Parse an ONNX graph into ``network`` or raise with the parser errors."""
    parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))
    if not parser.parse(Path(onnx_path).read_bytes()):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed for %s: %s" % (onnx_path, errs))


def build_one(engine_key, onnx_path, out_dir, profile, precision, calibrator=None):
    """Parse one ONNX graph and build + serialize its engine. Returns the path.

    Dispatches on the TensorRT generation: TRT<=10 uses weakly-typed networks with
    ``BuilderFlag.FP16`` (the Orin deploy target); TRT>=11 removed weak typing, so
    it builds a strongly-typed network from an FP16-converted ONNX.
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
        network = _build_weak_typed(builder, trt, onnx_path, config, policy,
                                    calibrator, engine_key)
    else:
        network = _build_strongly_typed(builder, trt, onnx_path, out_dir, policy,
                                        precision, engine_key)

    cache_path = out_dir / ("timing_%s.cache" % profile.target_tag)
    timing = _load_timing_cache(config, trt, cache_path)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None for %s" % engine_key)

    out_dir.mkdir(parents=True, exist_ok=True)
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


def _load_calibrators(calib_npz):
    """Build per-engine calibrators from an .npz of stacked input arrays."""
    import numpy as np
    from sparx_agency.tasks.planning.vlas.navdp.trt.engine.calibrator import make_calibrator

    data = np.load(calib_npz)
    cals = {}
    for key in ENGINE_KEYS:
        names = io_spec.input_names(key)
        if all(("%s/%s" % (key, n)) in data for n in names):
            arrays = {n: data["%s/%s" % (key, n)] for n in names}
            cals[key] = make_calibrator(arrays, str(Path(calib_npz).with_name(key + ".calib")))
    return cals


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--out-dir", default=None, help="default: <onnx-dir>/../<target_tag>")
    ap.add_argument("--precision", choices=["fp16", "int8"], default="fp16")
    ap.add_argument("--calib-npz", default=None, help="required for --precision int8")
    args = ap.parse_args()

    profile = detect()
    onnx_dir = Path(args.onnx_dir)
    out_dir = Path(args.out_dir) if args.out_dir else \
        onnx_dir.parent / profile.target_tag
    print("[hw] %s sm=%s workspace=%.1fGiB opt-target=%s"
          % (profile.gpu_name, profile.sm,
             profile.recommended_workspace_bytes / (1 << 30), profile.target_tag))

    # Build every navdp_*.onnx present (picks up optional variants like the
    # causal denoiser), skipping any without an IO spec.
    keys = sorted(p.stem for p in onnx_dir.glob("navdp_*.onnx"))
    keys = [k for k in keys if k in io_spec.SPECS] or list(ENGINE_KEYS)
    out_dir.mkdir(parents=True, exist_ok=True)
    with _cuda_context(needed=(args.precision == "int8")):
        calibrators = _load_calibrators(args.calib_npz) if args.precision == "int8" else {}
        for key in keys:
            path = build_one(key, onnx_dir / (key + ".onnx"), out_dir, profile,
                             args.precision, calibrator=calibrators.get(key))
            print("[ok] built", path.name)
    # Consumers (benchmark, server) read the head params from the engine dir.
    _copy_head_params(onnx_dir, out_dir)
    print("[done] engines in", out_dir)


if __name__ == "__main__":
    main()
