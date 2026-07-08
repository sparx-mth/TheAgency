"""Build the two YOLO-World TensorRT engines with a fully explicit config.

From one exported variant this builds:

  * the **backbone** engine -- static, text-free -> targets the **DLA** (core +
    GPU fallback + explicit DLA memory pools) at FP16.
  * the **head** engine -- text-fused, with a **dynamic ``N`` (prompt-count)
    optimization profile** so any prompt list runs without a rebuild -> **GPU**,
    FP16 (DLA cannot run dynamic shapes).

Nothing is left to TensorRT defaults (precision, device, pools, workspace,
optimization level all come from the :class:`BuildPolicy`). Each engine gets a
sibling ``.json`` manifest (TRT version, SM, IO, DLA-eligible layer count, dynamic
bounds) so the runtime version-locks and the DLA offload is visible.

Engines are locked to the exact GPU + TensorRT build. **DLA engines build only on
a Jetson**; off-Jetson the backbone silently falls back to a GPU engine so the
pipeline is still exercisable on a laptop.

Run (target device, TRT venv, PYTHONPATH = repo root):
    python -m sparx_agency.tasks.mapping.yolo_world_trt.build_engine \\
        --onnx-dir .../engines/onnx --variant s        # builds both roles
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sparx_agency.tasks.mapping.yolo_world_trt.build_policy import (
    build_policy, load_config,
)
from sparx_agency.tasks.mapping.yolo_world_trt.hardware import detect


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parse(network, trt, onnx_path):
    parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))
    if not parser.parse(Path(onnx_path).read_bytes()):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed for %s: %s" % (onnx_path, errs))


def _configure_precision(config, trt, policy, calibrator):
    config.set_flag(trt.BuilderFlag.FP16)
    if policy.use_int8:
        if calibrator is None:
            raise RuntimeError("INT8 requested but no calibrator was provided.")
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        config.int8_calibrator = calibrator


def _configure_dla(config, trt, policy):
    """Point the default device at the DLA, keep GPU fallback, size the pools."""
    config.default_device_type = trt.DeviceType.DLA
    config.DLA_core = policy.dla_core
    if policy.gpu_fallback:
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
    for pool, nbytes in (
        (trt.MemoryPoolType.DLA_MANAGED_SRAM, policy.dla_managed_sram_bytes),
        (trt.MemoryPoolType.DLA_LOCAL_DRAM, policy.dla_local_dram_bytes),
        (trt.MemoryPoolType.DLA_GLOBAL_DRAM, policy.dla_global_dram_bytes),
    ):
        try:
            config.set_memory_pool_limit(pool, int(nbytes))
        except (AttributeError, TypeError):
            pass


def _count_dla_eligible(config, network):
    try:
        return sum(int(config.can_run_on_DLA(network.get_layer(i)))
                   for i in range(network.num_layers))
    except Exception:  # noqa: BLE001
        return 0


def _add_head_profile(builder, config, io):
    """Add the dynamic-N optimization profile for the head engine.

    Feature-map inputs are static (min=opt=max = their shape); only ``txt_feats``
    varies along the class axis over ``[n_min, n_opt, n_max]``.
    """
    profile = builder.create_optimization_profile()
    for name, shape in zip(io["backbone"]["feat_names"], io["backbone"]["feat_shapes"]):
        t = tuple(int(d) for d in shape)
        profile.set_shape(name, t, t, t)

    axis = int(io["txt_n_axis"])
    base = [int(d) for d in io["txt_example_shape"]]
    head = io["head"]

    def _with_n(n):
        s = list(base)
        s[axis] = int(n)
        return tuple(s)

    profile.set_shape(head["txt_input"], _with_n(head["n_min"]),
                      _with_n(head["n_opt"]), _with_n(head["n_max"]))
    config.add_optimization_profile(profile)


def build_one(role, onnx_path, io, out_dir, profile, policy, calibrator=None):
    """Parse one ONNX graph and build + serialize its engine. Returns the path."""
    import tensorrt as trt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)          # explicit batch
    _parse(network, trt, onnx_path)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, policy.workspace_bytes)
    try:
        config.builder_optimization_level = policy.builder_optimization_level
    except AttributeError:
        pass

    _configure_precision(config, trt, policy, calibrator)
    dla_eligible = 0
    if policy.is_dynamic:
        _add_head_profile(builder, config, io)
    if policy.use_dla:
        _configure_dla(config, trt, policy)
        dla_eligible = _count_dla_eligible(config, network)

    cache_path = out_dir / ("timing_%s.cache" % profile.target_tag)
    existing = cache_path.read_bytes() if cache_path.exists() else b""
    timing = config.create_timing_cache(existing)
    config.set_timing_cache(timing, ignore_mismatch=False)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None for %s/%s "
                           "(DLA forced off-Jetson?)" % (policy.variant, role))

    device = "dla%d" % policy.dla_core if policy.use_dla else "gpu"
    engine_path = out_dir / ("yolo_world_%s.%s.%s.%s.engine"
                             % (policy.variant, role, policy.precision, device))
    engine_path.write_bytes(bytes(serialized))
    cache_path.write_bytes(bytes(timing.serialize()))
    _write_manifest(engine_path, role, onnx_path, io, profile, policy, trt,
                    network.num_layers, dla_eligible)
    return engine_path


def _write_manifest(engine_path, role, onnx_path, io, profile, policy, trt,
                    total_layers, dla_eligible):
    meta = {
        "role": role,
        "variant": policy.variant,
        "precision": policy.precision,
        "device": "dla%d" % policy.dla_core if policy.use_dla else "gpu",
        "use_dla": policy.use_dla,
        "gpu_fallback": policy.gpu_fallback if policy.use_dla else None,
        "dynamic_N": policy.is_dynamic,
        "trt_version": str(trt.__version__),
        "sm": profile.sm,
        "target_tag": profile.target_tag,
        "gpu_name": profile.gpu_name,
        "power_budget_w": profile.power_budget_w,
        "imgsz_hw": io.get("imgsz_hw"),
        "embed_dim": io.get("embed_dim"),
        "onnx_sha256": _sha256(onnx_path),
        "total_layers": int(total_layers),
        "dla_eligible_layers": int(dla_eligible),
        "workspace_bytes": policy.workspace_bytes,
        "builder_optimization_level": policy.builder_optimization_level,
    }
    if role == "backbone":
        meta["feat_names"] = io["backbone"]["feat_names"]
        meta["feat_shapes"] = io["backbone"]["feat_shapes"]
    else:
        meta["head"] = {k: io["head"][k] for k in
                        ("feat_inputs", "txt_input", "output", "output_example_shape",
                         "n_min", "n_opt", "n_max")}
        meta["txt_n_axis"] = io["txt_n_axis"]
        meta["txt_example_shape"] = io["txt_example_shape"]
    Path(str(engine_path) + ".json").write_text(json.dumps(meta, indent=2))


def build_variant(onnx_dir, variant, roles, profile, cfg, precision=None, dla=None):
    """Build the requested roles for one variant. Returns ``{role: engine_path}``."""
    onnx_dir = Path(onnx_dir)
    io = json.loads((onnx_dir / ("yolo_world_%s.io.json" % variant)).read_text())
    out_dir = onnx_dir.parent / profile.target_tag
    built = {}
    for role in roles:
        onnx_path = onnx_dir / ("yolo_world_%s.%s.onnx" % (variant, role))
        policy = build_policy(role, variant, profile, config=cfg,
                              precision=precision, dla=dla)
        if policy.use_int8:
            raise SystemExit("INT8 is not wired yet; use fp16. See README 'INT8'.")
        print("[build] %s/%s device=%s dynamic=%s opt=%d"
              % (variant, role, "dla%d" % policy.dla_core if policy.use_dla else "gpu",
                 policy.is_dynamic, policy.builder_optimization_level))
        built[role] = build_one(role, onnx_path, io, out_dir, profile, policy)
        print("[ok] built", built[role].name)
    return built


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx-dir", required=True, help="dir with yolo_world_<v>.*.onnx")
    ap.add_argument("--variant", required=True, choices=["s", "m", "l", "x"])
    ap.add_argument("--role", choices=["backbone", "head", "both"], default="both")
    ap.add_argument("--precision", choices=["fp16", "int8"], default=None)
    ap.add_argument("--dla", dest="dla", action="store_true", default=None,
                    help="force DLA for the backbone (errors off-Jetson)")
    ap.add_argument("--no-dla", dest="dla", action="store_false",
                    help="force GPU-only backbone (skip DLA even on Orin)")
    args = ap.parse_args()

    profile = detect()
    cfg = load_config()
    roles = ("backbone", "head") if args.role == "both" else (args.role,)

    print("[hw] %s sm=%s dla_cores=%d power=%sW workspace=%.1fGiB tag=%s"
          % (profile.gpu_name, profile.sm, profile.dla_cores, profile.power_budget_w,
             profile.recommended_workspace_bytes / (1 << 30), profile.target_tag))
    if args.dla is None and cfg.get("dla", {}).get("enable", True) \
            and not profile.allow_dla and "backbone" in roles:
        print("[note] DLA enabled in config but this board has none -> GPU backbone.")

    built = build_variant(args.onnx_dir, args.variant, roles, profile, cfg,
                          precision=args.precision, dla=args.dla)
    print("[done] %s ->" % args.variant, {r: p.name for r, p in built.items()})


if __name__ == "__main__":
    main()
