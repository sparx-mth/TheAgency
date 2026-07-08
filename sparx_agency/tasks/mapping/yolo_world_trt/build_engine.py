"""Build a YOLO-World TensorRT engine with a fully explicit, DLA-aware config.

Nothing is left to TensorRT defaults. From the :class:`BuildPolicy` this sets:
the precision (FP16 default, INT8 opt-in), the DLA target + core + GPU fallback,
the DLA SRAM / local-DRAM / global-DRAM memory pools, the workspace pool, and the
builder optimization level. A sibling ``.json`` manifest records the TensorRT
version, target SM, precision, IO, the baked class list, and how many layers were
DLA-eligible -- so the runtime can version-lock and you can see the DLA offload at
a glance.

Engines are locked to the exact GPU + TensorRT build that produced them and are
NOT portable. **DLA engines can only be built ON a Jetson** (the x86 dev box has
no DLA): if you force ``--dla`` off-Jetson the build errors; with the default
(``dla=None``) it silently falls back to a GPU FP16 engine so the pipeline is
still exercisable on a laptop.

Run (target device, TRT venv, PYTHONPATH = repo root):
    python -m sparx_agency.tasks.mapping.yolo_world_trt.build_engine \\
        --onnx .../engines/onnx/yolo_world_s.onnx --variant s
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
    """Parse an ONNX graph into ``network`` or raise with the parser errors."""
    parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))
    if not parser.parse(Path(onnx_path).read_bytes()):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed for %s: %s" % (onnx_path, errs))


def _configure_precision(config, trt, policy, calibrator):
    """Set FP16 (always) and, if requested, INT8 + calibrator."""
    config.set_flag(trt.BuilderFlag.FP16)        # floor for DLA; default for GPU
    if policy.use_int8:
        if calibrator is None:
            raise RuntimeError("INT8 requested but no calibrator was provided.")
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        config.int8_calibrator = calibrator


def _configure_dla(config, trt, policy):
    """Point the default device at the DLA, keep GPU fallback, size the pools.

    Returns the number of layers the builder reports as DLA-eligible (an estimate
    of the offload; the tail -- head decode, some Slice/Concat -- runs on GPU).
    """
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
        except (AttributeError, TypeError):       # older TRT: pool may be absent
            pass


def _count_dla_eligible(config, network):
    """Best-effort count of DLA-eligible layers (0 if the API is unavailable)."""
    try:
        return sum(int(config.can_run_on_DLA(network.get_layer(i)))
                   for i in range(network.num_layers))
    except Exception:  # noqa: BLE001
        return 0


def build_one(onnx_path, variant, out_dir, profile, policy, calibrator=None):
    """Parse one ONNX graph and build + serialize its engine. Returns the path."""
    import tensorrt as trt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)          # explicit-batch, static shape
    _parse(network, trt, onnx_path)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, policy.workspace_bytes)
    try:
        config.builder_optimization_level = policy.builder_optimization_level
    except AttributeError:
        pass

    _configure_precision(config, trt, policy, calibrator)
    dla_eligible = 0
    if policy.use_dla:
        _configure_dla(config, trt, policy)
        dla_eligible = _count_dla_eligible(config, network)

    cache_path = out_dir / ("timing_%s.cache" % profile.target_tag)
    existing = cache_path.read_bytes() if cache_path.exists() else b""
    timing = config.create_timing_cache(existing)
    config.set_timing_cache(timing, ignore_mismatch=False)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None for %s "
                           "(if DLA was forced off-Jetson this is expected)" % variant)

    device = "dla%d" % policy.dla_core if policy.use_dla else "gpu"
    engine_path = out_dir / ("yolo_world_%s.%s.%s.engine"
                             % (variant, policy.precision, device))
    engine_path.write_bytes(bytes(serialized))
    cache_path.write_bytes(bytes(timing.serialize()))
    _write_manifest(engine_path, variant, onnx_path, profile, policy, trt,
                    network.num_layers, dla_eligible)
    return engine_path


def _load_classes(onnx_path):
    """Load the baked prompt list written next to the ONNX by export_onnx."""
    sidecar = Path(onnx_path).with_name(Path(onnx_path).stem + ".classes.json")
    if sidecar.exists():
        return json.loads(sidecar.read_text())
    return {}


def _write_manifest(engine_path, variant, onnx_path, profile, policy, trt,
                    total_layers, dla_eligible):
    """Write the sibling ``.json`` the runtime version-locks against."""
    classes = _load_classes(onnx_path)
    meta = {
        "variant": variant,
        "precision": policy.precision,
        "device": "dla%d" % policy.dla_core if policy.use_dla else "gpu",
        "use_dla": policy.use_dla,
        "gpu_fallback": policy.gpu_fallback if policy.use_dla else None,
        "trt_version": str(trt.__version__),
        "sm": profile.sm,
        "target_tag": profile.target_tag,
        "gpu_name": profile.gpu_name,
        "power_budget_w": profile.power_budget_w,
        "imgsz_hw": list(policy.imgsz),
        "onnx_sha256": _sha256(onnx_path),
        "prompts": classes.get("prompts", []),
        "nc": classes.get("nc", 0),
        "conf_thresh": policy.conf_thresh,
        "iou_thresh": policy.iou_thresh,
        "max_det": policy.max_det,
        "total_layers": int(total_layers),
        "dla_eligible_layers": int(dla_eligible),
        "workspace_bytes": policy.workspace_bytes,
        "builder_optimization_level": policy.builder_optimization_level,
    }
    Path(str(engine_path) + ".json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="path to yolo_world_<variant>.onnx")
    ap.add_argument("--variant", required=True, choices=["s", "m", "l", "x"])
    ap.add_argument("--out-dir", default=None,
                    help="default: <onnx-dir>/../<target_tag>")
    ap.add_argument("--precision", choices=["fp16", "int8"], default=None)
    ap.add_argument("--dla", dest="dla", action="store_true", default=None,
                    help="force DLA (errors off-Jetson)")
    ap.add_argument("--no-dla", dest="dla", action="store_false",
                    help="force GPU-only (skip DLA even on Orin)")
    ap.add_argument("--calib-npz", default=None, help="required for --precision int8")
    args = ap.parse_args()

    profile = detect()
    cfg = load_config()
    policy = build_policy(args.variant, profile, config=cfg,
                          precision=args.precision, dla=args.dla)

    onnx_path = Path(args.onnx)
    out_dir = Path(args.out_dir) if args.out_dir else \
        onnx_path.parent.parent / profile.target_tag

    print("[hw] %s sm=%s dla_cores=%d power=%sW workspace=%.1fGiB tag=%s"
          % (profile.gpu_name, profile.sm, profile.dla_cores,
             profile.power_budget_w, profile.workspace_bytes / (1 << 30),
             profile.target_tag))
    print("[policy] variant=%s imgsz=%dx%d precision=%s use_dla=%s core=%d opt=%d"
          % (policy.variant, policy.imgsz[0], policy.imgsz[1], policy.precision,
             policy.use_dla, policy.dla_core, policy.builder_optimization_level))
    if args.dla is None and cfg.get("dla", {}).get("enable", True) \
            and not profile.allow_dla:
        print("[note] DLA requested in config but this board has none -> GPU engine.")

    calibrator = None
    if policy.use_int8:
        raise SystemExit("INT8 calibration is not wired yet; use --precision fp16 "
                         "(the DLA default). See README 'INT8' for the plan.")

    path = build_one(onnx_path, args.variant, out_dir, profile, policy, calibrator)
    print("[ok] built", path.name, "->", out_dir)


if __name__ == "__main__":
    main()
