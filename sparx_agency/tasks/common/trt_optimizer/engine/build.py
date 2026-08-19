"""Build one TensorRT engine, then refuse to trust that it worked.

The builder knobs themselves live in
:mod:`..trt_optimizer.engine.builder_config`; this module is the orchestration
around them -- parse, build, verify, serialize, and write the sidecar the
runtime version-locks against.

The verification step is the reason this module exists rather than being three
lines at a call site. On TensorRT 11 every network is strongly typed and the
engine's precision is exactly whatever the ONNX carried, so
``build_serialized_network`` returning bytes says nothing whatsoever about
whether you got FP16 -- ask for FP16, hand it an FP32 graph, and you get a
perfectly valid FP32 engine and a cheerful log. Worse, an INT4 or NVFP4 graph
can be silently widened back to FP16 by the builder when no tactic exists at
that width. So the engine is deserialized immediately, its weight bytes per
parameter are measured, and a mismatch is a hard failure.

Engines are written as ``<key>.<precision>.engine`` with a sibling ``.json``
recording the TensorRT version, target SM and build notes, because a serialized
engine deserializes only under the exact TensorRT build and compute capability
that produced it -- down to the patch level, which JetPack point releases bump.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from sparx_agency.tasks.common.trt_optimizer.engine import builder_config
from sparx_agency.tasks.common.trt_optimizer.engine import calibrate
from sparx_agency.tasks.common.trt_optimizer.engine import precision as prec
from sparx_agency.tasks.common.trt_optimizer.export.onnx_export import sha256

#: Re-exported so callers configure a build without importing two modules.
BuildOptions = builder_config.BuildOptions


@contextmanager
def _cuda_context(needed):
    """Hold a CUDA primary context when a calibrator allocates device memory.

    Only the INT8 calibrator path needs one; an FP16 build lets TensorRT manage
    its own context and never touches pycuda.
    """
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


def _parse(network, trt, onnx_path):
    """Parse ONNX into the network, or raise with every parser error."""
    parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))
    if not parser.parse(Path(onnx_path).read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed for %s:\n  %s"
                           % (onnx_path, "\n  ".join(errors)))


def _create_network(builder, trt, strongly_typed):
    """Create the network definition for this TensorRT generation."""
    if strongly_typed:
        flag = getattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED", None)
        return builder.create_network(1 << int(flag)) if flag is not None \
            else builder.create_network(0)
    return builder.create_network(0)


def build_engine(onnx_path, out_dir, target, options=None, param_count=None,
                 calibrator=None, spec=None):
    """Build, verify and serialize one engine. Returns the engine path.

    Args:
        onnx_path: the FP32 ONNX produced by the export stage.
        out_dir: per-target engine directory (``engines/<target_tag>/``).
        target: the :class:`..target.Target` doing the building.
        options: :class:`BuildOptions`; defaults used when omitted.
        param_count: parameter count of the source graph. Supplying it enables
            the post-build precision verification, which is the only check that
            catches a silent widening back to FP32.
        calibrator: INT8 calibrator, on a TensorRT that still has them.
        spec: the :class:`..spec.GraphSpec` this graph came from. Required when
            the graph has dynamic axes -- it carries the min/opt/max profile
            TensorRT needs. A static graph does not need it.

    Returns:
        Path to the written ``.engine``.

    Raises:
        RuntimeError: on a parse failure, a build failure, or a built engine
            whose measured precision does not match the request.
    """
    import tensorrt as trt

    options = options or builder_config.BuildOptions()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = Path(onnx_path).stem
    notes = []

    if options.precision in calibrate.QDQ_PRECISIONS:
        # Pre-flight, not mid-build: an un-quantized graph parses perfectly and
        # builds a valid engine at the WRONG precision, so the time to find out
        # the toolchain cannot reach INT8 is before any of the work, not after.
        calibrate.require_int8_buildable(trt)

    strongly_typed = prec.is_strongly_typed(trt)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    parse_path = _bake(onnx_path, out_dir, key, options, target, strongly_typed,
                       notes)

    weight_bytes = Path(parse_path).stat().st_size
    config = builder.create_builder_config()
    builder_config.configure(config, trt, target, options, weight_bytes, notes)

    network, parse_path = _parse_network(builder, trt, parse_path, onnx_path,
                                         out_dir, key, options, strongly_typed,
                                         notes)
    if not strongly_typed:
        _apply_weak_precision(config, trt, options, calibrator, notes)

    _add_optimization_profile(builder, config, trt, spec, notes)

    cache_path = Path(options.timing_cache or
                      out_dir / ("timing_%s.cache" % target.target_tag))
    timing = builder_config.load_timing_cache(config, trt, cache_path, notes)

    with _cuda_context(needed=calibrator is not None):
        serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(
            "build_serialized_network returned None for %s. Re-run with a "
            "trt.Logger(VERBOSE) to see the builder's reason." % key)

    engine_path = out_dir / ("%s.%s.engine" % (key, options.precision))
    engine_path.write_bytes(bytes(serialized))
    cache_path.write_bytes(bytes(timing.serialize()))

    verification = _verify(engine_path, trt, options.precision, param_count,
                           notes, out_dir, key, options)
    _write_sidecar(engine_path, key, onnx_path, target, options, notes,
                   verification)
    return engine_path


def _apply_weak_precision(config, trt, options, calibrator, notes):
    """Set the precision flags a weakly-typed TensorRT (<= 10) needs.

    On the Orin's TensorRT 10 stack precision is *not* carried by the ONNX -- it
    is a builder flag, and the graph was parsed as FP32. Without this the
    builder happily produces an FP32 engine for a caller that asked for FP16,
    which is the quiet failure the strongly-typed path is protected against by
    the post-build byte-per-parameter check.

    Raises:
        RuntimeError: if the requested precision has no flag on this build, or
            INT8 was requested with no calibrator to quantize against.
    """
    precision = options.precision
    if precision == "fp32":
        return
    flag = {"fp16": "FP16", "bf16": "BF16", "int8": "INT8",
            "fp8": "FP8"}.get(precision)
    if flag is None or not builder_config.set_flag(config, trt, flag):
        raise RuntimeError(
            "TensorRT %s is weakly typed but has no BuilderFlag for %s, so the "
            "engine would silently be built FP32."
            % (trt.__version__, precision))
    notes.append("weakly-typed TensorRT: precision set with BuilderFlag.%s"
                 % flag)
    if precision == "int8":
        if calibrator is None:
            raise RuntimeError(
                "INT8 on a weakly-typed TensorRT needs a calibrator; none was "
                "supplied, and an uncalibrated INT8 engine has no valid "
                "dynamic ranges.")
        config.int8_calibrator = calibrator
        builder_config.set_flag(config, trt, "PREFER_PRECISION_CONSTRAINTS")


def _bake(onnx_path, out_dir, key, options, target, strongly_typed, notes):
    """Produce the graph to parse, walking the FP16 ladder on a conversion error.

    ``onnxconverter_common`` does not merely produce a graph TensorRT might
    reject -- it can fail outright on one, and which rung it fails on depends on
    the keep-list, because blocking an op changes the cast topology the
    converter then has to clean up. Measured on a torchvision segmentation
    model: the strongest keep-list raises ``ValueError: The downstream node of
    the second cast node should be graph output``, and a weaker one converts.

    So conversion is tried rung by rung exactly as parsing is.

    Raises:
        PrecisionUnavailable: when no rung converts, naming the alternatives.
    """
    if not strongly_typed:
        return Path(onnx_path)
    out_path = out_dir / ("%s.%s.onnx" % (key, options.precision))
    if options.precision != "fp16":
        return Path(prec.bake_precision(onnx_path, out_path, options.precision))

    failures = []
    for label, keep_ops in prec.FP16_LADDER:
        try:
            path = Path(prec.bake_precision(onnx_path, out_path, "fp16",
                                            keep_ops=keep_ops))
        except Exception as exc:  # noqa: BLE001  (any converter failure is a rung failure)
            failures.append("%s -> %s: %s" % (label, exc.__class__.__name__, exc))
            continue
        notes.append("precision baked into the ONNX (fp16, rung %r); TensorRT "
                     "%s is strongly typed and has no precision flags"
                     % (label, target.trt_version))
        if failures:
            notes.append("earlier FP16 rung(s) failed to convert: %s"
                         % "; ".join(failures))
        return path
    raise prec.PrecisionUnavailable(
        "no FP16 conversion of %s succeeded on this toolchain. Attempts:\n  - %s\n"
        "Options: build this graph FP32 (mark it precision_sensitive), install "
        "nvidia-modelopt whose autocast is a different implementation, or "
        "simplify the graph before export." % (key, "\n  - ".join(failures)))


def _parse_network(builder, trt, parse_path, onnx_path, out_dir, key, options,
                   strongly_typed, notes):
    """Parse the graph, walking down the FP16 ladder if the types clash.

    Keeping LayerNorm/Softmax/ReduceMean in FP32 is better numerically but can
    leave a Half/Float boundary the parser refuses. Rather than abandoning FP16
    for the whole engine -- which is the workaround the older builders in this
    repo use -- try the next rung down and record which one was used.

    Returns:
        ``(network, parse_path)`` for the graph that actually parsed.

    Raises:
        RuntimeError: the original parser error, if it was not a type clash or
            if no rung of the ladder parsed.
    """
    network = _create_network(builder, trt, strongly_typed)
    try:
        _parse(network, trt, parse_path)
        return network, parse_path
    except RuntimeError as first:
        if not (strongly_typed and options.precision == "fp16"
                and prec.is_type_mismatch(first)):
            raise
        notes.append("FP16 rung 0 (%s) produced a graph the parser rejected on "
                     "a Half/Float boundary; walking down the ladder"
                     % prec.FP16_LADDER[0][0])
        for label, keep_ops in prec.FP16_LADDER[1:]:
            candidate = Path(prec.bake_precision(
                onnx_path, out_dir / ("%s.fp16.onnx" % key), "fp16",
                keep_ops=keep_ops))
            network = _create_network(builder, trt, strongly_typed)
            try:
                _parse(network, trt, candidate)
            except RuntimeError:
                continue
            notes.append("FP16 conversion used rung %r; the stronger keep-list "
                         "did not survive the parser" % label)
            return network, candidate
        raise


def _add_optimization_profile(builder, config, trt, spec, notes):
    """Attach the min/opt/max profile a dynamic graph needs, or verify none is.

    TensorRT tunes tactics for ``opt`` and guarantees correctness across
    ``[min, max]``. Getting ``opt`` wrong is the common mistake: a profile whose
    ``opt`` is far from the size actually run can be slower than a static
    engine, because every tactic was chosen for a shape that never appears.

    Raises:
        RuntimeError: if the graph declares dynamic axes but no spec was passed,
            since TensorRT would otherwise fail with a much less obvious message
            about a missing profile.
    """
    if spec is None or not getattr(spec, "is_dynamic", False):
        return
    profile = builder.create_optimization_profile()
    for name, shape_profile in spec.profiles.items():
        profile.set_shape(name, tuple(shape_profile.min),
                          tuple(shape_profile.opt), tuple(shape_profile.max))
        notes.append("profile %s: min=%s opt=%s max=%s"
                     % (name, tuple(shape_profile.min), tuple(shape_profile.opt),
                        tuple(shape_profile.max)))
    config.add_optimization_profile(profile)
    notes.append("dynamic engine: tactics tuned for the opt shapes above; a "
                 "profile switch at run time costs tail latency, so keep the "
                 "range no wider than the input genuinely varies")


def _verify(engine_path, trt, precision, param_count, notes, out_dir, key,
            options):
    """Deserialize the fresh engine, verify precision, dump layer info."""
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("Engine %s built but does not deserialize."
                           % engine_path.name)
    if options.detailed_profiling:
        _dump_layer_info(engine, trt, out_dir / ("%s.layers.json" % key), notes)
    if param_count:
        ok, bpe, message = prec.verify_engine_precision(
            engine, precision, param_count, trt_module=trt)
        notes.append(message)
        if not ok:
            raise RuntimeError(
                "Engine %s was requested at %s but measured %.2f bytes per "
                "parameter: %s. A successful build proves nothing about "
                "precision on a strongly-typed TensorRT."
                % (engine_path.name, precision, bpe, message))
        # bpe is None when verify_engine_precision SKIPPED the check (this
        # TensorRT exposes no EngineStat.TOTAL_WEIGHTS_SIZE). ok is True there
        # so the build is not blocked, but the sidecar must not record a check
        # that never ran as a check that passed.
        return {"bytes_per_param": bpe, "verified": bpe is not None,
                "skipped": bpe is None}
    notes.append("precision NOT verified: no param_count supplied")
    return {"verified": False, "skipped": False}


def _dump_layer_info(engine, trt, out_path, notes):
    """Write the engine inspector's JSON so a later regression is diagnosable."""
    try:
        inspector = engine.create_engine_inspector()
        info = inspector.get_engine_information(
            trt.LayerInformationFormat.JSON)
        Path(out_path).write_text(info)
        notes.append("layer information written to %s" % out_path.name)
    except Exception as exc:  # noqa: BLE001  (inspector is a diagnostic, not the build)
        notes.append("engine inspector unavailable: %s" % exc.__class__.__name__)


def _write_sidecar(engine_path, key, onnx_path, target, options, notes,
                   verification):
    """Write the ``.engine.json`` the runtime version-locks against."""
    import tensorrt as trt

    meta = {
        "engine_key": key,
        "precision": options.precision,
        "trt_version": str(trt.__version__),
        "sm": target.hardware.sm,
        "target_tag": target.target_tag,
        "gpu_name": target.hardware.gpu_name,
        "onnx_sha256": sha256(onnx_path),
        "builder_optimization_level": options.optimization_level,
        "max_aux_streams": options.max_aux_streams,
        "verification": verification,
        "build_notes": notes,
    }
    Path(str(engine_path) + ".json").write_text(json.dumps(meta, indent=2))
