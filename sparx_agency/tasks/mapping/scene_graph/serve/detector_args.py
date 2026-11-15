"""Lightweight detector CLI/config parsing, shared by serving and profiling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from sparx_agency.core.mapping.detection.registry import default_detection_registry
from sparx_agency.core.mapping.detection.yolo_world import YoloWorldConfig
from sparx_agency.tasks.mapping.scene_graph.serve.contract import DEFAULT_HOSPITAL_VOCABULARY, DEFAULT_PORT


def parse_args(argv=None):
    """Config values precede CLI overrides; model paths are relative to the JSON."""
    parser = argparse.ArgumentParser(description="Selectable local detection and visual verification")
    parser.add_argument("--config", help="JSON settings; command-line flags override")
    parser.add_argument("--backend", choices=default_detection_registry().names(), default="yolo_world")
    parser.add_argument("--model", help="Legacy/main local checkpoint override")
    parser.add_argument("--grounding-model", help="Local Grounding DINO snapshot directory")
    parser.add_argument("--blip2-model", help="Local BLIP-2 FLAN-T5 snapshot directory")
    parser.add_argument("--yolo-model", help="Local YOLO checkpoint, used only when selected")
    parser.add_argument("--clip-model", help="Local CLIP ViT-B/32 weights; required for offline hybrid mode")
    parser.add_argument("--device", default="cuda:0", help="Use cpu explicitly alongside a simulator")
    parser.add_argument("--allow-shared-gpu", action=argparse.BooleanOptionalAction, default=False,
                        help="Explicit operator authorization to share CUDA with rendering/desktop; preflight owners and memory first")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--conf", type=float, default=None, help="Explicit grounded emission floor, not STOP confidence")
    parser.add_argument("--yolo-conf", type=float, help="Hybrid YOLO emission floor; defaults to --conf")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--shortest-edge", type=int, default=800)
    parser.add_argument("--longest-edge", type=int, default=1333)
    parser.add_argument("--chunk-size", type=int, default=80)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--verify-labels", default="*", help="Default '*' verifies every class; comma-list opts into selective verification")
    parser.add_argument("--max-verifications", type=int, default=8, help="Per-frame box budget; overflow abstains")
    parser.add_argument("--verification-iou", type=float, default=0.6)
    parser.add_argument("--blip2-max-tokens", type=int, default=12)
    parser.add_argument("--classes", default=",".join(DEFAULT_HOSPITAL_VOCABULARY))
    parser.add_argument("--selftest", action="store_true")
    arguments = list(sys.argv[1:] if argv is None else argv)
    preliminary = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    preliminary.add_argument("--config")
    config_arg, _ = preliminary.parse_known_args(arguments)
    arguments = _config_arguments(parser, config_arg.config) + arguments
    args = parser.parse_args(arguments)
    _complete_models(parser, args)
    return args


def _config_arguments(parser, filename):
    if filename is None:
        return []
    path = Path(filename).expanduser().resolve()
    try:
        values = json.loads(path.read_text())
        if not isinstance(values, dict):
            raise ValueError("Detector config must be a JSON object")
        allowed = {action.dest for action in parser._actions} - {"config", "help", "selftest"}
        if set(values) - allowed:
            raise ValueError("Unknown detector settings: %s" % sorted(set(values) - allowed))
        arguments = []
        for name, value in values.items():
            if value is None:
                continue
            if name == "allow_shared_gpu":
                if not isinstance(value, bool):
                    raise ValueError("Setting allow_shared_gpu must be a boolean")
                arguments.append("--allow-shared-gpu" if value else "--no-allow-shared-gpu")
                continue
            if isinstance(value, (dict, list, bool)):
                raise ValueError("Setting %s must be a string or number" % name)
            if name in ("model", "grounding_model", "blip2_model", "yolo_model", "clip_model"):
                if not str(value).strip():
                    raise ValueError("Setting %s must be a non-empty local path" % name)
                model = Path(str(value)).expanduser()
                value = str(model if model.is_absolute() else path.parent / model)
            arguments.append("--%s=%s" % (name.replace("_", "-"), value))
        return arguments
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


def _complete_models(parser, args):
    for name in ("model", "grounding_model", "blip2_model", "yolo_model", "clip_model"):
        value = getattr(args, name)
        if value is not None and not value.strip():
            parser.error("--%s must be a non-empty local path" % name.replace("_", "-"))
    if args.backend == "yolo_world":
        args.model = args.model or args.yolo_model or YoloWorldConfig().model_path
    elif args.backend in ("grounding_dino", "grounded_vlm", "hybrid"):
        args.model = args.model or args.grounding_model
    if not args.model and not args.selftest:
        parser.error("--model (or --grounding-model) is required for %s" % args.backend)
    if args.conf is None:
        if args.backend != "yolo_world" and not args.selftest:
            parser.error("--conf is required for %s; do not inherit YOLO thresholds" % args.backend)
        args.conf = 0.25
    if args.backend in ("grounded_vlm", "hybrid") and not args.blip2_model and not args.selftest:
        parser.error("--blip2-model is required for visual verification")
    if args.backend == "hybrid":
        args.yolo_model = args.yolo_model or YoloWorldConfig().model_path
        if not args.clip_model and not args.selftest:
            parser.error("--clip-model is required for offline hybrid mode")
    for name in ("model", "grounding_model", "blip2_model", "yolo_model", "clip_model"):
        value = getattr(args, name)
        if value is not None:
            if not value.strip():
                parser.error("--%s must be a non-empty local path" % name.replace("_", "-"))
            setattr(args, name, str(Path(value).expanduser()))
    if args.verify_labels != "*" and not args.verify_labels.strip():
        parser.error("--verify-labels must list categories or be '*'")




