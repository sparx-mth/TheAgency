"""Shared, model-free detector selection for Gibson entrypoints."""
from __future__ import annotations

from sparx_agency.core.mapping.detection.registry import default_detection_registry


def add_detector_options(parser, *, url_default=None, required=False, backend_default=None):
    parser.add_argument("--detector-url", default=url_default, required=url_default is None)
    parser.add_argument("--detector-backend", choices=default_detection_registry().names(),
                        default=backend_default, required=required)
    parser.add_argument("--detector-timeout-s", type=float, default=30.0,
                        help="Bounded HTTP timeout; increase explicitly for CPU BLIP-2")


def detector_flags(args):
    flags = ["--detector-url", args.detector_url,
             "--detector-timeout-s", str(getattr(args, "detector_timeout_s", 30.0))]
    backend = getattr(args, "detector_backend", None)
    if backend is not None:
        flags += ["--detector-backend", backend]
    return flags
