"""Shared, model-free detector selection for Gibson entrypoints."""
from __future__ import annotations

from sparx_agency.core.mapping.detection.registry import default_detection_registry

#: The object-detection pipeline every Gibson entrypoint runs unless told otherwise.
#: YOLO-World only; Grounding DINO / BLIP-2 (``grounded_vlm``, ``hybrid``) stay
#: selectable but are not the default -- their stair verification is deferred work.
DEFAULT_DETECTOR_BACKEND = "yolo_world"


def add_detector_options(parser, *, url_default=None, required=False, backend_default=DEFAULT_DETECTOR_BACKEND):
    parser.add_argument("--detector-url", default=url_default, required=url_default is None)
    parser.add_argument("--detector-backend", choices=default_detection_registry().names(),
                        default=backend_default, required=required and backend_default is None,
                        help="Detector service mode to expect on --detector-url (default: %s)" % backend_default)
    parser.add_argument("--detector-timeout-s", type=float, default=30.0,
                        help="Bounded HTTP timeout; increase explicitly for CPU BLIP-2")


def detector_flags(args):
    flags = ["--detector-url", args.detector_url,
             "--detector-timeout-s", str(getattr(args, "detector_timeout_s", 30.0))]
    backend = getattr(args, "detector_backend", None)
    if backend is not None:
        flags += ["--detector-backend", backend]
    return flags
