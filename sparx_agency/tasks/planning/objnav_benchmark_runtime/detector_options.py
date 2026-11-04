"""Shared, model-free perception selection for every benchmark entrypoint: the detector and the segmenter services, or the simulator's ground truth."""
from __future__ import annotations

from sparx_agency.core.mapping.detection.registry import default_detection_registry

#: The object-detection pipeline every benchmark entrypoint runs unless told otherwise.
#: YOLO-World only; Grounding DINO / BLIP-2 (``grounded_vlm``, ``hybrid``) stay
#: selectable but are not the default -- their stair verification is deferred work.
DEFAULT_DETECTOR_BACKEND = "yolo_world"

#: ``--perception``: the served detector/segmenter (the benchmark), or the simulator's own annotations
#: standing in for both (the perception upper bound; non-publishable).
PERCEPTION_SERVICE = "service"
PERCEPTION_GROUND_TRUTH = "ground-truth"
PERCEPTION_CHOICES = (PERCEPTION_SERVICE, PERCEPTION_GROUND_TRUTH)


def add_detector_options(parser, *, url_default=None, required=False, backend_default=DEFAULT_DETECTOR_BACKEND):
    parser.add_argument("--detector-url", default=url_default, required=url_default is None)
    parser.add_argument("--detector-backend", choices=default_detection_registry().names(),
                        default=backend_default, required=required and backend_default is None,
                        help="Detector service mode to expect on --detector-url (default: %s)" % backend_default)
    parser.add_argument("--detector-timeout-s", type=float, default=30.0,
                        help="Bounded HTTP timeout; increase explicitly for CPU BLIP-2")
    add_segmenter_options(parser)
    add_perception_options(parser)


def add_perception_options(parser):
    """``--perception``: which perception the method runs on -- the services, or the simulator's ground truth.

    ``ground-truth`` bypasses the detector AND the segmenter: boxes, labels and
    instance masks come from the simulator's semantic sensor
    (``ground_truth/``), the method is otherwise unchanged, and the run is
    recorded as non-publishable. Only an adapter whose environment publishes
    semantic frames (HM3D today) can honour it.
    """
    parser.add_argument("--perception", choices=PERCEPTION_CHOICES, default=PERCEPTION_SERVICE,
                        help="service (default): the detector at --detector-url and, with --segmentation, the "
                             "segmenter. ground-truth: the simulator's semantic annotations in place of both -- "
                             "the method's upper bound under ideal perception, never a benchmark result")
    parser.add_argument("--ground-truth-min-pixels", type=int, default=None, metavar="N",
                        help="With --perception ground-truth: an annotated instance covering fewer than N pixels is "
                             "not reported (default 64, an 8x8 patch at 640x480); the oracle's declared resolution floor")


def detector_flags(args):
    """The perception flags a campaign forwards to every ``gibson.run`` (or another adapter's ``run``) job: the detector's, the segmenter's and the perception mode's."""
    flags = ["--detector-url", args.detector_url,
             "--detector-timeout-s", str(getattr(args, "detector_timeout_s", 30.0))]
    backend = getattr(args, "detector_backend", None)
    if backend is not None:
        flags += ["--detector-backend", backend]
    return flags + segmenter_flags(args) + perception_flags(args)


def perception_flags(args):
    flags = []
    perception = getattr(args, "perception", PERCEPTION_SERVICE)
    if perception != PERCEPTION_SERVICE:
        flags += ["--perception", perception]
    if getattr(args, "ground_truth_min_pixels", None) is not None:
        flags += ["--ground-truth-min-pixels", str(args.ground_truth_min_pixels)]
    return flags


def add_segmenter_options(parser):
    """The box-prompted segmentation service (``segmentation_server.py``) behind the ``segmentation`` setting.

    ``--segmentation`` turns the setting on from the command line (the same
    override route as ``--explorer``: it must agree with ``--policy-config``);
    ``--segmenter-url`` names the service, which the run needs only when the
    setting is on and refuses to start without. OFF by default -- the
    comparison experiment of 2026-10-12.
    """
    parser.add_argument("--segmentation", action="store_true", default=None,
                        help="Segment the target's boxes (Segment Anything service at --segmenter-url) and read "
                             "their depth on the object's own pixels; off by default")
    parser.add_argument("--segmenter-url", default=None,
                        help="The segmentation service; required with --segmentation (or a policy config that turns it on)")
    parser.add_argument("--segmenter-model", default=None,
                        help="Checkpoint file name the service must serve (mobile_sam.pt, sam2.1_t.pt); any when omitted")
    parser.add_argument("--segmenter-timeout-s", type=float, default=30.0, help="Bounded HTTP timeout per frame")


def segmenter_flags(args):
    flags = []
    if getattr(args, "segmentation", None):
        flags.append("--segmentation")
    if getattr(args, "segmenter_url", None):
        flags += ["--segmenter-url", args.segmenter_url, "--segmenter-timeout-s", str(getattr(args, "segmenter_timeout_s", 30.0))]
    if getattr(args, "segmenter_model", None):
        flags += ["--segmenter-model", args.segmenter_model]
    return flags
