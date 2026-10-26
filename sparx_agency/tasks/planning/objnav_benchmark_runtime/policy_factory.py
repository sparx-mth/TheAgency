"""Build the one search policy every benchmark entrypoint runs, and prove its services are the ones recorded.

A benchmark adapter (Gibson today; HM3D, MP3D or another simulator tomorrow)
owns its dataset, protocol, scorer and label table. What it must NOT own is
how the method is assembled: which services it needs, how their identities
are pinned, which settings override which, and what the frozen configuration
records about it. Two entrypoints carrying a copy each of this function is
exactly how the *same* algorithm stops being the same across benchmarks; there
is one builder, and an adapter's CLI calls :func:`build_policy` with the one
thing that differs per dataset -- its label table, whose ``vocabulary()`` the
detector service must have been started with.

Order of precedence, lowest first: ``RPTSettings`` defaults < ``defaults``
(the entrypoint's: a development protocol lifting ``allow_stair_traversal``)
< ``--policy-config`` < the explicit command-line switches (``--explorer``,
``--segmentation``), which must agree with the policy config when both are
given.

``--perception ground-truth`` swaps the two perception services for the
simulator's own annotations (``ground_truth/``): the adapter passes the
:class:`~sparx_agency.tasks.planning.objnav_benchmark_runtime.ground_truth.source.GroundTruthSource`
its environment publishes frames to, the detector and the segmenter are built
on it, ``segmentation`` is on unless the policy config says otherwise (the
instance masks are the point), and the run is **non-publishable**: it is the
method's upper bound under ideal perception, not a benchmark result.

Importing this module loads no service client, model or simulator; the HTTP
clients are imported when a policy is built.
"""
from __future__ import annotations

from dataclasses import asdict
import json

from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.tasks.planning.objnav_benchmark_runtime.detector_options import (
    PERCEPTION_GROUND_TRUTH, PERCEPTION_SERVICE)

#: The ``--agent`` choices every adapter CLI offers, and what each is.
AGENT_CHOICES = ("rpt", "falcon-yolo", "stop")
AGENT_HELP = ("rpt: the room-search method (LLM + RPT* + frontier/FALCON + closing). falcon-yolo: the "
              "exploration baseline -- goal-free planar FALCON, the same YOLO perception and the same "
              "closing, no LLM (core/planning/object_search/baselines/falcon_yolo.py). stop: the plumbing diagnostic")


class StopDiagnostic:
    """Non-privileged plumbing check, never labelled as the search method."""
    name = "diagnostic-stop"

    def reset(self, episode, target):
        pass

    def plan(self, observation):
        return NavigationCommand.stop_here()


def build_policy(args, label_mapper, defaults=None, ground_truth=None):
    """The policy and its frozen description.

    Args:
        args: Parsed CLI with the method options (``agent``, ``policy_config``, ``explorer``,
            ``seed``, the detector options of ``detector_options.add_detector_options``);
            ``--policy-config`` JSON overrides every default.
        label_mapper: The benchmark's :class:`~sparx_agency.core.planning.objnav.labels.table_mapper.TableLabelMapper`;
            its ``vocabulary()`` is what the detector service must have been started with, in that order.
        defaults: ``RPTSettings`` fields an entrypoint sets below the policy
            config -- the multi-storey development protocol lifts
            ``allow_stair_traversal``, whose benchmark default is False.
        ground_truth: With ``--perception ground-truth``, the
            :class:`~...ground_truth.source.GroundTruthSource` the adapter's
            environment publishes semantic frames to. Required then, refused
            otherwise.

    Returns:
        ``(policy, method)``: the :class:`~sparx_agency.core.planning.objnav.interfaces.policy.SearchPolicy`
        and the ``method`` entry of the frozen configuration (its settings, the services'
        identities, ``publishable``).

    Raises:
        ValueError: Options that contradict each other or the policy config.
        RuntimeError: A service is unreachable, misconfigured or hides the detections the
            method needs; or ground-truth perception was asked of an adapter that cannot
            publish semantic frames.
    """
    if args.agent == "stop":
        return StopDiagnostic(), {"method": "diagnostic-stop", "publishable": False}
    baseline = args.agent == "falcon-yolo"
    oracle = getattr(args, "perception", PERCEPTION_SERVICE) == PERCEPTION_GROUND_TRUTH
    if oracle and ground_truth is None:
        raise RuntimeError("--perception ground-truth needs an environment that publishes the simulator's semantic "
                           "frames (the HM3D adapter); this entrypoint has none to offer")
    if ground_truth is not None and not oracle:
        raise ValueError("a ground-truth source was given but --perception is %r" % (args.perception,))
    if oracle and getattr(args, "segmenter_url", None):
        raise ValueError("--perception ground-truth reads its masks from the simulator; --segmenter-url contradicts it")
    from sparx_agency.core.planning.object_search.settings import RPTSettings
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.services.llm import public_service_url
    overrides = json.loads(args.policy_config.read_text()) if args.policy_config else {}
    if not isinstance(overrides, dict):
        raise ValueError("--policy-config must hold a JSON object")
    baseline_overrides = overrides.pop("baseline", None)
    if baseline_overrides is not None and not baseline:
        raise ValueError("--policy-config holds a \"baseline\" object, which only --agent falcon-yolo reads")
    if baseline:
        # The baseline explores with planar FALCON and nothing else: no --explorer frontier under it.
        if (args.explorer or "falcon") != "falcon" or overrides.get("local_exploration", "falcon") != "falcon":
            raise ValueError("--agent falcon-yolo explores with FALCON; --explorer / local_exploration cannot be frontier")
        overrides["local_exploration"] = "falcon"
    elif args.explorer is not None:
        if overrides.get("local_exploration", args.explorer) != args.explorer:
            raise ValueError("--explorer disagrees with --policy-config")
        overrides["local_exploration"] = args.explorer
    if getattr(args, "segmentation", None):
        if overrides.get("segmentation", True) is not True:
            raise ValueError("--segmentation disagrees with --policy-config")
        overrides["segmentation"] = True
    if oracle:
        # The exact instance masks are what the oracle adds over boxes; a policy config saying
        # segmentation=false is honoured (ground-truth boxes read the old way), never silently flipped.
        overrides.setdefault("segmentation", True)
    settings = RPTSettings(**dict(defaults or {}, **dict(overrides, seed=args.seed)))
    config = client = None
    services = []
    if not baseline:
        from sparx_agency.core.mapping.topology.llm_client import LLMClient, LLMConfig
        from sparx_agency.tasks.planning.objnav_benchmark_runtime.services.llm import VerifiedLLMClient
        config = LLMConfig.from_env()
        config.seed = args.seed
        client = VerifiedLLMClient(LLMClient(config))
        services.append(("LLM", client))
    segmenter = None
    if oracle:
        from sparx_agency.tasks.planning.objnav_benchmark_runtime.ground_truth.perception import (
            DEFAULT_MIN_PIXELS, GroundTruthDetector, GroundTruthSegmenter)
        min_pixels = getattr(args, "ground_truth_min_pixels", None)
        detector = GroundTruthDetector(ground_truth, DEFAULT_MIN_PIXELS if min_pixels is None else min_pixels)
        services.append(("detector", detector))
        if settings.segmentation:
            segmenter = GroundTruthSegmenter(detector)
            services.append(("segmenter", segmenter))
    else:
        from sparx_agency.tasks.planning.objnav_benchmark_runtime.services.http_detector import HttpDetector
        if getattr(args, "ground_truth_min_pixels", None) is not None:
            raise ValueError("--ground-truth-min-pixels is the oracle's; --perception is %r" % (args.perception,))
        detector = HttpDetector(public_service_url(args.detector_url), label_mapper.vocabulary(),
                                expected_backend=args.detector_backend,
                                timeout_s=getattr(args, "detector_timeout_s", 30.0))
        services.append(("detector", detector))
        if settings.segmentation:
            from sparx_agency.tasks.planning.objnav_benchmark_runtime.services.http_segmenter import HttpSegmenter
            if not getattr(args, "segmenter_url", None):
                raise RuntimeError("segmentation is on (--segmentation / policy config) but no --segmenter-url was given; "
                                   "start segmentation_server.py and name it, or turn the setting off")
            segmenter = HttpSegmenter(public_service_url(args.segmenter_url), timeout_s=getattr(args, "segmenter_timeout_s", 30.0),
                                      expected_model=getattr(args, "segmenter_model", None))
            services.append(("segmenter", segmenter))
    problems, identities = [], {}
    for name, service in services:
        try:
            identities[name] = service.health()
        except Exception as exc:
            problems.append("%s: %s" % (name, exc))
    if problems:
        raise RuntimeError("; ".join(problems))
    if baseline:
        from sparx_agency.core.planning.object_search.baselines.falcon_yolo import BaselineSettings, FalconYoloBaseline
        if baseline_overrides is not None and not isinstance(baseline_overrides, dict):
            raise ValueError("--policy-config \"baseline\" must be a JSON object of BaselineSettings fields")
        policy = FalconYoloBaseline(detector, settings, segmenter=segmenter,
                                    baseline=BaselineSettings(**(baseline_overrides or {})))
        floor = settings.detection_confidence
    else:
        from sparx_agency.core.planning.object_search.policy import RPTSearchPolicy
        policy = RPTSearchPolicy(detector, client, settings, segmenter=segmenter)
        floor = min(settings.detection_confidence, policy.door_settings.confidence)
    emitted = float(identities["detector"]["metadata"]["detector_config"]["conf_thresh"])
    if emitted > floor:
        raise RuntimeError("Detector emission threshold hides door candidates; restart with --conf 0.05")
    info = policy.configuration()
    if config is not None:
        llm = asdict(config)
        llm.pop("api_key", None)
        info.update(llm=llm, llm_identity=identities["LLM"])
    if oracle:
        # No service, no URL, no timeout: the identity says what answered. A perception upper
        # bound is a diagnostic of the method, never a benchmark result -- publishable=False,
        # like the stop diagnostic and Gibson's target override. A service run's record gains no
        # key here: a resume compares the recorded config verbatim, and runs recorded before the
        # oracle existed must stay resumable.
        info.update(perception=PERCEPTION_GROUND_TRUTH, detector=identities["detector"], detector_url=None,
                    publishable=False,
                    ground_truth={"bypassed": ["detector", "segmenter"] if segmenter is not None else ["detector"],
                                  "min_pixels": detector.min_pixels, "confidence": 1.0,
                                  "goal_instances": "the publisher's goal object ids per scene and category",
                                  "context_labels": "HM3D-Semantics raw names via ground_truth/hm3d_categories.py",
                                  "category_table_sha256": identities["detector"]["metadata"]["checkpoint_sha256"]})
        if segmenter is not None:
            info.update(segmenter=identities["segmenter"], segmenter_url=None)
        return policy, info
    info.update(detector=identities["detector"], detector_url=args.detector_url, detector_timeout_s=detector.timeout_s)
    if segmenter is not None:
        info.update(segmenter=identities["segmenter"], segmenter_url=args.segmenter_url, segmenter_timeout_s=segmenter.timeout_s)
    return policy, info
