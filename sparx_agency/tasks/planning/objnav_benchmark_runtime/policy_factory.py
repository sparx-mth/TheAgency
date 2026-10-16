"""Build the one search policy every benchmark entrypoint runs, and prove its services are the ones recorded.

``gibson.run`` and ``hm3d.run`` used to carry a copy each of this function; the
copies had already drifted (one knew the segmenter, the other did not; one set
``stop_distance_m`` the other left alone), which is exactly how the *same*
algorithm stops being the same across benchmarks. There is one builder now. A
benchmark passes in its label table and, at most, the embodiment of its own
agent; every method knob is ``RPTSettings``' default unless ``--policy-config``
says otherwise, so a number from one benchmark can be put beside a number from
another and the method behind them is byte for byte the same.

Order of precedence, lowest first: ``RPTSettings`` defaults < ``defaults``
(the entrypoint's: a development protocol lifting ``allow_stair_traversal``,
a benchmark's agent height and radius) < ``--policy-config`` < the explicit
command-line switches (``--explorer``, ``--segmentation``), which must agree
with the policy config when both are given.

Python 3.8 syntax; the heavy imports stay inside :func:`build_policy` so that
importing a benchmark's CLI loads no model client.
"""
from __future__ import annotations

from dataclasses import asdict
import json

from sparx_agency.core.planning.objnav.types.command import NavigationCommand


class StopDiagnostic:
    """Non-privileged plumbing check, never labelled as the search method."""

    name = "diagnostic-stop"

    def reset(self, episode, target):
        pass

    def plan(self, observation):
        return NavigationCommand.stop_here()


def build_policy(args, label_mapper, defaults=None):
    """The policy and its frozen description.

    Args:
        args: Parsed CLI. Read: ``agent``, ``policy_config``, ``explorer``,
            ``seed``, ``detector_url``, ``detector_backend``,
            ``detector_timeout_s`` and, when present, ``segmentation``,
            ``segmenter_url``, ``segmenter_timeout_s``, ``segmenter_model``.
        label_mapper: The benchmark's :class:`TableLabelMapper`; its
            ``vocabulary()`` is what the detector service must have been
            started with.
        defaults: ``RPTSettings`` fields the entrypoint sets below the policy
            config (see the module docstring).

    Returns:
        ``(policy, info)``: the policy and the configuration record the run
        manifest carries -- the policy's own ``configuration()`` plus the LLM
        and detector (and segmenter) identities as the services reported them.

    Raises:
        ValueError: A policy config that is not a JSON object, or a switch that
            disagrees with it.
        RuntimeError: A service that does not answer, or a detector whose
            emission threshold would hide door candidates.
    """
    if args.agent == "stop":
        return StopDiagnostic(), {"method": "diagnostic-stop", "publishable": False}
    from sparx_agency.core.mapping.topology.llm_client import LLMClient, LLMConfig
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception import HttpDetector
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_policy import RPTSearchPolicy, RPTSettings
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.services import VerifiedLLMClient, public_service_url
    overrides = json.loads(args.policy_config.read_text()) if args.policy_config else {}
    if not isinstance(overrides, dict):
        raise ValueError("--policy-config must hold a JSON object")
    if args.explorer is not None:
        if overrides.get("local_exploration", args.explorer) != args.explorer:
            raise ValueError("--explorer disagrees with --policy-config")
        overrides["local_exploration"] = args.explorer
    if getattr(args, "segmentation", None):
        if overrides.get("segmentation", True) is not True:
            raise ValueError("--segmentation disagrees with --policy-config")
        overrides["segmentation"] = True
    settings = RPTSettings(**dict(defaults or {}, **dict(overrides, seed=args.seed)))
    config = LLMConfig.from_env()
    config.seed = args.seed
    client = VerifiedLLMClient(LLMClient(config))
    detector = HttpDetector(public_service_url(args.detector_url), label_mapper.vocabulary(),
                            expected_backend=args.detector_backend,
                            timeout_s=getattr(args, "detector_timeout_s", 30.0))
    services = [("LLM", client), ("detector", detector)]
    segmenter = None
    if settings.segmentation:
        from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.segmentation import HttpSegmenter
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
    policy = RPTSearchPolicy(detector, client, settings, segmenter=segmenter)
    emitted = float(identities["detector"]["metadata"]["detector_config"]["conf_thresh"])
    if emitted > min(settings.detection_confidence, policy.door_settings.confidence):
        raise RuntimeError("Detector emission threshold hides door candidates; restart with --conf 0.05")
    info = policy.configuration()
    llm = asdict(config)
    llm.pop("api_key", None)
    info.update(llm=llm, llm_identity=identities["LLM"], detector=identities["detector"],
                detector_url=args.detector_url, detector_timeout_s=detector.timeout_s)
    if segmenter is not None:
        info.update(segmenter=identities["segmenter"], segmenter_url=args.segmenter_url, segmenter_timeout_s=segmenter.timeout_s)
    return policy, info
