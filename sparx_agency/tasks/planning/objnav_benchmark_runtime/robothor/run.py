"""RoboTHOR preflight and evaluation CLI; simulator/model contexts are lazy.

Unlike the Gibson adapter -- which predates the shared runtime -- this drives
the extracted :func:`~...evaluation.run_evaluation`, so resumption,
configuration locks, provenance, recording and the dashboard are the shared
ones and are not reimplemented here. What is local is exactly what is
RoboTHOR: the episode files, the challenge profile, the LoCoBot embodiment,
and the build/platform choice.

Nothing is downloaded and no simulator or service is started. ``--preflight``
checks the dataset, the services, the GPU and the build without rendering a
frame.

Python 3.8 syntax.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

from sparx_agency.core.planning.objnav.labels.datasets.robothor import (
    robothor_label_mapper,
)
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.tasks.planning.objnav_benchmark.results_io import default_run_dir
from sparx_agency.tasks.planning.objnav_benchmark_runtime.evaluation import (
    EvaluationSettings, run_evaluation,
)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.provenance import (
    select_episodes, source_fingerprint,
)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.dataset import (
    RobothorDataset,
)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.env import (
    SUCCESS_ANY_INSTANCE, SUCCESS_RULES, RobothorEnv,
)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.protocol import (
    PROTOCOL, SCENES, VAL_EPISODES_PER_SCENE,
)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.thor.simulator import (
    AI2ThorRGBDSimulator,
)

#: How the Unity build is reached on Linux, and what each costs.
#:
#: ``reference-linux64`` runs the build the challenge pins. That build has
#: **no CloudRendering variant**, so it needs an X display, and on a
#: PRIME-offload laptop it reaches the discrete GPU only with the two
#: ``__NV_*`` offload variables set -- without them it renders on the
#: integrated GPU and nothing says so.
#:
#: ``cloud-rendering`` is headless Vulkan and reaches the discrete GPU cleanly,
#: but only exists for builds from ai2thor 3.5 onward, i.e. **not** the
#: reference build. Choosing it is choosing a different simulator than every
#: published RoboTHOR number was measured on, which is why it has to be named.
PLATFORMS = ("reference-linux64", "cloud-rendering")


class StopDiagnostic:
    """Non-privileged plumbing check, never labelled as the search method."""

    name = "diagnostic-stop"

    def reset(self, episode, target):
        pass

    def plan(self, observation):
        return NavigationCommand.stop_here()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes-dir", default=os.environ.get("ROBOTHOR_EPISODES_DIR"),
                   help="The robothor-challenge dataset root, or its val/ dir")
    p.add_argument("--output", type=Path)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--print-vocabulary", action="store_true")
    p.add_argument("--agent", choices=("rpt", "stop"), default="rpt")
    p.add_argument("--policy-config", type=Path)
    p.add_argument("--detector-url", default="http://127.0.0.1:18092")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--scene", choices=SCENES)
    p.add_argument("--record", action="store_true")
    p.add_argument("--video-fps", type=int, default=6)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--platform", choices=PLATFORMS, default="reference-linux64")
    p.add_argument("--thor-build", default=PROTOCOL.thor_build_id,
                   help="Unity build commit; defaults to the challenge's")
    p.add_argument("--success-rule", choices=SUCCESS_RULES,
                   default=SUCCESS_ANY_INSTANCE)
    p.add_argument("--geodesic-telemetry", action="store_true",
                   help="Ask the navmesh for a distance every step (slow); "
                        "only for a recorded run's distance-to-goal curve")
    p.add_argument("--allow-build-mismatch", action="store_true")
    p.add_argument("--allow-shared-gpu", action="store_true")
    p.add_argument("--frozen-lock", type=Path)
    return p


def thor_platform(name):
    """The ``ai2thor.platform`` class for ``name``, imported lazily."""
    if name == "cloud-rendering":
        from ai2thor.platform import CloudRendering
        return CloudRendering
    from ai2thor.platform import Linux64
    return Linux64


def _runtime(issues):
    """Installed package versions, and what is missing, for the manifest."""
    packages = {}
    for module, distribution in (("numpy", "numpy"), ("scipy", "scipy"),
                                 ("skimage", "scikit-image"),
                                 ("skfmm", "scikit-fmm"), ("cv2", "opencv-python"),
                                 ("networkx", "networkx"), ("requests", "requests"),
                                 ("ai2thor", "ai2thor")):
        try:
            loaded = importlib.import_module(module)
            try:
                packages[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                packages[distribution] = getattr(loaded, "__version__", "unknown")
        except (ImportError, OSError) as exc:
            issues.append("%s unavailable: %s" % (module, exc))
    return packages


def _method(args):
    """The policy and its identity, with every service checked before running."""
    if args.agent == "stop":
        return StopDiagnostic(), {"method": "diagnostic-stop", "publishable": False}
    from sparx_agency.core.mapping.topology.llm_client import LLMClient, LLMConfig
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.perception import (
        HttpDetector)
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_policy import (
        RPTSearchPolicy, RPTSettings)
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.services import (
        VerifiedLLMClient, public_service_url)

    overrides = json.loads(args.policy_config.read_text()) if args.policy_config else {}
    if not isinstance(overrides, dict):
        raise ValueError("--policy-config must hold a JSON object")
    settings = RPTSettings(**dict(embodiment(), **dict(overrides, seed=args.seed)))
    config = LLMConfig.from_env()
    config.seed = args.seed
    client = VerifiedLLMClient(LLMClient(config))
    mapper = robothor_label_mapper()
    detector = HttpDetector(public_service_url(args.detector_url), mapper.vocabulary())
    problems, identities = [], {}
    for name, service in (("LLM", client), ("detector", detector)):
        try:
            identities[name] = service.health()
        except Exception as exc:
            problems.append("%s: %s" % (name, exc))
    if problems:
        raise RuntimeError("; ".join(problems))
    policy = RPTSearchPolicy(detector, client, settings)
    emitted = float(identities["detector"]["metadata"]["detector_config"]["conf_thresh"])
    if emitted > min(settings.detection_confidence, policy.door_settings.confidence):
        raise RuntimeError(
            "Detector emission threshold hides door candidates; restart it "
            "with --conf 0.05")
    info = policy.configuration()
    llm = asdict(config)
    llm.pop("api_key", None)
    info.update(llm=llm, llm_identity=identities["LLM"],
                detector=identities["detector"], detector_url=args.detector_url)
    return policy, info


def embodiment():
    """The LoCoBot's own geometry, and the standoff the method aims for.

    Not inherited from the Gibson profile. Gibson's ``body_height_m`` of 0.88
    is numerically its *camera* height, which the shared runbook warns against;
    the LoCoBot's collider is 0.9 m tall and 0.175 m in radius, read from the
    pinned build's own ``BaseFPSAgentController`` bot branch.

    ``stop_distance_m`` is a method choice and never an override of the
    evaluator: RoboTHOR's success needs the goal within ``visibilityDistance``
    (1.0 m) *and* unoccluded in frame, so stopping at 0.75 m leaves a quarter
    of a metre of margin while staying inside the rule.

    ``map_size_m`` is 30 rather than the 80 m default: RoboTHOR apartments are
    about 10 m across, and an 80 m grid at 0.1 m is a 800x800 watershed
    re-segmented every ten steps for no benefit.
    """
    return {"body_height_m": PROTOCOL.body_height_m,
            "body_radius_m": PROTOCOL.agent_radius_m,
            "preferred_clearance_m": 0.25,
            "stop_distance_m": 0.75,
            "map_size_m": 30.0}


def _gpu_gate(args):
    """Refuse to render on a GPU another process already owns."""
    if args.allow_shared_gpu:
        return
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True)
    used = max(int(line) for line in result.stdout.split() if line.strip())
    if used > 512:
        raise RuntimeError(
            "Rendering GPU is occupied (%d MiB); keep the detector on CPU, or "
            "pass --allow-shared-gpu deliberately" % used)


def _platform_advice(args, issues):
    """Name the two silent ways a RoboTHOR run renders on the wrong device."""
    if args.platform == "reference-linux64":
        if not os.environ.get("DISPLAY"):
            issues.append(
                "reference-linux64 needs an X display; set DISPLAY, or choose "
                "--platform cloud-rendering (a different build -- see README)")
        offload = (os.environ.get("__NV_PRIME_RENDER_OFFLOAD"),
                   os.environ.get("__GLX_VENDOR_LIBRARY_NAME"))
        if offload != ("1", "nvidia"):
            issues.append(
                "On a PRIME-offload machine plain GLX renders on the "
                "integrated GPU and says nothing. Export "
                "__NV_PRIME_RENDER_OFFLOAD=1 and __GLX_VENDOR_LIBRARY_NAME=nvidia")
    elif not os.environ.get("VK_DRIVER_FILES") and not os.environ.get(
            "VK_ICD_FILENAMES"):
        issues.append(
            "cloud-rendering picks a Vulkan device by index and may choose the "
            "integrated GPU. Export "
            "VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json")


def prepare(args):
    """Verify the exact requested selection, services, build and platform."""
    issues = []
    if args.seed < 0:
        issues.append("--seed must be non-negative")
    if args.agent == "stop" and args.limit is None:
        issues.append("The stop diagnostic requires an explicit --limit")
    if args.scene and (args.limit is None
                       or not 1 <= args.limit <= VAL_EPISODES_PER_SCENE):
        issues.append("A single-scene run requires --limit between 1 and %d"
                      % VAL_EPISODES_PER_SCENE)
    if not 1 <= args.video_fps <= 60:
        issues.append("--video-fps must be between 1 and 60")
    if args.record:
        try:
            from sparx_agency.tasks.planning.objnav_benchmark_runtime.recording import (
                ffmpeg_executable)
            ffmpeg_executable()
        except Exception as exc:
            issues.append("Recording: %s" % exc)
    try:
        _gpu_gate(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        issues.append("Rendering GPU: %s" % exc)
    _platform_advice(args, issues)
    runtime = _runtime(issues)
    build_match = args.thor_build == PROTOCOL.thor_build_id
    if not build_match and not args.allow_build_mismatch:
        issues.append(
            "The challenge and every published RoboTHOR number use Unity build "
            "%s; this run asks for %s. Pass --allow-build-mismatch only for an "
            "acknowledged port." % (PROTOCOL.thor_build_id, args.thor_build))
    if args.platform == "cloud-rendering" and build_match:
        issues.append(
            "The reference build %s has no CloudRendering variant. Either use "
            "--platform reference-linux64, or choose a newer --thor-build and "
            "accept --allow-build-mismatch." % PROTOCOL.thor_build_id)

    dataset = env = policy = None
    ids, method = (), {}
    if not args.episodes_dir:
        issues.append("Set --episodes-dir (or ROBOTHOR_EPISODES_DIR) to the "
                      "robothor-challenge dataset directory")
    else:
        try:
            dataset = RobothorDataset(
                args.episodes_dir, split=PROTOCOL.split, scene=args.scene,
                require_full_split=args.scene is None)
            env = RobothorEnv(
                dataset, _simulator(args), success_rule=args.success_rule,
                geodesic_telemetry=args.geodesic_telemetry)
            ids = select_episodes(env.episode_ids(), args.limit, args.shards,
                                  args.shard_index)
        except (OSError, ValueError, KeyError, ImportError) as exc:
            issues.append("Dataset/environment: %s" % exc)
    try:
        policy, method = _method(args)
    except Exception as exc:
        issues.append("Method services/configuration: %s" % exc)

    config = {
        "protocol": asdict(PROTOCOL), "runtime": runtime,
        "source_sha256": source_fingerprint(), "method": method,
        "embodiment": embodiment(), "seed": args.seed,
        "thor_build_id": args.thor_build,
        "reference_build_match": build_match,
        "platform": args.platform,
        "success_rule": args.success_rule,
        # Ground-truth pose, as every mapping-based comparison method uses on
        # the Habitat benchmarks. The official RoboTHOR runner hides it; see
        # the README's protocol-deviation section.
        "pose_source": "simulator-ground-truth",
        "selected_episode_ids": list(ids), "shards": args.shards,
        "shard_index": args.shard_index, "limit": args.limit,
        "scene": args.scene,
        "recording": {"enabled": args.record, "fps": args.video_fps},
        "kinematics": asdict(PROTOCOL.kinematics()),
        "full_split": (args.scene is None and args.limit is None
                       and args.shards == 1),
        "dataset": dataset.manifest() if dataset else None,
    }
    return env, policy, config, issues


def _simulator(args):
    """The bridge, with the requested build and platform. Starts nothing yet."""
    camera = PROTOCOL.camera()
    return AI2ThorRGBDSimulator(
        camera, PROTOCOL.actions(), PROTOCOL.agent_radius_m,
        height_m=PROTOCOL.body_height_m,
        origin_height_m=PROTOCOL.origin_height_m,
        initialize=PROTOCOL.initialize(), commit_id=args.thor_build,
        width=PROTOCOL.width, height=PROTOCOL.height,
        platform=thor_platform(args.platform),
        far_plane_sentinel_m=(PROTOCOL.camera_far_plane_m
                              - PROTOCOL.camera_near_plane_m))


def _write_adapter_audit(output, env):
    """Adapter-only telemetry the shared record schema has no field for.

    The visibility-rule tally and the navmesh query counts say whether the two
    judgement calls this adapter had to make -- which instance must be visible,
    and how often the geodesic needed a relaxed tolerance -- actually mattered
    on this run. Written separately so nothing in the shared harness has to
    grow a RoboTHOR-shaped field.
    """
    from sparx_agency.tasks.planning.objnav_benchmark.results_io import (
        strict_json, write_atomically)

    payload = {"success_rule": env.success_rule,
               "visibility_rule_audit": dict(env.visibility_tally),
               "geodesic_queries": env.geodesic_query_counts()}
    write_atomically(Path(output) / "adapter_audit.json",
                     strict_json(payload, "adapter audit", indent=2) + "\n")


def settings():
    """The protocol's own accounting, not the shared conservative defaults."""
    return EvaluationSettings(
        require_stop_for_success=PROTOCOL.require_stop_for_success,
        path_length_dimension=PROTOCOL.path_length_dimension,
        path_length_epsilon_m=PROTOCOL.path_length_epsilon_m,
        kinematics=PROTOCOL.kinematics(), on_agent_error="record")


def main(argv=None):
    args = parser().parse_args(argv)
    if args.print_vocabulary:
        print(",".join(robothor_label_mapper().vocabulary()))
        return 0
    if args.resume and args.output is None:
        parser().error("--resume requires --output")
    env, policy, config, issues = prepare(args)
    try:
        if issues:
            print(json.dumps({"ready": False, "issues": issues}, indent=2))
            return 1
        if args.preflight:
            print(json.dumps({"ready": True, "configuration": config}, indent=2))
            return 0
        _gpu_gate(args)
        output = args.output or default_run_dir(PROTOCOL.benchmark, PROTOCOL.split)
        evaluated = env
        summary = run_evaluation(
            env, policy, robothor_label_mapper(), output=output, config=config,
            settings=settings(), episode_ids=config["selected_episode_ids"],
            resume=args.resume, frozen_lock=args.frozen_lock,
            record=args.record, video_fps=args.video_fps)
        env = None  # run_evaluation closes it, even on a recording failure.
        _write_adapter_audit(output, evaluated)
        from sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.report import (
            write_report)
        write_report(output)
        print("Results: %s (%d episodes)" % (output, summary.overall.n_episodes))
        return 0
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
