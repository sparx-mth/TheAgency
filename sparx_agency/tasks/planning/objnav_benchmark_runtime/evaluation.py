"""Shared prepared-environment execution; dataset loading and scoring stay in adapters."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Optional

from sparx_agency.core.planning.objnav.action_converter.params import ActionConverterParams
from sparx_agency.core.planning.objnav.agent.headless_agent import HeadlessObjNavAgent
from sparx_agency.core.planning.objnav.agent.params import HeadlessAgentParams
from sparx_agency.tasks.planning.objnav_benchmark.kinematics import KinematicTolerance
from sparx_agency.tasks.planning.objnav_benchmark.logger import MetricsLogger
from sparx_agency.tasks.planning.objnav_benchmark.results_io import strict_json
from sparx_agency.tasks.planning.objnav_benchmark.runner import run_benchmark
from sparx_agency.tasks.planning.objnav_benchmark_runtime.provenance import (
    check_frozen_configuration, select_episodes, source_fingerprint,
)


@dataclass(frozen=True)
class EvaluationSettings:
    """Conservative harness defaults; each adapter declares any protocol differences."""

    require_stop_for_success: bool = True
    path_length_dimension: str = "3d"
    path_length_epsilon_m: float = 0.0
    path_tolerance_m: Optional[float] = 1e-3
    kinematics: Optional[KinematicTolerance] = field(default_factory=KinematicTolerance)
    on_agent_error: str = "record"


def evaluation_configuration(config, episode_ids, settings=None, *, policy):
    """Bind supplied model/data/protocol metadata to actual source, selection and options."""
    options = settings or EvaluationSettings()
    describe = getattr(policy, "configuration", None)
    converter = getattr(policy, "converter_params", ActionConverterParams())
    actual = {"selected_episode_ids": list(select_episodes(episode_ids)),
              "evaluation": asdict(options), "source_sha256": source_fingerprint(),
              "policy": describe() if describe is not None else {"method": policy.name},
              "agent_converter": asdict(converter)}
    # Compare the serialized contract: tuples and JSON lists represent the same
    # configuration after a saved lock/run manifest is read back for resumption.
    result = json.loads(strict_json(dict(config), "evaluation configuration"))
    actual = json.loads(strict_json(actual, "prepared evaluation configuration"))
    for key, value in actual.items():
        if key in result and result[key] != value:
            raise ValueError("Prepared configuration changed: " + key)
        result[key] = value
    return result


def run_evaluation(env, policy, label_mapper, *, output, config, settings=None,
                   episode_ids=None, resume=False, frozen_lock=None,
                   expected_episode_ids=None, record=False, video_fps=6,
                   writer_factory=None, dashboard=True, progress=None):
    """Run the exact prepared policy, optionally locked/recorded, and close the env.

    Adapters own data loading, published episode identities, model provisioning,
    camera/body/action specifications, privileged measurements and preflight.
    No simulator or model is launched/downloaded here. A frozen lock is checked
    against THIS execution configuration before output creation or environment
    reset, never only against an earlier preflight.

    Extra evaluator telemetry is recorder-only and optional. The policy sees
    only ObjNavObservation. Recording preserves the converter settings used by
    route commitment; wrappers do not silently construct a different controller.

    An optional ``progress(index, total, row)`` is called after each scored
    episode, for an adapter's console line. It receives the recorded row the
    logger wrote, never evaluator telemetry the policy may not see.
    """
    options = settings or EvaluationSettings()
    recorder = None
    active_env = env
    try:
        ids = select_episodes(env.episode_ids() if episode_ids is None else episode_ids)
        if expected_episode_ids is not None and ids != select_episodes(expected_episode_ids):
            raise ValueError("Selection differs from the required published episode identities")
        prepared = evaluation_configuration(config, ids, options, policy=policy)
        if frozen_lock is not None:
            check_frozen_configuration(prepared, frozen_lock,
                                       expected_episode_ids=expected_episode_ids)
        converter = getattr(policy, "converter_params", ActionConverterParams())
        params = HeadlessAgentParams(converter=converter)
        output = Path(output).expanduser()
        # MetricsLogger claims the output/resume lock before any video is written.
        with MetricsLogger(output, prepared, resume=resume) as logger:
            if record:
                from sparx_agency.tasks.planning.objnav_benchmark_runtime.dashboard import write_live_page
                from sparx_agency.tasks.planning.objnav_benchmark_runtime.recording import (
                    EpisodeRecorder, PolicyProbe, RecordingAgent, RecordingEnv, VideoSink,
                )
                write_live_page(output)
                recorder = EpisodeRecorder(output, policy, fps=video_fps,
                                           writer_factory=writer_factory or VideoSink)
                probe = PolicyProbe(policy)
                agent = RecordingAgent(HeadlessObjNavAgent(probe, label_mapper, params, name=policy.name),
                                       probe, recorder)
                active_env = RecordingEnv(env, recorder)
            else:
                agent = HeadlessObjNavAgent(policy, label_mapper, params, name=policy.name)

            def on_episode(index, total, row):
                if recorder is not None:
                    recorder.complete(row)
                if progress is not None:
                    progress(index, total, row)

            summary = run_benchmark(
                active_env, agent, logger=logger, episode_ids=ids, progress=on_episode,
                require_stop_for_success=options.require_stop_for_success,
                path_length_dimension=options.path_length_dimension,
                path_length_epsilon_m=options.path_length_epsilon_m,
                path_tolerance_m=options.path_tolerance_m, kinematics=options.kinematics,
                on_agent_error=options.on_agent_error)
        if record and dashboard:
            from sparx_agency.tasks.planning.objnav_benchmark_runtime.dashboard import write_dashboard
            write_dashboard(output)
        return summary
    finally:
        try:
            if recorder is not None:
                recorder.close()
        finally:
            active_env.close()

