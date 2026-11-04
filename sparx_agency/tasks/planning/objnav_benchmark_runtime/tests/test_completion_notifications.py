"""Operator notifications use completed recordings, not guessed artifact paths."""
import json
from types import SimpleNamespace

import pytest

from sparx_agency.core.planning.objnav.agent.params import HeadlessAgentParams
from sparx_agency.tasks.planning.objnav_benchmark_runtime.gibson import run_development as run


@pytest.mark.parametrize("recording", [False, True])
def test_completion_metrics_precede_pause(tmp_path, monkeypatch, capsys, recording):
    from sparx_agency.tasks.planning.objnav_benchmark_runtime import recording as rec, dashboard

    class Logger:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Recorder:
        def __init__(self, *args, **kwargs):
            self.episode_dir = tmp_path / "recordings" / "fixture"

        def complete(self, row):
            self.episode_dir.mkdir(parents=True)
            (self.episode_dir / "video.mp4").write_bytes(b"fixture")

        def close(self):
            pass

    def benchmark(*args, progress, **kwargs):
        progress(1, 1, SimpleNamespace(episode_id="scene/0", success=True, spl=0.5,
                                      distance_to_goal_m=0.2, wall_s=10.0, steps=20))
        return "summary"

    def pause(seconds):
        output = capsys.readouterr().out
        assert "EPISODE COMPLETE" in output
        assert "SR=1 SPL=0.5000 DTG=0.2 m Runtime=10.00 s Average FPS=2.000" in output
        assert str(tmp_path / "recordings" / "fixture" / "video.mp4") in output if recording else "not recorded" in output
        assert seconds == 60

    monkeypatch.setattr(run, "MetricsLogger", Logger)
    monkeypatch.setattr(run, "run_benchmark", benchmark)
    monkeypatch.setattr(run.time, "sleep", pause)
    monkeypatch.setattr(rec, "EpisodeRecorder", Recorder)
    monkeypatch.setattr(dashboard, "write_live_page", lambda *args: None)
    monkeypatch.setattr(dashboard, "write_dashboard", lambda *args: None)
    env = SimpleNamespace(evaluation_diagnostics=lambda: {}, close=lambda: None,
                          protocol=SimpleNamespace(path_length_dimension="3d", path_length_epsilon_m=1e-5,
                                                   kinematics=lambda: None))
    policy = SimpleNamespace(name="fixture", converter_params=HeadlessAgentParams().converter,
                             reset=lambda *args: None, plan=lambda *args: None)
    args = SimpleNamespace(expect_config=None, output=tmp_path, record=recording, video_fps=6,
                           resume=False, inspection_pause=60)
    config = {"selected_episode_ids": ["scene/0"], "recording": {"episode_ids": ["scene/0"]}}
    assert run.execute(args, env, policy, config) == "summary"
    result = json.loads((tmp_path / "performance_summary.json").read_text())
    assert result["average_fps"] == 2
    assert result["total_runtime_s"] == 10


@pytest.mark.parametrize("pause", ["-1", "nan", "inf"])
def test_invalid_inspection_pause_fails_before_preflight(pause):
    with pytest.raises(ValueError, match="Inspection pause"):
        run.main(["--manifest", "unused.json", "--scene", "scene", "--output", "unused",
                  "--explorer", "frontier", "--detector-url", "http://localhost:1", "--inspection-pause", pause])

