"""The RoboTHOR CLI's refusals, and the report it writes.

The preflight exists to fail loudly in the four ways a RoboTHOR sweep fails
quietly: rendering on the wrong GPU, running a build that is not the one the
published numbers were measured on, calling a subset the published split, and
scoring against a paper number from the other split.
"""
from __future__ import annotations

import json

import pytest

from sparx_agency.tasks.planning.objnav_benchmark.comparison import comparison_table
from sparx_agency.tasks.planning.objnav_benchmark.errors import HarnessError
from sparx_agency.tasks.planning.objnav_benchmark.summaries import ReportedResult
from sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor import run as cli
from sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.protocol import (
    PROTOCOL, SCENES, VAL_EPISODES, VAL_EPISODES_PER_SCENE,
)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.reported import (
    METHOD_PROPERTIES, OUR_PROPERTIES, TEST_SPLIT_RESULTS, VALIDATION_RESULTS,
    properties_row, validation_results,
)


def issues(argv, env=None, monkeypatch=None):
    """The preflight's complaints for ``argv``, as a list of strings."""
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value) if value else monkeypatch.delenv(
            key, raising=False)
    args = cli.parser().parse_args(argv)
    return cli.prepare(args)[3]


# --------------------------------------------------------------- the profile


def test_the_published_split_shape_is_stated_once_and_agrees():
    assert len(SCENES) == 15
    assert VAL_EPISODES == len(SCENES) * VAL_EPISODES_PER_SCENE == 1800
    assert SCENES[0] == "FloorPlan_Val1_1" and SCENES[-1] == "FloorPlan_Val3_5"


def test_the_protocol_keeps_the_harness_defaults_that_robothor_shares():
    """Gibson had to override both of these; RoboTHOR must not."""
    options = cli.settings()
    assert options.require_stop_for_success is True
    assert options.path_length_dimension == "3d"
    assert options.path_length_epsilon_m == 0.0


def test_the_embodiment_is_the_locobot_and_not_the_gibson_profile():
    body = cli.embodiment()
    assert body["body_radius_m"] == 0.175          # not Gibson's 0.18
    assert body["body_height_m"] == 0.9            # the collider, not the camera
    assert body["body_height_m"] != PROTOCOL.camera_height_m
    assert body["preferred_clearance_m"] >= body["body_radius_m"]
    # The standoff stays inside the challenge's own visibility distance.
    assert body["stop_distance_m"] < PROTOCOL.visibility_distance_m


def test_the_embodiment_builds_a_valid_settings_object():
    from sparx_agency.tasks.planning.objnav_benchmark_runtime.methods.rpt_policy import (
        RPTSettings)
    settings = RPTSettings(**cli.embodiment())
    assert settings.body_radius_m == PROTOCOL.agent_radius_m


# -------------------------------------------------------------- the refusals


def test_a_newer_build_must_be_asked_for_explicitly(monkeypatch):
    complaints = issues(["--thor-build", "f0825767"], monkeypatch=monkeypatch)
    assert any("published RoboTHOR number" in c for c in complaints)
    allowed = issues(["--thor-build", "f0825767", "--allow-build-mismatch"],
                     monkeypatch=monkeypatch)
    assert not any("published RoboTHOR number" in c for c in allowed)


def test_cloud_rendering_cannot_silently_mean_the_reference_build(monkeypatch):
    complaints = issues(["--platform", "cloud-rendering"], monkeypatch=monkeypatch)
    assert any("no CloudRendering variant" in c for c in complaints)


def test_the_prime_offload_trap_is_named(monkeypatch):
    monkeypatch.delenv("__NV_PRIME_RENDER_OFFLOAD", raising=False)
    monkeypatch.delenv("__GLX_VENDOR_LIBRARY_NAME", raising=False)
    monkeypatch.setenv("DISPLAY", ":1")
    complaints = issues([], monkeypatch=monkeypatch)
    assert any("integrated GPU" in c for c in complaints)

    monkeypatch.setenv("__NV_PRIME_RENDER_OFFLOAD", "1")
    monkeypatch.setenv("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
    quiet = issues([], monkeypatch=monkeypatch)
    assert not any("integrated GPU" in c for c in quiet)


def test_cloud_rendering_warns_about_the_vulkan_device_index(monkeypatch):
    monkeypatch.delenv("VK_DRIVER_FILES", raising=False)
    monkeypatch.delenv("VK_ICD_FILENAMES", raising=False)
    complaints = issues(["--platform", "cloud-rendering"], monkeypatch=monkeypatch)
    assert any("VK_DRIVER_FILES" in c for c in complaints)


def test_a_single_scene_run_needs_an_explicit_limit(monkeypatch):
    complaints = issues(["--scene", SCENES[0]], monkeypatch=monkeypatch)
    assert any("single-scene run requires --limit" in c for c in complaints)


def test_the_stop_diagnostic_needs_an_explicit_limit(monkeypatch):
    complaints = issues(["--agent", "stop"], monkeypatch=monkeypatch)
    assert any("stop diagnostic requires an explicit --limit" in c
               for c in complaints)


def test_a_missing_dataset_is_named_rather_than_crashed(monkeypatch):
    monkeypatch.delenv("ROBOTHOR_EPISODES_DIR", raising=False)
    complaints = issues([], monkeypatch=monkeypatch)
    assert any("--episodes-dir" in c for c in complaints)


def test_a_subset_run_is_never_labelled_the_full_split(monkeypatch):
    args = cli.parser().parse_args(["--limit", "5"])
    assert cli.prepare(args)[2]["full_split"] is False
    args = cli.parser().parse_args(["--shards", "4", "--shard-index", "1"])
    assert cli.prepare(args)[2]["full_split"] is False
    args = cli.parser().parse_args([])
    assert cli.prepare(args)[2]["full_split"] is True


def test_the_configuration_records_what_would_change_a_number(monkeypatch):
    config = cli.prepare(cli.parser().parse_args([]))[2]
    for key in ("protocol", "embodiment", "thor_build_id", "platform",
                "success_rule", "pose_source", "reference_build_match",
                "source_sha256", "kinematics"):
        assert key in config, key
    assert config["protocol"] == __import__(
        "dataclasses").asdict(PROTOCOL)


def test_the_vocabulary_is_printable_without_any_simulator(capsys):
    assert cli.main(["--print-vocabulary"]) == 0
    printed = capsys.readouterr().out.strip().split(",")
    assert "alarm clock" in printed and "spray bottle" in printed
    # The door detector's prompts have to survive into the service vocabulary.
    assert "door frame" in printed and "open doorway" in printed


# ------------------------------------------------------- the reported numbers


def test_every_reported_row_is_a_fraction_with_a_source():
    for result in VALIDATION_RESULTS + TEST_SPLIT_RESULTS:
        assert 0.0 <= result.success_rate <= 1.0
        assert 0.0 <= result.spl <= 1.0
        assert result.source and result.benchmark == "robothor"


def test_the_validation_and_test_rows_are_kept_apart():
    assert all(r.split == "val" for r in VALIDATION_RESULTS)
    assert all(r.split == "test" for r in TEST_SPLIT_RESULTS)
    # The rows the literature mislabels are in the test group, not the val one.
    methods = {r.method for r in TEST_SPLIT_RESULTS}
    assert any("ProcTHOR + fine-tune" in m for m in methods)
    assert not any("ProcTHOR" in r.method for r in VALIDATION_RESULTS)


def test_a_test_split_row_cannot_be_tabled_beside_a_validation_run():
    """The harness refuses it, which is why the split field is load-bearing."""
    summary = _fake_val_summary()
    with pytest.raises(HarnessError):
        comparison_table(summary, list(TEST_SPLIT_RESULTS))
    # ...and accepts the validation ones.
    assert "SG-Nav" in comparison_table(summary, list(VALIDATION_RESULTS))


def test_tf_and_nm_are_labels_not_metrics():
    """OSG Navigator's Table 2 flags: training-free, non-metric."""
    assert properties_row("CogNav") == ("yes", "no")
    assert properties_row("EmbCLIP (trained, 200M steps)") == ("no", "no")
    assert properties_row("nobody has heard of this") == ("?", "?")
    # Ours: training-free, but metric -- we build and plan on an occupancy map.
    assert properties_row("ours", OUR_PROPERTIES) == ("yes", "no")
    assert set(METHOD_PROPERTIES) <= {r.method for r in VALIDATION_RESULTS}


def test_the_trained_reference_row_can_be_dropped():
    zero_shot = validation_results(include_trained=False)
    assert all(METHOD_PROPERTIES[r.method][0] for r in zero_shot)
    assert len(zero_shot) == len(VALIDATION_RESULTS) - 1


def _fake_val_summary():
    """The smallest BenchmarkSummary the comparison table will accept."""
    from sparx_agency.tasks.planning.objnav_benchmark.summaries import (
        BenchmarkSummary, GroupSummary)
    overall = GroupSummary(
        key="all", n_episodes=1, success_rate=0.5, success_rate_ci=(0.1, 0.9),
        spl=0.25, spl_ci=(0.1, 0.4), soft_spl=0.3, distance_to_goal_m=1.0,
        n_unreachable_end=0, mean_steps=42.0, mean_path_length_m=7.5)
    return BenchmarkSummary(
        benchmark="robothor", split="val", agent="sparx-rpt-llm-host-sweep",
        overall=overall, by_category=(), by_scene=(),
        terminations={"stop": 1, "step_limit": 0, "agent_error": 0}, agent_errors=0,
        confidence=0.95)
