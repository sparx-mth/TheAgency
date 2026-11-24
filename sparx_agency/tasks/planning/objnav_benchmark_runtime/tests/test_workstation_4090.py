"""Offline launch previews and synthetic workstation budgets; never invoke a benchmark or model server."""
import json
import os
import shlex
import subprocess

import pytest

from sparx_agency.tasks.planning.objnav_benchmark_runtime import workstation_4090 as ws


@pytest.fixture(autouse=True)
def clean_budgets(monkeypatch):
    for key in ("LLM_VRAM_MIB", "YOLO_VRAM_MIB", "GPU_RESERVE_MIB", "HABITAT_VRAM_MIB", "DETECTOR_PORT", "GPU_DEVICE"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("dataset,mode,count,jobs", [("hm3d", [], 2000, 7), ("mp3d", ["--dry-run"], 2195, 4)])
def test_entrypoint_previews_full_split_without_installation(tmp_path, dataset, mode, count, jobs):
    launcher = ws.RUNTIME / dataset / "run_4090.sh"
    output = tmp_path / "runs with spaces"
    result = subprocess.run([str(launcher), *mode, "--output", str(output)], cwd=tmp_path,
                            env={"PATH": os.environ["PATH"], "HOME": str(tmp_path), "SPLIT": "train"},
                            capture_output=True, text=True, check=True)
    assert "Dry run only" in result.stdout and f"{count} episodes" in result.stdout
    tokens = shlex.split(next(line for line in result.stdout.splitlines() if line.startswith("env ")))
    runner = tokens.index(str(launcher.with_name("run_benchmark.sh")))
    env = dict(token.split("=", 1) for token in tokens[1:runner])
    assert env["SPLIT"] == "val" and env["HABITAT_RENDERERS"] == str(jobs)
    assert env["OLLAMA_NUM_PARALLEL"] == "2" and env["LLM_NUM_BATCH"] == "256"
    assert env["LLM_BASE_URL"] == "http://127.0.0.1:11450"
    assert env["DETECTOR_PORT"] == ("18092" if dataset == "hm3d" else "18093")
    assert tokens[runner + 1:] == [
        "--version", "v1", "--agent", "rpt", "--perception", "service", "--jobs", str(jobs), "--output", str(output),
    ] + (["--shard-episodes", "20", "--llm-servers", "1"] if dataset == "mp3d" else [])
    assert not output.exists()


@pytest.mark.parametrize("dataset,used_vram,available_ram,expected", [
    ("hm3d", 400, 58, 7), ("mp3d", 400, 58, 4),
    ("hm3d", 2400, 58, 4), ("mp3d", 400, 30, 2),
])
def test_worker_ceiling_accounts_for_services_and_ram(monkeypatch, dataset, used_vram, available_ram, expected):
    monkeypatch.setattr(ws, "gpu_sample", lambda _: (0, used_vram, 24564))
    monkeypatch.setattr(ws, "ram_sample", lambda: ((64 - available_ram) * 1024, 64 * 1024))
    monkeypatch.setattr(ws.os, "sched_getaffinity", lambda _: set(range(24)))
    profile = ws.PROFILES[dataset]
    assert ws.workstation_jobs(profile, profile.workers, ws.profile_environment(profile, 2, 256, 11450)) == expected


@pytest.mark.parametrize("gpu,available_ram", [((0, 0, 8192), 60), ((0, 16000, 24564), 60), ((0, 0, 24564), 16)])
def test_no_worker_starts_on_laptop_or_when_one_would_not_fit(monkeypatch, gpu, available_ram):
    monkeypatch.setattr(ws, "gpu_sample", lambda _: gpu)
    monkeypatch.setattr(ws, "ram_sample", lambda: ((64 - available_ram) * 1024, 64 * 1024))
    profile = ws.PROFILES["hm3d"]
    with pytest.raises(ValueError, match="required|not enough"):
        ws.workstation_jobs(profile, 1, ws.profile_environment(profile, 2, 256, 11450))


@pytest.mark.parametrize("slots,batch", [(3, 256), (2, 512)])
def test_larger_server_needs_explicit_budget(monkeypatch, slots, batch):
    profile = ws.PROFILES["hm3d"]
    with pytest.raises(ValueError, match="measured LLM_VRAM_MIB"):
        ws.profile_environment(profile, slots, batch, 11450)
    monkeypatch.setenv("LLM_VRAM_MIB", "16000")
    settings = ws.profile_environment(profile, slots, batch, 11450)
    assert (settings["OLLAMA_NUM_PARALLEL"], settings["LLM_NUM_BATCH"]) == (str(slots), str(batch))


def test_resume_keeps_actual_concurrency_but_does_not_bypass_headroom(tmp_path):
    settings = ws.profile_environment(ws.PROFILES["mp3d"], 2, 256, 11450)
    assert ws.campaign_jobs(tmp_path, "mp3d", 3, settings) == 3
    assert json.loads((tmp_path / "workstation_4090.json").read_text())["jobs"] == 3
    assert ws.campaign_jobs(tmp_path, "mp3d", 4, settings) == 3
    with pytest.raises(ValueError, match="resume needs 3"):
        ws.campaign_jobs(tmp_path, "mp3d", 2, settings)
    with pytest.raises(ValueError, match="different settings"):
        ws.campaign_jobs(tmp_path, "mp3d", 3, dict(settings, OLLAMA_NUM_PARALLEL="1"))
