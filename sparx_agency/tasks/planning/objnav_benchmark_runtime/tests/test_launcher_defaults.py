"""The shared launcher's environment defaults (``benchmark_services.sh``): which model answers which role."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

SERVICES = Path(__file__).resolve().parents[1] / "benchmark_services.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="the launcher is a bash script")


def defaults(**environment):
    """``{name: value}`` of the LLM variables after ``objnav_services_defaults`` in an otherwise empty environment."""
    names = ("LLM_MODEL", "LLM_BACKEND", "LLM_NUM_CTX", "LLM_MAX_TOKENS", "LLM_NUM_BATCH",
             "OLLAMA_NUM_PARALLEL", "OLLAMA_MAX_LOADED_MODELS", "OLLAMA_KV_CACHE_TYPE", "OLLAMA_FLASH_ATTENTION")
    script = 'source "$1" && objnav_services_defaults && objnav_model_defaults && printf "%s\\n" ' + " ".join('"$%s"' % name for name in names)
    env = dict({"HOME": os.environ.get("HOME", "/tmp"), "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "AGENT": "rpt"}, **environment)
    out = subprocess.run(["bash", "-c", script, "_", str(SERVICES)], env=env, capture_output=True, text=True, check=True)
    return dict(zip(names, out.stdout.splitlines()))


def test_one_warm_quantized_visual_model_with_bounded_prefill_and_cache():
    assert defaults() == {"LLM_MODEL": "qwen2.5vl:7b", "LLM_BACKEND": "ollama", "LLM_NUM_CTX": "4096",
                          "LLM_MAX_TOKENS": "32", "LLM_NUM_BATCH": "256", "OLLAMA_NUM_PARALLEL": "1",
                          "OLLAMA_MAX_LOADED_MODELS": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0", "OLLAMA_FLASH_ATTENTION": "1"}


def test_a_model_named_in_the_environment_is_kept():
    chosen = defaults(LLM_MODEL="local-vision", OLLAMA_NUM_PARALLEL="2")
    assert chosen["LLM_MODEL"] == "local-vision" and chosen["OLLAMA_NUM_PARALLEL"] == "2"


def test_room_launcher_keeps_the_historical_text_model_reply_and_window_budgets():
    room = defaults(AGENT="rpt_room_llm")
    assert room["LLM_MODEL"] == "qwen2.5:3b-instruct" and room["LLM_MAX_TOKENS"] == "768"
    assert room["OLLAMA_MAX_LOADED_MODELS"] == "2" and room["OLLAMA_KV_CACHE_TYPE"] != "q8_0"


def test_the_python_entrypoints_and_shell_ask_the_same_single_model_with_the_same_budget(monkeypatch):
    from sparx_agency.core.mapping.topology.llm_client import LLMConfig
    for name in ("LLM_MODEL", "LLM_BACKEND", "LLM_NUM_CTX", "LLM_MAX_TOKENS", "LLM_NUM_BATCH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_REASONING_MODEL", "unused-room-model")
    cfg = LLMConfig.from_env(vision=True)
    shell = defaults()
    assert cfg.models() == (shell["LLM_MODEL"],)
    assert (cfg.num_ctx, cfg.max_tokens, cfg.num_batch) == tuple(
        int(shell[k]) for k in ("LLM_NUM_CTX", "LLM_MAX_TOKENS", "LLM_NUM_BATCH"))
