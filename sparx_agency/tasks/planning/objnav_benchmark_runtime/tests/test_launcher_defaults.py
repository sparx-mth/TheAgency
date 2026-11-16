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
    names = ("LLM_MODEL", "LLM_REASONING_MODEL", "LLM_BACKEND")
    script = 'source "$1" && objnav_services_defaults && printf "%s\\n" ' + " ".join('"$%s"' % name for name in names)
    env = dict({"HOME": os.environ.get("HOME", "/tmp"), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}, **environment)
    out = subprocess.run(["bash", "-c", script, "_", str(SERVICES)], env=env, capture_output=True, text=True, check=True)
    return dict(zip(names, out.stdout.splitlines()))


def test_the_oracle_defaults_to_the_model_the_development_laptop_can_hold():
    """Since 2026-11-16 the node oracle's default is the 3B, the model the default prompt version is tuned on; the
    14B of the runs published before is one variable away."""
    assert defaults() == {"LLM_MODEL": "qwen2.5:3b-instruct", "LLM_REASONING_MODEL": "qwen2.5:3b-instruct",
                          "LLM_BACKEND": "ollama"}


def test_a_model_named_in_the_environment_is_kept():
    chosen = defaults(LLM_REASONING_MODEL="qwen2.5:14b-instruct", LLM_MODEL="llama3.2:3b")
    assert chosen["LLM_REASONING_MODEL"] == "qwen2.5:14b-instruct" and chosen["LLM_MODEL"] == "llama3.2:3b"
