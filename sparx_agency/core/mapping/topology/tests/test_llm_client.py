"""Tests for the topology LLM client — HTTP layer fully mocked."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest
import requests

from sparx_agency.core.mapping.topology import llm_client as llm_client_mod
from sparx_agency.core.mapping.topology.llm_client import (
    LLMClient,
    LLMConfig,
    _best_effort_json,
)


class FakeResponse:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class FakeSession:
    """Records post/get calls; replies from a scripted queue."""

    def __init__(self, replies: Optional[List[Any]] = None):
        self.replies = list(replies or [])
        self.posts: List[Dict[str, Any]] = []
        self.gets: List[Dict[str, Any]] = []

    def post(self, url, data=None, headers=None, timeout=None):
        self.posts.append({"url": url, "payload": json.loads(data),
                           "headers": dict(headers or {}),
                           "timeout": timeout})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def get(self, url, headers=None, timeout=None):
        self.gets.append({"url": url, "headers": dict(headers or {}),
                          "timeout": timeout})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _ollama_reply(content: str) -> FakeResponse:
    return FakeResponse({"message": {"role": "assistant",
                                     "content": content}})


def _openai_reply(content: str) -> FakeResponse:
    return FakeResponse(
        {"choices": [{"message": {"role": "assistant",
                                  "content": content}}]})


def _client(backend: str, replies: List[Any], **cfg_kw) -> LLMClient:
    client = LLMClient(LLMConfig(backend=backend, **cfg_kw))
    client.sess = FakeSession(replies)
    return client


# ---------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------
def test_from_env_defaults(monkeypatch):
    for var in ("LLM_BACKEND", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY",
                "LLM_TEMPERATURE", "LLM_TIMEOUT_S"):
        monkeypatch.delenv(var, raising=False)
    cfg = LLMConfig.from_env()
    assert cfg.backend == "ollama"
    assert cfg.base_url == "http://localhost:11434"
    assert cfg.model == "qwen2.5:3b-instruct"
    assert cfg.api_key == ""
    assert cfg.temperature == pytest.approx(0.2)
    assert cfg.timeout_s == pytest.approx(30.0)


def test_from_env_overrides_and_normalization(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", " OpenAI ")
    monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com/v1/")
    monkeypatch.setenv("LLM_MODEL", "gpt-x")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
    monkeypatch.setenv("LLM_TIMEOUT_S", "5")
    cfg = LLMConfig.from_env()
    assert cfg.backend == "openai"                        # stripped+lowered
    assert cfg.base_url == "https://api.example.com/v1"   # trailing / gone
    assert cfg.api_key == "sk-test"
    assert cfg.temperature == pytest.approx(0.7)
    assert cfg.timeout_s == pytest.approx(5.0)


# ---------------------------------------------------------------------
#  Backend request shapes
# ---------------------------------------------------------------------
def test_ollama_request_shape_forces_json():
    client = _client("ollama", [_ollama_reply('{"a": 1}')],
                     base_url="http://localhost:11434", model="m")
    reply = client.chat_json("SYS", "USR")
    assert reply == {"a": 1}
    post = client.sess.posts[0]
    assert post["url"] == "http://localhost:11434/api/chat"
    assert post["payload"]["format"] == "json"        # forced-JSON mode
    assert post["payload"]["stream"] is False
    assert post["payload"]["model"] == "m"
    assert post["payload"]["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
    ]
    assert post["payload"]["options"]["temperature"] == pytest.approx(0.2)


def test_openai_request_shape_forces_json_and_auth():
    client = _client("openai", [_openai_reply('{"b": 2}')],
                     base_url="https://api.example.com/v1", model="m",
                     api_key="sk-test")
    reply = client.chat_json("SYS", "USR")
    assert reply == {"b": 2}
    post = client.sess.posts[0]
    assert post["url"] == "https://api.example.com/v1/chat/completions"
    assert post["payload"]["response_format"] == {"type": "json_object"}
    assert post["headers"]["Authorization"] == "Bearer sk-test"
    assert post["payload"]["messages"][0]["role"] == "system"


def test_openai_compat_alias_and_no_auth_header_without_key():
    client = _client("openai-compat", [_openai_reply('{"c": 3}')],
                     base_url="http://localhost:8000/v1")
    assert client.chat_json("s", "u") == {"c": 3}
    assert "Authorization" not in client.sess.posts[0]["headers"]


def test_unknown_backend_raises():
    client = _client("granite", [])
    with pytest.raises(ValueError, match="Unknown LLM_BACKEND"):
        client.chat_text("s", "u")


def test_temperature_override_per_call():
    client = _client("ollama", [_ollama_reply('{}')])
    client.chat_json("s", "u", temperature=0.9)
    payload = client.sess.posts[0]["payload"]
    assert payload["options"]["temperature"] == pytest.approx(0.9)


def test_unexpected_reply_shape_raises_runtime_error():
    client = _client("ollama", [FakeResponse({"weird": True})])
    with pytest.raises(RuntimeError, match="unexpected Ollama reply"):
        client.chat_text("s", "u")


# ---------------------------------------------------------------------
#  Retry
# ---------------------------------------------------------------------
def test_retry_once_then_success(monkeypatch):
    naps: List[float] = []
    monkeypatch.setattr(llm_client_mod.time, "sleep", naps.append)
    client = _client("ollama", [requests.ConnectionError("boom"),
                                _ollama_reply('{"ok": true}')])
    assert client.chat_json("s", "u") == {"ok": True}
    assert len(client.sess.posts) == 2
    assert naps == [0.4]                       # one backoff between tries


def test_retry_exhausted_raises(monkeypatch):
    monkeypatch.setattr(llm_client_mod.time, "sleep", lambda _t: None)
    client = _client("ollama", [requests.ConnectionError("a"),
                                requests.Timeout("b")])
    with pytest.raises(RuntimeError, match="failed after 2 tries"):
        client.chat_text("s", "u")
    assert len(client.sess.posts) == 2


def test_http_error_status_is_retried(monkeypatch):
    monkeypatch.setattr(llm_client_mod.time, "sleep", lambda _t: None)
    client = _client("ollama", [FakeResponse({}, status_code=500),
                                _ollama_reply('{"ok": 1}')])
    assert client.chat_json("s", "u") == {"ok": 1}


# ---------------------------------------------------------------------
#  JSON rescue
# ---------------------------------------------------------------------
def test_best_effort_json_direct():
    assert _best_effort_json('{"x": 1}') == {"x": 1}


def test_best_effort_json_fenced():
    text = '```json\n{"x": 1}\n```'
    assert _best_effort_json(text) == {"x": 1}


def test_best_effort_json_prose_prefix():
    text = 'Sure! Here is the JSON:\n{"label": "kitchen"}'
    assert _best_effort_json(text) == {"label": "kitchen"}


def test_best_effort_json_garbage_raises_with_raw_text():
    with pytest.raises(ValueError, match="not json at all"):
        _best_effort_json("not json at all")


def test_chat_json_rescues_fenced_reply():
    client = _client("ollama", [_ollama_reply('```\n{"y": 2}\n```')])
    assert client.chat_json("s", "u") == {"y": 2}


# ---------------------------------------------------------------------
#  Ping
# ---------------------------------------------------------------------
def test_ping_ollama_hits_api_tags():
    client = _client("ollama", [FakeResponse({}, status_code=200)],
                     base_url="http://localhost:11434")
    assert client.ping() is True
    assert client.sess.gets[0]["url"] == "http://localhost:11434/api/tags"


def test_ping_openai_treats_401_as_reachable():
    client = _client("openai", [FakeResponse({}, status_code=401)],
                     base_url="https://api.example.com/v1")
    assert client.ping() is True
    assert client.sess.gets[0]["url"].endswith("/models")


def test_ping_never_raises_on_network_error():
    client = _client("ollama", [requests.ConnectionError("down")])
    assert client.ping() is False
