"""The LLM answer store, the client's record/replay path, and the exclusive flush -- HTTP fully mocked."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from sparx_agency.core.mapping.topology.llm_answer_cache import LLMAnswerStore, request_key
from sparx_agency.core.mapping.topology.llm_client import LLMClient, LLMConfig
from sparx_agency.core.mapping.topology.llm_exclusive import ExclusiveColdStart


class _Reply:
    def __init__(self, content: str):
        self._payload = {"message": {"role": "assistant", "content": content}, "prompt_eval_count": 3,
                         "eval_count": 2, "total_duration": 1000}
        self.status_code = 200

    def json(self) -> Dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        pass


class _Session:
    """Answers every POST with the next scripted content; records the payloads."""

    def __init__(self, contents: List[str]):
        self.contents = list(contents)
        self.payloads: List[Dict[str, Any]] = []

    def post(self, url, data=None, headers=None, timeout=None):
        self.payloads.append(json.loads(data))
        return _Reply(self.contents.pop(0))


def _client(contents, answers=None) -> LLMClient:
    client = LLMClient(LLMConfig(model="m", reasoning_model="m", seed=17), answers=answers)
    client.sess = _Session(contents)
    return client


def test_the_key_moves_with_every_answer_shaping_field_and_nothing_else():
    base = dict(backend="ollama", model="m", system="s", user="u", temperature=0.0, seed=17, num_ctx=8192,
                max_tokens=768)
    key = request_key(**base)
    assert key == request_key(**dict(base))
    for field, other in (("model", "n"), ("system", "s2"), ("user", "u2"), ("temperature", 0.2), ("seed", 18),
                         ("num_ctx", 4096), ("max_tokens", 1536), ("backend", "openai")):
        assert request_key(**dict(base, **{field: other})) != key, field


def test_a_recorded_answer_is_replayed_by_a_later_store_without_asking_the_model(tmp_path):
    first = _client(['{"a": 1}'], LLMAnswerStore(record_dir=tmp_path / "a"))
    assert first.chat_json("sys", "user") == {"a": 1}
    assert first.answer_stats() == {"replayed": 0, "repeated": 0, "asked": 1, "recorded": 1}
    second = _client([], LLMAnswerStore(record_dir=tmp_path / "b", replay_dirs=[tmp_path / "a"]))
    assert second.chat_json("sys", "user") == {"a": 1}, "no scripted reply left: the model was not asked"
    assert second.answer_stats() == {"replayed": 1, "repeated": 0, "asked": 0, "recorded": 1}
    line = json.loads(next((tmp_path / "b").glob("answers-*.jsonl")).read_text())
    assert line["source"] == "replay" and line["user"] == "user" and line["answer"] == '{"a": 1}', (
        "a replaying run's own directory holds every answer it used")


def test_a_question_asked_again_in_a_run_gets_its_first_answer_and_a_new_one_asks_the_model(tmp_path):
    client = _client(['{"a": 1}', '{"b": 2}'], LLMAnswerStore(record_dir=tmp_path))
    assert client.chat_json("sys", "q1") == {"a": 1}
    assert client.chat_json("sys", "q1") == {"a": 1}
    assert client.chat_json("sys", "q2") == {"b": 2}
    assert client.answer_stats() == {"replayed": 0, "repeated": 1, "asked": 2, "recorded": 2}
    assert len(client.sess.payloads) == 2


def test_the_two_routes_are_keyed_apart(tmp_path):
    client = LLMClient(LLMConfig(model="small", reasoning_model="big"), answers=LLMAnswerStore(record_dir=tmp_path))
    client.sess = _Session(['{"r": "small"}', '{"r": "big"}'])
    assert client.chat_json("s", "u") == {"r": "small"}
    assert client.chat_json("s", "u", reasoning=True) == {"r": "big"}


def test_a_resumed_run_keeps_its_answers_and_does_not_record_them_twice(tmp_path):
    _client(['{"a": 1}'], LLMAnswerStore(record_dir=tmp_path)).chat_json("s", "u")
    again = _client([], LLMAnswerStore(record_dir=tmp_path))
    assert again.chat_json("s", "u") == {"a": 1}
    assert again.answer_stats()["repeated"] == 1 and again.answer_stats()["recorded"] == 0


def test_disagreeing_shards_a_missing_replay_dir_and_a_corrupt_line_raise_and_a_torn_last_line_is_skipped(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        LLMAnswerStore(replay_dirs=[tmp_path / "missing"])
    with pytest.raises(ValueError, match="needs"):
        LLMAnswerStore()
    good = {"schema": "llm_answers/1", "key": "k", "answer": "x"}
    (tmp_path / "answers-h-1.jsonl").write_text(json.dumps(good) + "\n" + '{"schema": "llm_ans')
    assert LLMAnswerStore(replay_dirs=[tmp_path]).loaded == 1
    (tmp_path / "answers-h-2.jsonl").write_text(json.dumps(dict(good, answer="y")) + "\n")
    with pytest.raises(ValueError, match="disagree"):
        LLMAnswerStore(replay_dirs=[tmp_path])
    (tmp_path / "answers-h-2.jsonl").write_text("not json\n" + json.dumps(good) + "\n")
    with pytest.raises(ValueError, match="corrupt"):
        LLMAnswerStore(replay_dirs=[tmp_path])


def test_the_store_comes_from_the_environment_only_when_named(tmp_path):
    assert LLMAnswerStore.from_env({}) is None
    store = LLMAnswerStore.from_env({"LLM_ANSWERS_RECORD": str(tmp_path / "rec")})
    assert store.record_dir == tmp_path / "rec" and store.replay_dirs == ()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    both = LLMAnswerStore.from_env({"LLM_ANSWERS_REPLAY": "%s:%s" % (tmp_path / "a", tmp_path / "b")})
    assert both.record_dir is None and both.replay_dirs == (tmp_path / "a", tmp_path / "b")


def test_a_failed_call_records_nothing(tmp_path):
    client = _client([], LLMAnswerStore(record_dir=tmp_path))
    with pytest.raises(IndexError):
        client.chat_json("s", "u")
    assert not list(tmp_path.glob("answers-*.jsonl"))


class _ServerSession(_Session):
    """A server whose model stays listed in /api/ps for ``lingering`` polls after an unload."""

    def __init__(self, contents, lingering=2):
        super().__init__(contents)
        self.lingering, self.polls, self.unloads = lingering, 0, []

    def post(self, url, data=None, headers=None, timeout=None):
        if url.endswith("/api/generate"):
            self.unloads.append(json.loads(data))
            self.polls = self.lingering
            return _Reply("")
        return super().post(url, data, headers, timeout)

    def get(self, url, timeout=None):
        self.polls -= 1
        reply = _Reply("")
        reply._payload = {"models": [{"name": "m"}] if self.polls >= 0 else []}
        return reply


def test_the_exclusive_mode_unloads_the_model_and_waits_for_it_to_go_before_every_request(tmp_path):
    client = _client([])
    client.sess = _ServerSession(['{"a": 1}', '{"b": 2}'])
    client.exclusive = ExclusiveColdStart(tmp_path / "llm.lock")
    assert client.chat_json("sys", "q1") == {"a": 1}
    assert client.chat_json("sys", "q2") == {"b": 2}
    assert client.sess.unloads == [{"model": "m", "keep_alive": 0}] * 2
    assert [p["messages"][1]["content"] for p in client.sess.payloads] == ["q1", "q2"]
    assert client.exclusive.stats["requests"] == 2


def test_a_model_that_never_unloads_raises(tmp_path, monkeypatch):
    from sparx_agency.core.mapping.topology import llm_exclusive
    client = _client([])
    client.sess = _ServerSession(['{"a": 1}'], lingering=10 ** 6)
    client.exclusive = ExclusiveColdStart(tmp_path / "llm.lock")
    monkeypatch.setattr(llm_exclusive.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="still had m loaded"):
        client.exclusive.unload(client.sess, "http://x", "m", timeout_s=0.05)


def test_the_exclusive_lock_comes_from_the_environment(tmp_path):
    assert ExclusiveColdStart.from_env("http://127.0.0.1:11434", {}) is None
    assert ExclusiveColdStart.from_env("http://127.0.0.1:11434", {"LLM_EXCLUSIVE": "0"}) is None
    named = ExclusiveColdStart.from_env("http://h:1", {"LLM_EXCLUSIVE": str(tmp_path / "x.lock")})
    assert named.lock_path == tmp_path / "x.lock"
    default = ExclusiveColdStart.from_env("http://127.0.0.1:11434", {"LLM_EXCLUSIVE": "1"})
    assert default.lock_path.name == "sparx-llm-http_127_0_0_1_11434.lock"


def test_the_same_question_in_two_episodes_is_two_keys_and_each_episode_replays_its_own_answer(tmp_path):
    """2026-11-01: two scene workers asked one room-label question and got two answers; one store holds both."""
    first = _client(['{"c": 0.90}'], LLMAnswerStore(record_dir=tmp_path / "a"))
    first.set_answer_scope("scene1/000001")
    first.chat_json("s", "same question")
    second = _client(['{"c": 0.95}'], LLMAnswerStore(record_dir=tmp_path / "a"))   # another worker, same directory
    second.set_answer_scope("scene2/000007")
    second.chat_json("s", "same question")
    replay = _client([], LLMAnswerStore(replay_dirs=[tmp_path / "a"]))
    replay.set_answer_scope("scene2/000007")
    assert replay.chat_json("s", "same question") == {"c": 0.95}
    replay.set_answer_scope("scene1/000001")
    assert replay.chat_json("s", "same question") == {"c": 0.90}
    assert replay.answer_stats()["replayed"] == 2
    scopes = sorted(json.loads(line)["scope"] for shard in (tmp_path / "a").glob("answers-*.jsonl")
                    for line in shard.read_text().splitlines())
    assert scopes == ["scene1/000001", "scene2/000007"]


def test_a_scope_without_a_store_is_ignored():
    client = _client([])
    client.set_answer_scope("x/000001")
    assert client.answers is None


def test_the_frozen_description_does_not_change_as_the_run_records(tmp_path):
    """A resumed scene compares its configuration verbatim with the one it started with (2026-11-01)."""
    store = LLMAnswerStore(record_dir=tmp_path)
    before = store.describe()
    _client(['{"a": 1}'], store).chat_json("s", "u")
    assert LLMAnswerStore(record_dir=tmp_path).describe() == before
