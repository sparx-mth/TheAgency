"""Record and replay LLM answers keyed on the exact request.

A run that records writes every answer its model gave -- the room-label calls
and the node oracle alike, which both reach the model through
:meth:`~sparx_agency.core.mapping.topology.llm_client.LLMClient.chat_text` --
to disk; a run that replays serves the same answer to the same request and
asks the model only what it has no answer for. Two runs that put the same
questions then get the same answers, whatever the server would have said the
second time: on 2026-10-31 the temperature-0, seeded ``qwen2.5:3b`` oracle
answered identical prompts differently in two runs of the same 40 episodes
(``LESSONS.md``), and 37 of the 40 parted at their first call.

The key is everything that shapes the answer: backend, model, system and user
prompt, temperature, seed, context window, reply cap and the JSON format --
never the timeout, keep-alive or GPU placement. A request asked again within a
run is answered from the store as well, so with a store attached an answer is
a function of its request.

On disk a store is a directory of JSON-lines shards, one per process
(``answers-<host>-<pid>.jsonl``), so the scene workers of a campaign never
write one file. Each line holds the key, the route, the full prompts and the
answer, so a recording also says what was asked. Opening a store reads every
shard of its replay directories and of its own record directory (a resumed
run keeps its answers).

ROS-free, no third-party imports, no side effects until a store is opened.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

SCHEMA = "llm_answers/1"
SHARD_GLOB = "answers-*.jsonl"

#: Environment variables of :meth:`LLMAnswerStore.from_env`.
RECORD_ENV = "LLM_ANSWERS_RECORD"
REPLAY_ENV = "LLM_ANSWERS_REPLAY"


def request_key(backend: str, model: str, system: str, user: str, temperature: float, seed: Optional[int],
                num_ctx: Optional[int], max_tokens: int, response_format: str = "json") -> str:
    """SHA-256 of the request's answer-shaping fields, canonically serialised."""
    canonical = json.dumps({"backend": backend, "model": model, "system": system, "user": user,
                            "temperature": float(temperature), "seed": seed,
                            "num_ctx": None if num_ctx is None else int(num_ctx), "max_tokens": int(max_tokens),
                            "format": response_format},
                           sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class LLMAnswerStore:
    """Answers by request key, loaded from replay and record directories, appended to this process's shard.

    Args:
        record_dir: Where this run writes the answers it uses (created); None
            records nothing.
        replay_dirs: Directories whose answers are served first, read once at
            opening. A key recorded twice with different answers raises: a
            store that holds two answers to one question replays neither.

    Raises:
        ValueError: Neither directory was given, a replay directory does not
            exist, or the shards disagree on an answer.
    """

    def __init__(self, record_dir: Optional[Path] = None, replay_dirs: Sequence[Path] = ()):
        if record_dir is None and not replay_dirs:
            raise ValueError("an LLM answer store needs a record directory, a replay directory, or both")
        self.record_dir = None if record_dir is None else Path(record_dir)
        self.replay_dirs = tuple(Path(d) for d in replay_dirs)
        self._answers: Dict[str, str] = {}
        self._recorded: set = set()          # keys already in the record directory
        self._replayable: set = set()        # keys the replay directories answered
        self.loaded = 0
        for directory in self.replay_dirs:
            if not directory.is_dir():
                raise ValueError("LLM answer replay directory %s does not exist" % directory)
            self.loaded += self._load(directory, own=False)
        if self.record_dir is not None:
            self.record_dir.mkdir(parents=True, exist_ok=True)
            self.loaded += self._load(self.record_dir, own=True)
        self._shard = None if self.record_dir is None else self.record_dir / (
            "answers-%s-%d.jsonl" % (socket.gethostname(), os.getpid()))
        # replayed: answered from a replay directory; repeated: a question this run had already put;
        # asked: no answer stored, the model was asked; recorded: lines written to this process's shard.
        self.stats = {"replayed": 0, "repeated": 0, "asked": 0, "recorded": 0}

    @classmethod
    def from_env(cls, environ: Optional[Dict[str, str]] = None) -> Optional["LLMAnswerStore"]:
        """The store ``LLM_ANSWERS_RECORD`` / ``LLM_ANSWERS_REPLAY`` (``os.pathsep``-separated) name; None if neither."""
        env = os.environ if environ is None else environ
        record = env.get(RECORD_ENV, "").strip()
        replay = [p for p in env.get(REPLAY_ENV, "").split(os.pathsep) if p.strip()]
        if not record and not replay:
            return None
        return cls(Path(record) if record else None, [Path(p.strip()) for p in replay])

    def describe(self) -> Dict[str, Any]:
        """What a run's frozen configuration records of the store."""
        return {"schema": SCHEMA, "record": None if self.record_dir is None else str(self.record_dir),
                "replay": [str(d) for d in self.replay_dirs], "loaded": self.loaded,
                "key": "sha256 of backend, model, system, user, temperature, seed, num_ctx, max_tokens, format"}

    def get(self, key: str) -> Optional[str]:
        """The stored answer to ``key``, or None; counts it as replayed, repeated or asked."""
        answer = self._answers.get(key)
        if answer is None:
            self.stats["asked"] += 1
        else:
            self.stats["replayed" if key in self._replayable else "repeated"] += 1
        return answer

    def put(self, key: str, answer: str, entry: Dict[str, Any]) -> None:
        """Keep ``answer`` for ``key`` and write it to this process's shard unless the record directory has it."""
        self._answers.setdefault(key, answer)
        if self._shard is None or key in self._recorded:
            return
        line = dict(entry, schema=SCHEMA, key=key, answer=answer,
                    recorded_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with self._shard.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        self._recorded.add(key)
        self.stats["recorded"] += 1

    def _load(self, directory: Path, own: bool) -> int:
        count = 0
        for shard in sorted(directory.glob(SHARD_GLOB)):
            for entry in _read_lines(shard):
                key, answer = entry["key"], entry["answer"]
                known = self._answers.get(key)
                if known is not None and known != answer:
                    raise ValueError("LLM answer shards disagree on key %s (%s)" % (key, shard))
                if known is None:
                    count += 1
                self._answers[key] = answer
                if own:
                    self._recorded.add(key)
                else:
                    self._replayable.add(key)
        return count


def _read_lines(path: Path) -> Iterable[Dict[str, Any]]:
    """The shard's entries; a torn last line (a killed writer) is skipped, any other bad line raises."""
    lines: List[str] = path.read_text(encoding="utf-8").splitlines()
    for number, text in enumerate(lines):
        if not text.strip():
            continue
        try:
            entry = json.loads(text)
        except ValueError:
            if number == len(lines) - 1:
                return
            raise ValueError("corrupt LLM answer shard %s, line %d" % (path, number + 1))
        if entry.get("schema") != SCHEMA:
            raise ValueError("LLM answer shard %s holds schema %r, not %r" % (path, entry.get("schema"), SCHEMA))
        yield entry
