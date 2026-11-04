"""One Ollama request at a time across processes, each on a freshly loaded model: an answer set by its request alone.

Measured on 2026-10-31 (Ollama 0.33.2, ``qwen2.5:3b-instruct`` Q4 on an RTX 5070
Laptop, ``OLLAMA_NUM_PARALLEL=1``, temperature 0, seed 17, 20-40 recorded
prompts of the room-label and node-oracle routes, sequential unless stated):

- Sent in another order, 13-18 of 40 answers changed (15 of 17 beyond the
  reason text) -- the same prompts, one at a time.
- A one-token flush request before each prompt changed nothing: the server
  keeps the KV state of many earlier prompts (a prompt seen 19 long prompts
  earlier was evaluated in 30 ms, a fresh one in 520 ms), so a flush does not
  empty what the next prompt reuses.
- A per-request tag at the head of the system prompt (no two requests share
  more than the chat template's opening) still changed 15-17 of 20 fresh
  answers with the order: the shared opening's state is computed once, by
  whichever request came first.
- Unloading the model before every request (this module) changed none of 20
  between two orders: nothing survives from one request to the next.

The cost is a model load and a cold prompt evaluation per request: 6.6 s a
call against 3.7 s (prompt evaluation 0.73 s against 0.09 s). The lock holds
only between processes that use it; anything else sending to the same server
breaks the guarantee and is itself disturbed by the unloads. POSIX only
(``fcntl``).
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

#: ``LLM_EXCLUSIVE``: unset, ``""`` or ``"0"`` off; ``"1"`` on, with a lock file per server under the temp dir;
#: anything else is the lock file's path.
EXCLUSIVE_ENV = "LLM_EXCLUSIVE"
UNLOAD_TIMEOUT_S = 60.0


class ExclusiveColdStart:
    """A cross-process lock around each request, and the model unload done under it before the request.

    Args:
        lock_path: The lock file every participating process opens.
    """

    def __init__(self, lock_path: Path):
        self.lock_path = Path(lock_path)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.stats = {"requests": 0, "wait_s": 0.0, "unload_s": 0.0}

    @classmethod
    def from_env(cls, base_url: str, environ: Optional[Dict[str, str]] = None) -> Optional["ExclusiveColdStart"]:
        """The lock ``LLM_EXCLUSIVE`` asks for, or None."""
        value = (os.environ if environ is None else environ).get(EXCLUSIVE_ENV, "").strip()
        if value in ("", "0"):
            return None
        if value == "1":
            name = re.sub(r"[^A-Za-z0-9]+", "_", base_url).strip("_")
            return cls(Path(tempfile.gettempdir()) / ("sparx-llm-%s.lock" % name))
        return cls(Path(value))

    def describe(self) -> Dict[str, Any]:
        """What a run's frozen configuration records of the lock."""
        return {"lock": str(self.lock_path),
                "cold_start": "the model unloaded (keep_alive 0) and confirmed gone (/api/ps), under a host-wide "
                              "lock, before every request"}

    @contextlib.contextmanager
    def held(self) -> Iterator[None]:
        """Hold the host-wide lock for one unload-and-request."""
        with open(self.lock_path, "a") as fh:
            started = time.monotonic()
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            self.stats["wait_s"] += time.monotonic() - started
            self.stats["requests"] += 1
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def unload(self, session, base_url: str, model: str, timeout_s: float = UNLOAD_TIMEOUT_S) -> None:
        """Ask the server to drop ``model`` and wait until ``/api/ps`` no longer lists it.

        Raises:
            RuntimeError: The model was still loaded after ``timeout_s``.
        """
        started = time.monotonic()
        session.post(base_url + "/api/generate", data=json.dumps({"model": model, "keep_alive": 0}),
                     headers={"Content-Type": "application/json"}, timeout=timeout_s).raise_for_status()
        while time.monotonic() - started < timeout_s:
            listed = session.get(base_url + "/api/ps", timeout=5.0)
            listed.raise_for_status()
            names = {m.get("name") or m.get("model") for m in (listed.json().get("models") or [])}
            if model not in names:
                self.stats["unload_s"] += time.monotonic() - started
                return
            time.sleep(0.1)
        raise RuntimeError("Ollama still had %s loaded %.0f s after the unload" % (model, timeout_s))
