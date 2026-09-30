"""Tiny HTTP wrapper around a local LLM for the scene-graph stack.

Two backends, chosen by the ``LLM_BACKEND`` env var:

``LLM_BACKEND=ollama`` (default)
    Talks to an Ollama server at ``LLM_BASE_URL`` (default
    ``http://localhost:11434``). Model name in ``LLM_MODEL``
    (default ``qwen2.5:3b-instruct``). No API key needed.

``LLM_BACKEND=openai`` (or ``openai-compat``)
    Talks to any OpenAI-compatible ``/chat/completions`` endpoint
    (OpenAI, Together, Groq, vLLM, llama-cpp-server, LM Studio, ...).
    Needs ``LLM_BASE_URL`` (e.g. ``https://api.openai.com/v1``) and,
    if the server requires it, ``LLM_API_KEY``.

 Two models, one client. ``LLM_MODEL`` (default ``qwen2.5:3b-instruct``) serves
the cheap, frequent calls; ``LLM_REASONING_MODEL`` (default
``qwen2.5:14b-instruct``, with ``LLM_REASONING_TIMEOUT_S``,
``LLM_REASONING_MAX_TOKENS`` and ``LLM_REASONING_NUM_CTX``) serves the one
judgement per loop point that the search cannot afford to get wrong -- see
:attr:`LLMConfig.reasoning_model`. Callers select it with
``chat_json(..., reasoning=True)``.

The client keeps one ``requests.Session`` per instance, forces JSON
output when possible (Ollama ``format: json`` / OpenAI
``response_format: json_object``), and retries once with a short
backoff on network errors. Callers get a plain Python dict parsed from
the model's JSON reply.

This module is ROS-free on purpose and has zero import-time side
effects: nothing reads the environment or opens a connection until
:meth:`LLMConfig.from_env` / a chat method is called.

Note on the JSON rescue: :func:`sparx_agency.core.mapping.topology.
llm_nav_planner._extract_json_dict` is a deliberate separate sibling —
it returns ``None`` on failure and repairs trailing commas for a
callable-based planner, while :func:`_best_effort_json` here raises
``ValueError`` (carrying the raw text) per this client's contract.
Ported from the SJTU ``semantic_mapper/llm_client.py``.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import requests


def _seed_from_env(raw: str) -> Optional[int]:
    """``""`` or ``"none"`` means send no seed; anything else must parse."""
    text = str(raw).strip().lower()
    if text in ("", "none", "null"):
        return None
    return int(text)


# ---------------------------------------------------------------------
#  Config
# ---------------------------------------------------------------------
@dataclass
class LLMConfig:
    """Connection settings for :class:`LLMClient`.

    Attributes:
        backend: ``"ollama"``, ``"openai"`` or ``"openai-compat"``.
        base_url: Server root URL (no trailing slash).
        model: Model name understood by the backend -- the DEFAULT model, for
            the cheap, frequent calls (room-type classification, target-name
            matching).
        api_key: Bearer token for OpenAI-compatible servers ("" = none).
        temperature: Default sampling temperature. 0.0, not 0.2: an
            "optimal" visit order that changes when the scene graph did not
            is not auditable, and a search nobody can replay from a recording
            is a search nobody can debug.
        timeout_s: Per-request timeout in seconds, for the default model.
        seed: Sampling seed sent to backends that accept one. None sends no
            seed at all. With temperature 0 this is belt and braces, but
            Ollama's greedy decode is not bit-stable across keep-alive
            reloads without it.
        keep_alive: How long the backend should hold the model in memory.
            Ollama unloads after 5 minutes idle by default, and the oracle
            ticks every 10 s but can be quiet for longer -- a cold reload
            costs more than ``timeout_s`` and lands the tick in the uniform
            fallback, which reads as "the LLM said nothing useful" when in
            fact it was never asked.
        max_tokens: Cap on the reply. A small model that falls into repeating
            a JSON fragment otherwise runs until ``timeout_s``, producing the
            same silent uniform-fallback tick.
        reasoning_model: The model for the REASONING call -- the search
            oracle that values every node of the search (rooms and stairs)
            from the whole state of the map. That judgement is where a small
            model fails (a 3B double-counts search effort into semantics and
            cannot weigh a staircase against an unknown room), and the search
            makes it only at loop points, so a larger and slower model is
            affordable here. ``qwen2.5:14b-instruct`` by default: ~9 GB at
            Q4, which fits beside Habitat in the 30 GB of the development
            laptop and answers a dozen-node prompt in about a minute on its
            CPU; ``qwen2.5:32b-instruct`` (~20 GB, several minutes) where the
            RAM allows, or a hosted model through the OpenAI-compatible
            backend. A model named here must be provisioned: the runtime's
            health check looks for it and refuses to run without it, rather
            than falling back to the small model unannounced.
        reasoning_timeout_s: Timeout for that call. Generous by default: the
            search waits for its answer rather than running on a guess, and
            a 14B model on a CPU takes a minute or two per call.
        reasoning_max_tokens: Reply cap for that call. A dozen nodes with a
            reason each is ~600 tokens; models that think aloud need room.
        reasoning_num_ctx: Context window requested for that call (Ollama
            ``num_ctx``). The node prompt is ~1.5K tokens plus the reply;
            Ollama truncates a request that outgrows its window from the
            FRONT, which would silently drop the system prompt's rules, so
            the window is asked for explicitly rather than left to the
            server's default.
    """

    backend: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:3b-instruct"
    api_key: str = ""
    temperature: float = 0.0
    timeout_s: float = 30.0
    seed: Optional[int] = 0
    keep_alive: str = "30m"
    max_tokens: int = 768
    reasoning_model: str = "qwen2.5:14b-instruct"
    reasoning_timeout_s: float = 600.0
    reasoning_max_tokens: int = 2048
    reasoning_num_ctx: int = 8192

    def __post_init__(self) -> None:
        if not str(self.reasoning_model).strip():
            self.reasoning_model = self.model

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Build a config from ``LLM_*`` environment variables."""

        def get(k: str, d: str) -> str:
            return os.environ.get(k, d)

        return cls(
            backend=get("LLM_BACKEND", "ollama").strip().lower(),
            base_url=get("LLM_BASE_URL", "http://localhost:11434").rstrip("/"),
            model=get("LLM_MODEL", "qwen2.5:3b-instruct"),
            api_key=get("LLM_API_KEY", ""),
            temperature=float(get("LLM_TEMPERATURE", "0")),
            timeout_s=float(get("LLM_TIMEOUT_S", "30")),
            seed=_seed_from_env(get("LLM_SEED", "0")),
            keep_alive=get("LLM_KEEP_ALIVE", "30m"),
            max_tokens=int(get("LLM_MAX_TOKENS", "768")),
            reasoning_model=get("LLM_REASONING_MODEL", "qwen2.5:14b-instruct"),
            reasoning_timeout_s=float(get("LLM_REASONING_TIMEOUT_S", "600")),
            reasoning_max_tokens=int(get("LLM_REASONING_MAX_TOKENS", "2048")),
            reasoning_num_ctx=int(get("LLM_REASONING_NUM_CTX", "8192")),
        )

    def models(self) -> Tuple[str, ...]:
        """Every model this config names, deduplicated -- what a health check must find provisioned."""
        return tuple(dict.fromkeys((self.model, self.reasoning_model)))


# ---------------------------------------------------------------------
#  Client
# ---------------------------------------------------------------------
class LLMClient:
    """JSON-first chat client over Ollama or an OpenAI-compatible server.

    Call as::

        llm = LLMClient.from_env()
        reply = llm.chat_json(system="...", user="...")

    ``reply`` is whatever dict the model wrote. If the reply wasn't
    valid JSON, :meth:`chat_json` raises ``ValueError`` with the raw
    text for logging.
    """

    def __init__(self, cfg: Optional[LLMConfig] = None):
        self.cfg = cfg or LLMConfig.from_env()
        self.sess = requests.Session()

    # -- Public --------------------------------------------------------
    @classmethod
    def from_env(cls) -> "LLMClient":
        """Build a client whose config is read from the environment."""
        return cls(LLMConfig.from_env())

    def chat_json(self, system: str, user: str,
                  temperature: Optional[float] = None,
                  reasoning: bool = False) -> Dict[str, Any]:
        """Send system+user prompt, return the parsed JSON dict.

        Args:
            system: The system prompt.
            user: The user prompt.
            temperature: Sampling temperature; the config's when None.
            reasoning: Use the config's ``reasoning_model``, with its own
                timeout and reply cap, instead of the default model. For the
                one call per loop point that values every node of the search.

        Raises:
            ValueError: The model's reply was not rescuable JSON.
            RuntimeError: The HTTP request failed after retry, or the
                server reply had an unexpected shape.
        """
        text = self.chat_text(system, user, temperature, reasoning=reasoning)
        return _best_effort_json(text)

    def chat_text(self, system: str, user: str,
                  temperature: Optional[float] = None,
                  reasoning: bool = False) -> str:
        """Send system+user prompt, return the raw reply text."""
        t = self.cfg.temperature if temperature is None else float(temperature)
        route = _Route(model=self.cfg.reasoning_model if reasoning else self.cfg.model,
                       timeout_s=self.cfg.reasoning_timeout_s if reasoning else self.cfg.timeout_s,
                       max_tokens=self.cfg.reasoning_max_tokens if reasoning else self.cfg.max_tokens,
                       num_ctx=int(self.cfg.reasoning_num_ctx) if reasoning else None)
        if self.cfg.backend == "ollama":
            return self._ollama_chat(system, user, t, route)
        if self.cfg.backend in ("openai", "openai-compat"):
            return self._openai_chat(system, user, t, route)
        raise ValueError(f"Unknown LLM_BACKEND: {self.cfg.backend!r}")

    def ping(self) -> bool:
        """Return True if the server answers. Never raises."""
        try:
            if self.cfg.backend == "ollama":
                r = self.sess.get(f"{self.cfg.base_url}/api/tags",
                                  timeout=3.0)
                return r.status_code == 200
            r = self.sess.get(f"{self.cfg.base_url}/models",
                              headers=self._auth_header(), timeout=3.0)
            # Many servers return 401 on /models unauthenticated — that
            # still means "reachable".
            return r.status_code in (200, 401)
        except requests.RequestException:
            return False

    # -- Backends ------------------------------------------------------
    def _ollama_chat(self, system: str, user: str, temperature: float,
                     route: "_Route") -> str:
        url = f"{self.cfg.base_url}/api/chat"
        options = {"temperature": temperature,
                   "num_predict": int(route.max_tokens)}
        if route.num_ctx is not None:
            options["num_ctx"] = int(route.num_ctx)
        if self.cfg.seed is not None:
            options["seed"] = int(self.cfg.seed)
        payload = {
            "model": route.model,
            "stream": False,
            "format": "json",             # ask Ollama to enforce JSON
            "keep_alive": str(self.cfg.keep_alive),
            "options": options,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        r = self._post_with_retry(url, payload, timeout_s=route.timeout_s)
        data = r.json()
        # Ollama's reply: {"message": {"role":"assistant","content":"..."}}
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as e:
            raise RuntimeError(f"unexpected Ollama reply: {data}") from e

    def _openai_chat(self, system: str, user: str, temperature: float,
                     route: "_Route") -> str:
        url = f"{self.cfg.base_url}/chat/completions"
        payload = {
            "model": route.model,
            "temperature": temperature,
            "max_tokens": int(route.max_tokens),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        r = self._post_with_retry(url, payload,
                                  extra_headers=self._auth_header(),
                                  timeout_s=route.timeout_s)
        data = r.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"unexpected OpenAI-compat reply: {data}") from e

    # -- HTTP plumbing -------------------------------------------------
    def _auth_header(self) -> Dict[str, str]:
        if self.cfg.api_key:
            return {"Authorization": f"Bearer {self.cfg.api_key}"}
        return {}

    def _post_with_retry(self, url: str, payload: Dict[str, Any],
                         extra_headers: Optional[Dict[str, str]] = None,
                         tries: int = 2, timeout_s: Optional[float] = None):
        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        timeout = self.cfg.timeout_s if timeout_s is None else float(timeout_s)
        last_err: Optional[Exception] = None
        for attempt in range(tries):
            try:
                r = self.sess.post(url, data=json.dumps(payload),
                                   headers=headers,
                                   timeout=timeout)
                r.raise_for_status()
                return r
            except requests.RequestException as e:
                last_err = e
                if attempt + 1 < tries:
                    time.sleep(0.4)
        raise RuntimeError(f"LLM request failed after {tries} tries: {last_err}")


@dataclass(frozen=True)
class _Route:
    """Which model a call goes to, and how long and how wide it may be."""

    model: str
    timeout_s: float
    max_tokens: int
    num_ctx: Optional[int] = None


# ---------------------------------------------------------------------
#  Reply-field coercion
# ---------------------------------------------------------------------
_TRUE_WORDS = frozenset(("true", "yes", "y", "1"))
_FALSE_WORDS = frozenset(("false", "no", "n", "0", ""))


def coerce_bool(value: Any, default: bool = False) -> bool:
    """Read a model's boolean field, which is often not a boolean.

    ``bool()`` is the wrong tool and fails in the dangerous direction:
    small instruct models routinely answer ``{"match": "false"}`` with the
    word quoted, and ``bool("false")`` is ``True`` because the string is
    non-empty. Flown consequence: qwen2.5:3b-instruct returned the string
    ``"false"`` for every non-matching class, the target watcher read every
    detection as a hit, and the hospital search latched ``/target_seen`` on
    the first object it ever saw — a shelf, carrying the model's own
    reason "CLASS is a different object".

    Numbers are safe under ``float()`` (``float("0.9")`` is 0.9), so this
    quirk is specific to booleans and this helper is the only place that
    should read one out of a reply.

    Args:
        value: The raw field from the parsed reply (bool, str, or number).
        default: Returned when the value is absent or unrecognised.

    Returns:
        The boolean the model meant.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        word = value.strip().lower()
        if word in _TRUE_WORDS:
            return True
        if word in _FALSE_WORDS:
            return False
    return default


# ---------------------------------------------------------------------
#  JSON rescue
# ---------------------------------------------------------------------
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def _best_effort_json(text: str) -> Dict[str, Any]:
    """Parse a model reply that may be fenced or padded with prose.

    Small models sometimes wrap their JSON in ``` fences or prepend a
    one-line explanation. Strip the fence, pull out the first ``{...}``
    block, and try ``json.loads``.

    Raises:
        ValueError: No parseable JSON object was found; the raw text is
            included in the message for logging.
    """
    t = text.strip()
    # ``` fences
    if t.startswith("```"):
        t = t.strip("`")
        # drop an optional 'json' tag on the first line
        nl = t.find("\n")
        if nl > 0:
            t = t[nl + 1:]
    # Try direct
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # Pull first { ... }
    m = _JSON_OBJ.search(t)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"LLM did not return valid JSON. Raw:\n{text}")
