"""Model readiness/provenance around existing LLMClient; no model downloads."""
from __future__ import annotations

from urllib.parse import urlsplit

from sparx_agency.core.planning.objnav.errors import ObjNavInternalError


def public_service_url(url):
    """Reject credentials in a URL that will be written to run metadata."""
    parsed = urlsplit(url)
    if (parsed.scheme not in ("http", "https") or not parsed.netloc
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Use an HTTP(S) service URL without credentials, query, or fragment; "
                         "keep the LLM key in LLM_API_KEY")
    return url.rstrip("/")


class VerifiedLLMClient:
    """Check the requested models, not merely a live port or an HTTP 401.

    Every model the config names (one vision model for live RPT) must be provisioned. Ollama
    exposes a digest per model, pinned for the run. OpenAI-compatible
    providers often expose only model ids: that limitation is recorded rather
    than claiming immutable remote weights. GET requests never start inference.
    """

    def __init__(self, client):
        self.client = client
        self.identity = None
        public_service_url(client.cfg.base_url)

    def health(self):
        cfg = self.client.cfg
        ollama = cfg.backend == "ollama"
        endpoint = "/api/tags" if ollama else "/models"
        response = self.client.sess.get(
            cfg.base_url.rstrip("/") + endpoint,
            headers=self.client._auth_header(), timeout=min(5.0, cfg.timeout_s))
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models" if ollama else "data", [])
        wanted = tuple(getattr(cfg, "models", lambda: (cfg.model,))())
        identity = {"models": {}}
        for model in wanted:
            names = {model}
            if ollama and ":" not in model:
                names.add(model + ":latest")
            matches = [item for item in models if isinstance(item, dict)
                       and item.get("name" if ollama else "id") in names]
            if not matches:
                raise ObjNavInternalError("Requested LLM model %r is not provisioned" % model)
            identity["models"][model] = {"digest": matches[0].get("digest"),
                                         "immutable_revision_exposed": bool(matches[0].get("digest"))}
            quant = (matches[0].get("details") or {}).get("quantization_level")
            if quant:
                identity["models"][model]["quantization"] = quant
        identity.update(model=cfg.model, reasoning_model=getattr(cfg, "reasoning_model", cfg.model),
                        digest=identity["models"][cfg.model]["digest"],
                        immutable_revision_exposed=identity["models"][cfg.model]["immutable_revision_exposed"])
        if self.identity is not None and identity != self.identity:
            raise ObjNavInternalError("LLM model revision changed during evaluation")
        self.identity = identity
        return identity

    def probe(self):
        """Prove the server accepts an image and returns a valid frontier probability, without using replay."""
        import numpy as np
        from sparx_agency.core.planning.object_search.frontier_views.scoring import ask_frontier, marked_view

        image = marked_view(np.zeros((96, 128, 3), dtype=np.uint8), (24, 16, 96, 80))
        answers, self.client.answers = getattr(self.client, "answers", None), None
        try:
            cfg = self.client.cfg
            if cfg.backend == "ollama":
                response = self.client.sess.post(cfg.base_url + "/api/show", json={"model": cfg.model},
                                                 timeout=min(10.0, cfg.timeout_s))
                response.raise_for_status()
                capabilities = response.json().get("capabilities")
                if capabilities is not None and "vision" not in capabilities:
                    raise ValueError("the configured Ollama model does not advertise vision")
            ask_frontier(self.client, "chair", [image])
        except (OSError, ValueError, RuntimeError) as exc:
            raise ObjNavInternalError("Frontier VLM %r failed its image probe: %s" % (self.client.cfg.model, exc)) from exc
        finally:
            self.client.answers = answers

    def chat_json(self, *args, **kwargs):
        self.health()
        return self.client.chat_json(*args, **kwargs)

    def set_answer_scope(self, scope):
        """Forwarded to the wrapped client's answer store, when it keeps one."""
        forward = getattr(self.client, "set_answer_scope", None)
        if forward is not None:
            forward(scope)

    def answer_stats(self):
        """The wrapped client's answer-store counters, or None."""
        stats = getattr(self.client, "answer_stats", None)
        return stats() if stats is not None else None


class VerifiedRoomClient(VerifiedLLMClient):
    """Preflight the original room labeller and node oracle, never an image request."""

    def probe(self):
        from sparx_agency.core.mapping.topology.search_node_oracle import (
            ROOM, SearchContext, SearchNode, SearchNodeOracle, system_prompt)
        nodes = (SearchNode(id=0, kind=ROOM, label="living_room", area_m2=20.0, frontier_clusters=1,
                            here=True, objects=("couch",)),
                 SearchNode(id=1, kind=ROOM, label="unknown", area_m2=6.0, frontier_clusters=1, exits=1))
        context = SearchContext(house="one living room entered; one unknown room; unexplored frontier reachable")
        oracle = SearchNodeOracle(self.client)
        answers, self.client.answers = self.client.answers, None
        try:
            reply = oracle.ask(oracle.prompt("toilet", nodes, context), system_prompt(nodes, context))
            if oracle.score(reply, nodes) is None:
                raise ObjNavInternalError("Room oracle failed its text probe: %r" % (reply,))
            if not isinstance(self.client.chat_json("Reply with JSON only.", 'Return {"ok": true}.'), dict):
                raise ObjNavInternalError("Room labeller failed its text probe")
        finally:
            self.client.answers = answers
