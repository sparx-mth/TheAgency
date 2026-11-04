"""Model readiness/provenance around existing LLMClient; no model downloads."""
from __future__ import annotations

from urllib.parse import urlsplit

from sparx_agency.core.planning.objnav.errors import ObjNavInternalError

#: The probe's two nodes: a node prompt of the run's own shape, small enough to answer in seconds.
PROBE_TARGET = "toilet"
PROBE_HOUSE = ("rooms found: living_room (entered); 1 room unidentified; 1 opening not yet looked into; "
               "unexplored frontier reachable; actions used about 0 of 500")


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

    Every model the config names -- the default one for the cheap calls and
    the reasoning one for the node oracle -- must be provisioned. Ollama
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
        identity.update(model=cfg.model, reasoning_model=getattr(cfg, "reasoning_model", cfg.model),
                        digest=identity["models"][cfg.model]["digest"],
                        immutable_revision_exposed=identity["models"][cfg.model]["immutable_revision_exposed"])
        if self.identity is not None and identity != self.identity:
            raise ObjNavInternalError("LLM model revision changed during evaluation")
        self.identity = identity
        return identity

    def probe(self):
        """Ask both routes one real question with the run's own options; raise unless both answer usably.

        :meth:`health` proves the models are provisioned, not that the server
        answers the node oracle's request -- its model, window, reply cap and
        timeout. The 2026-10-28 sweep passed every health check and then got
        no oracle answer in 2,000 episodes. Run once, before the first episode.

        Raises:
            ObjNavInternalError: A route raised, or the oracle's reply does not
                score the probe's nodes.
        """
        from sparx_agency.core.mapping.topology.search_node_oracle import (
            ROOM, SearchContext, SearchNode, SearchNodeOracle, system_prompt)
        nodes = (SearchNode(id=0, kind=ROOM, label="living_room", area_m2=20.0, frontier_clusters=1,
                            here=True, objects=("couch",)),
                 SearchNode(id=1, kind=ROOM, label="unknown", area_m2=6.0, frontier_clusters=1, exits=1))
        oracle = SearchNodeOracle(self.client)
        context = SearchContext(house=PROBE_HOUSE)
        # The probe proves the SERVER answers: an answer store would replay it, and record it into the run.
        answers, self.client.answers = getattr(self.client, "answers", None), None
        try:
            try:
                reply = oracle.ask(oracle.prompt(PROBE_TARGET, nodes, context), system_prompt(nodes, context))
            except Exception as exc:
                raise ObjNavInternalError("The node oracle's route (%s) failed a real oracle prompt: %s"
                                          % (self.client.cfg.reasoning_model, exc)) from exc
            if oracle.score(reply, nodes) is None:
                raise ObjNavInternalError("The node oracle's route (%s) answered a real oracle prompt unusably: %r"
                                          % (self.client.cfg.reasoning_model, reply))
            try:
                label = self.client.chat_json('Reply with JSON only.', 'Return {"ok": true}.')
            except Exception as exc:
                raise ObjNavInternalError("The room-label route (%s) failed: %s"
                                          % (self.client.cfg.model, exc)) from exc
            if not isinstance(label, dict):
                raise ObjNavInternalError("The room-label route (%s) did not answer a JSON object"
                                          % self.client.cfg.model)
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
