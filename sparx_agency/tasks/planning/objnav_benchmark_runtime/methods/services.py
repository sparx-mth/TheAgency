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
    """Check the requested model, not merely a live port or an HTTP 401.

    Ollama exposes a digest, which is pinned for the run. OpenAI-compatible
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
        names = {cfg.model}
        if ollama and ":" not in cfg.model:
            names.add(cfg.model + ":latest")
        matches = [item for item in models if isinstance(item, dict)
                   and item.get("name" if ollama else "id") in names]
        if not matches:
            raise ObjNavInternalError("Requested LLM model %r is not provisioned" % cfg.model)
        item = matches[0]
        identity = {"model": cfg.model, "digest": item.get("digest"),
                    "immutable_revision_exposed": bool(item.get("digest"))}
        if self.identity is not None and identity != self.identity:
            raise ObjNavInternalError("LLM model revision changed during evaluation")
        self.identity = identity
        return identity

    def chat_json(self, *args, **kwargs):
        self.health()
        return self.client.chat_json(*args, **kwargs)
