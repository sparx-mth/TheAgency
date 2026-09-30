"""One bounded schema repair around the node oracle's own scoring."""
from __future__ import annotations

from sparx_agency.core.mapping.topology.search_node_oracle import SearchNodeOracle


class RepairingNodeOracle(SearchNodeOracle):
    """Never silently substitute a uniform policy; retry one malformed response.

    A capable model rarely breaks the schema, but a broken reply costs the
    loop point its judgement, and a single repair prompt is cheap beside the
    call that produced it. A second failure is the caller's to refuse -- the
    runtime raises rather than flying on a flat distribution.
    """

    def __init__(self, client):
        super().__init__(client)
        self.repair_attempts = 0
        self.repair_successes = 0

    def probabilities(self, target, nodes, context=None):
        result = super().probabilities(target, nodes, context)
        if result.source == "llm":
            return result
        self.repair_attempts += 1
        user = self.prompt(target, nodes, context)
        user += ('\nYour previous answer did not satisfy the schema. Return exactly '
                 '{"nodes":[{"id":0,"why":"brief reason","p":50}, ...],"elsewhere":10} '
                 'with one entry per actual node id from the input (not the example id), '
                 'integer p values that sum with elsewhere to 100, and nothing outside the JSON object.')
        reply, repaired = None, None
        try:
            reply = self.ask(user)
            repaired = self.score(reply, nodes)
        except Exception:
            repaired = None
        if repaired is not None and reply is not None:
            self.repair_successes += 1
            # Kept under the prompt the model SHOULD have answered, so an
            # unchanged query next time reuses the repaired reply.
            self.remember(self.prompt(target, nodes, context), reply)
            return repaired
        # The runtime's existing source check raises. The failed attempt is
        # never converted into a scored navigation failure or a uniform policy.
        return result

