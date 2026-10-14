"""One bounded schema repair around the node oracle's own scoring."""
from __future__ import annotations

from sparx_agency.core.mapping.topology.search_node_oracle import (
    HOME_FLOOR, UNEXPLORED_ELSEWHERE, UNEXPLORED_FLOOR, SearchNodeOracle, system_prompt)


class RepairingNodeOracle(SearchNodeOracle):
    """Never silently substitute a uniform policy; retry one malformed response.

    A capable model rarely breaks the schema, but a broken reply costs the
    loop point its judgement, and a single repair prompt is cheap beside the
    call that produced it. A second failure is the caller's to refuse -- the
    runtime raises rather than flying on a flat distribution.
    """

    def __init__(self, client, unexplored_floor=UNEXPLORED_FLOOR, unexplored_elsewhere=UNEXPLORED_ELSEWHERE,
                 home_floor=HOME_FLOOR):
        super().__init__(client, unexplored_floor=unexplored_floor, unexplored_elsewhere=unexplored_elsewhere,
                         home_floor=home_floor)
        self.repair_attempts = 0
        self.repair_successes = 0

    def probabilities(self, target, nodes, context=None):
        result = super().probabilities(target, nodes, context)
        if result.source == "llm":
            return result
        self.repair_attempts += 1
        user = self.prompt(target, nodes, context)
        user += ('\nYour previous answer did not satisfy the schema. Return exactly '
                 '{"home":"...","house":"...","stage":"...","pass":"first|second",'
                 '"nodes":[{"id":0,"why":"brief reason","p":50}, ...]} '
                 'with one entry per actual node id from the input (not the example id), '
                 'independent integer p values in 0-100 (do not normalise across nodes), '
                 'and nothing outside the JSON object.')
        reply, repaired = None, None
        try:
            reply = self.ask(user, system_prompt(nodes, context))
            repaired = self.score(reply, nodes, self.unexplored_floor, self.unexplored_elsewhere, self.home_floor)
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

