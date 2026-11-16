"""One bounded schema repair around the node oracle's own scoring."""
from __future__ import annotations

from sparx_agency.core.mapping.topology.search_node_oracle import UNEXPLORED_FLOOR, USABLE_SOURCES, SearchNodeOracle


#: Appended to the user prompt of the one repair call: the shape again, and nothing of what to write in it -- every
#: value a placeholder. (Until 2026-11-16 the shape was shown filled in -- ``"id":0 ... "p":50`` -- which put a
#: number in front of a model that had just failed to produce its own.)
REPAIR_NOTE = ('\nYour previous answer did not satisfy the schema. Return exactly '
               '{"home":"...","house":"...","stage":"...","pass":"first|second",'
               '"nodes":[{"id":<int>,"why":"<brief reason>","p":<int 0-100>}, ...]} '
               'with one entry per node id from the input, '
               'independent integer p values (do not normalise across nodes), '
               'and nothing outside the JSON object.')


class RepairingNodeOracle(SearchNodeOracle):
    """Never silently substitute a uniform policy; retry one malformed response.

    A capable model rarely breaks the schema, but a broken reply costs the
    loop point its judgement, and a single repair prompt is cheap beside the
    call that produced it. A second failure is the caller's to refuse -- the
    runtime raises rather than flying on a flat distribution.
    """

    def __init__(self, client, unexplored_floor=UNEXPLORED_FLOOR, flat_p=None, prompt=None):
        super().__init__(client, unexplored_floor=unexplored_floor, flat_p=flat_p, prompt=prompt)
        self.repair_attempts = 0
        self.repair_successes = 0

    def probabilities(self, target, nodes, context=None):
        result = super().probabilities(target, nodes, context)
        if result.source in USABLE_SOURCES:
            return result
        self.repair_attempts += 1
        user = self.prompt(target, nodes, context)
        user += REPAIR_NOTE
        reply, repaired = None, None
        try:
            reply = self.ask(user, self.system(nodes, context))
            repaired = self.score(reply, nodes, self.unexplored_floor)
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

