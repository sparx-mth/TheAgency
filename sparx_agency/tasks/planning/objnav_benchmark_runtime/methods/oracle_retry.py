"""One bounded schema repair around the existing semantic oracle mathematics."""
from __future__ import annotations

from sparx_agency.core.mapping.topology.search_oracle import (
    SearchOracle, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, format_rooms_block)


class RepairingSearchOracle(SearchOracle):
    """Never silently substitute a uniform policy; retry one malformed response."""

    def __init__(self, client):
        super().__init__(client)
        self.repair_attempts = 0
        self.repair_successes = 0

    def probabilities(self, target, rooms):
        result = super().probabilities(target, rooms)
        if result.source == "llm":
            return result
        self.repair_attempts += 1
        user = USER_PROMPT_TEMPLATE.format(target=target, rooms_block=format_rooms_block(rooms), n_rooms=len(rooms))
        user += ('\nYour previous answer did not satisfy the schema. Return exactly '
                 '{"rooms":[{"id":0,"score":50,"why":"brief reason"}]} '
                 'using every actual room id from the input, not the example id. '
                 'No markdown or prose outside the JSON object.')
        try:
            reply = self._client.chat_json(SYSTEM_PROMPT, user)
            repaired = self._score(reply, rooms)
        except Exception:
            repaired = None
        if repaired is not None:
            self.repair_successes += 1
            return repaired
        # The runtime's existing source check raises. The failed attempt is
        # never converted into a scored navigation failure or a uniform policy.
        return result

