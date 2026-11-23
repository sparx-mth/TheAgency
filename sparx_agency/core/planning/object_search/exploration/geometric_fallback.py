"""Same bounded recovery ladder, with geometric frontier utility and no semantic or inter-storey destinations."""
from sparx_agency.core.planning.object_search.exploration.exploration_fallback import ExplorationFallback


class GeometricFallback(ExplorationFallback):
    def _inventory(self):
        return self.policy.frontiers.inventory

    def _floor_exit(self, obs, world, reason):
        return None

    def _exits(self, world, goals, obs=None):
        # The inventory is already ranked on size, heading and geodesic distance.
        return goals, []
