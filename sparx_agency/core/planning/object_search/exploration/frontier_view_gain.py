"""Floor-local view gain and trail discount, shared by the fallback ladder and historical room walks."""
from sparx_agency.core.planning.exploration.frontier_ranking import trail_visits
from sparx_agency.core.planning.exploration.view_gain import UnknownView, cone_from_camera


class FrontierViewGain:
    def __init__(self, policy):
        self.policy = policy
        self._view_cache = self._visits_cache = None

    def _visits(self, obs):
        p = self.policy
        ranking = p.sweep.settings.ranking
        sight = getattr(p, "sight", None)
        if sight is None or getattr(ranking, "visit_weight", 0.0) <= 0.0:
            return None
        if self._visits_cache is None or self._visits_cache[0] != int(obs.step):
            visits = trail_visits([(x, y) for _, x, y, _, _ in sight.poses()], ranking.visit_radius_m)
            self._visits_cache = (int(obs.step), visits)
        return self._visits_cache[1]

    def _unknown_view(self, obs, world):
        cached = self._view_cache
        if cached is not None and cached[0] == int(obs.step) and cached[1] is world:
            return cached[2]
        p = self.policy
        camera = p.episode.camera
        k = camera.intrinsics
        sight = getattr(p, "sight", None)
        view = UnknownView(world, cone_from_camera(k.width, k.fx, float(camera.max_depth_m),
                                                   p.glance_settings.blind_m if hasattr(p, "glance_settings") else 0.0),
                           resolved=sight.resolved(world) if sight is not None else None)
        self._view_cache = (int(obs.step), world, view)
        return view
