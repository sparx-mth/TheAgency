"""Inspect a persistent depthless bearing once from a transversely displaced viewpoint."""
import math

from sparx_agency.core.planning.object_search.closing.local_view import LocalView


class BearingFollowup(LocalView):
    def __init__(self, closing, obs):
        super().__init__(obs, closing.floor_id, "bearing_followup", max_walk=1.25, max_actions=12)
        self.yaw = closing.bearing["yaw"]
        self.left = (-math.sin(self.yaw), math.cos(self.yaw))

    def displaced(self, xy):
        return abs(sum((p - o) * d for p, o, d in zip(xy, self.origin, self.left))) >= 0.30

    def command(self, closing, obs, world):
        if self.goal is None:
            points = [(self.origin[0] + k * self.left[0], self.origin[1] + k * self.left[1])
                      for k in (0.65, -0.65, 0.9, -0.9)]
            self.choose(closing, obs, world, points, self.displaced)
        return self.advance(closing, obs, world, self.yaw, self.displaced)
