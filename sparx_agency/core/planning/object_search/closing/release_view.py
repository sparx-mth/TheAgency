"""A failed approach can buy one other viewing angle, never an unverified STOP."""
import math

from sparx_agency.core.planning.object_search.closing.local_view import LocalView, angle_change, ring_views


class ReleaseView(LocalView):
    def __init__(self, closing, obs, witness):
        super().__init__(obs, closing.floor_id, "approach_release_retry", max_walk=2.0, max_actions=16)
        self.sighting = tuple(closing.observed_xyz)
        self.focus = self.sighting[:2]
        self.witness = witness
        self.label = closing.label

    def displaced(self, xy):
        return (math.dist(self.origin, xy) >= 0.35 and math.dist(self.witness, xy) >= 0.45
                and angle_change(self.focus, self.witness, xy) >= math.radians(30))

    def command(self, closing, obs, world):
        if self.goal is None:
            radius = min(math.dist(self.witness, self.focus), closing.settings.terminal_distance_m + 0.5)
            points = ring_views(self.witness, self.focus, radius, (50, -50, 75, -75, 100, -100))
            self.choose(closing, obs, world, points, self.displaced, self.focus)
        yaw = math.atan2(self.focus[1] - obs.pose.y, self.focus[0] - obs.pose.x)
        return self.advance(closing, obs, world, yaw, self.displaced)

    def new_evidence(self, obs, floor_id, label, xyz, association_m):
        here = (obs.pose.x, obs.pose.y)
        return (self.finished is None and self.goal is not None and floor_id == self.floor_id
                and obs.step - self.start < self.max_actions and label == self.label
                and self.displaced(here) and math.dist(here, self.goal) <= 0.3
                and math.dist(xyz, self.sighting) <= association_m)
