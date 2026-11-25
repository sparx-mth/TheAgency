"""One local visual cross-check for an ambiguous terminal observation."""
import math

from sparx_agency.core.planning.object_search.closing.local_view import LocalView, angle_change, ring_views


def uncertain(closing, obs, measured):
    if measured is None or closing.last_seen != obs.step:
        return True
    if closing.conf is not None and closing.conf < closing.settings.stop_confidence:
        return True
    if closing.box is None:
        return True
    k = obs.camera.intrinsics
    edge = closing.settings.border_margin_px
    clipped = closing.box[0] <= edge or closing.box[2] >= k.width - edge
    off_axis = not closing.box[0] <= k.cx <= closing.box[2]
    near_disagrees = (math.dist((obs.pose.x, obs.pose.y), closing.xyz[:2]) - measured
                      > closing.policy.episode.action_spec.forward_step_m)
    return near_disagrees and (clipped or off_axis)


class StopView(LocalView):
    def __init__(self, closing, obs):
        super().__init__(obs, closing.floor_id, "stop_reposition", max_walk=1.0, max_actions=9)
        self.focus = tuple(closing.xyz[:2])

    def displaced(self, xy):
        return math.dist(self.origin, xy) >= 0.20 and angle_change(self.focus, self.origin, xy) >= math.radians(15)

    def command(self, closing, obs, world):
        if self.goal is None:
            radius = math.dist(self.origin, self.focus)
            points = ring_views(self.origin, self.focus, radius, (35, -35, 50, -50))
            self.choose(closing, obs, world, points, self.displaced, self.focus)
        yaw = math.atan2(self.focus[1] - obs.pose.y, self.focus[0] - obs.pose.x)
        return self.advance(closing, obs, world, yaw, self.displaced)
