"""StairTraversal: once its destination plateau is confirmed it walks off the flight, and keeps doing so at that height."""
import math
from types import SimpleNamespace

from sparx_agency.core.localization.floor_atlas import MultiFloorParams
from sparx_agency.core.planning.object_search.multifloor.stair_traversal import StairTraversal
from sparx_agency.core.planning.objnav.types.pose import AgentPose


class Terrain:
    """The calls a traversal makes, with the plateau under the agent set by the test."""

    def __init__(self):
        self.plateau, self.exits, self.flight_routes = 6.0, 0, 0
        self.world = SimpleNamespace(world_to_grid=lambda x, y: (0, 0), in_bounds=lambda x, y: False)

    def plateau_area(self, pose):
        return self.plateau

    def stair_distance(self, pose):
        return 0.1                              # still beside the stairs: arrival not yet allowed

    def exit_path(self, pose):
        self.exits += 1
        return [(pose.x, pose.y, pose.z), (pose.x - 1.0, pose.y, pose.z)]

    def continuation(self, pose, direction, source_height, guide=None):
        self.flight_routes += 1
        return [(pose.x, pose.y, pose.z), (pose.x + 1.0, pose.y, pose.z - 0.5)]

    def steering_path(self, pose, path, action_spec, choose=None, straight=False):
        return list(path[:2])


def traversal():
    params = MultiFloorParams()
    atlas = SimpleNamespace(elevation_m=2.8, floors={}, begin_transition=lambda pose: None)
    policy = SimpleNamespace(mapping=SimpleNamespace(atlas=atlas),
                             episode=SimpleNamespace(action_spec=SimpleNamespace(turn_angle_rad=math.radians(30))))
    terrain = Terrain()
    coordinator = SimpleNamespace(policy=policy, terrain=terrain, params=params, floor_id=1, source_trail=[],
                                  events=[], active={"id": 3, "direction": -1, "path": []})
    obs = SimpleNamespace(pose=AgentPose(0.0, 0.0, 2.8, 0.0), step=0)
    return StairTraversal(coordinator, obs), terrain, params


def test_a_confirmed_destination_plateau_is_walked_off_even_after_the_level_walk_test_lapses():
    """00820/000053 (2026-11-03): CONFIRM_DESTINATION at the middle storey lapsed on a step taken on the way off,
    and the flight route took the agent down the next flight twice. The walk-off is latched at that height."""
    t, terrain, params = traversal()
    for step, x in enumerate((0.0, -0.2, -0.4, -0.6), start=1):          # level and translated on the storey at 0.0
        t.observe(SimpleNamespace(pose=AgentPose(x, 0.0, 0.0, 0.0), step=step))
    assert t.phase == "CONFIRM_DESTINATION" and t.exit_height == 0.0 and not t.arrival_allowed
    terrain.plateau = 1.0                                                   # the next step: the test lapses
    obs = SimpleNamespace(pose=AgentPose(-0.8, 0.0, 0.0, 0.0), step=5)
    t.observe(obs)
    assert t.phase == "TRAVERSE"
    t.plan(obs)
    assert terrain.exits == 1 and terrain.flight_routes == 0, "off the flight, not on down it"
    # Down off the plateau's height (a real step down) ends the latch: the flight is the way on.
    low = SimpleNamespace(pose=AgentPose(-0.8, 0.0, -0.5, 0.0), step=6)
    t.observe(low)
    t.plan(low)
    assert t.exit_height is None and terrain.flight_routes == 1
    assert params.floor_match_m < 0.5
