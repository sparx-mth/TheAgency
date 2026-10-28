"""The scrape guard: a refused step inside a doorway, nearly aligned with it, is the jamb clipped -- not a wall.

HM3D moves the agent without sliding: any contact refuses the whole
MOVE_FORWARD, and a heading quantised to 30 degrees through a 0.8 m
doorway with a 0.36 m body has a few centimetres to spare. Until
2026-10-28 every refusal was written on the occupancy grid as the cell
ahead occupied, and a second one as a bar across the heading -- across the
doorway the body was walking through, which the planner then routed
around for the rest of the episode. This guard reads the refusal against
the route's channels (:mod:`.bottlenecks`): inside one, with the heading
within ``max_heading_deg`` of its axis, the refusal is a scrape and nothing
is written; the refused pose goes to the lattice instead, which re-plans the
entry from another heading or offset. A channel that refuses
``scrapes_before_mark`` alignments in ``window_actions`` is closed after all,
and its next refusal is written as before.
"""
from __future__ import annotations

import math

from sparx_agency.core.planning.object_search.navigation.bottlenecks import (heading_error, nearest_channel,
                                                                              route_channels)

#: How many of the latest verdicts are kept for the record.
EVENT_MEMORY = 40


class DoorwayScrapeGuard:
    """Decide whether a refused MOVE_FORWARD is written on the map.

    Args:
        policy: The search policy (its planner's clearance field, settings and door landmarks are read).
        settings: :class:`~sparx_agency.core.planning.object_search.navigation.settings.ScrapeSettings`.
    """

    def __init__(self, policy, settings):
        self.policy, self.settings = policy, settings
        self._counts = {}                       # channel key -> actions of the excused scrapes
        self.events = []
        self.stats = {"refusals": 0, "scrapes": 0, "written": 0, "exhausted": 0, "no_route": 0, "no_channel": 0,
                      "misaligned": 0}

    def judge(self, observation, world, route):
        """The scrape verdict for a refused step from ``observation.pose`` on ``route``, or None to write the map.

        Args:
            observation: The observation the refused step was decided on.
            world: The map in force (its clearance field names the channels).
            route: The route being walked, or None.
        """
        self.stats["refusals"] += 1
        if not self.settings.enabled:
            return None
        if route is None or world is None:
            self.stats["no_route"] += 1
            return None
        points = [(float(p.x), float(p.y)) if hasattr(p, "x") else (float(p[0]), float(p[1]))
                  for p in getattr(route, "points", route)]
        if len(points) < 2:
            self.stats["no_route"] += 1
            return None
        p = self.policy
        fields = p.planner.fields_for(world)
        doors = p.doors.confirmed() if getattr(p, "doors", None) is not None else ()
        channels = route_channels(points, fields.clearance, world.params.origin_x, world.params.origin_y,
                                  world.resolution, p.planner_params.channel_half_width_m, doors=doors,
                                  door_match_m=self.settings.door_match_m)
        pose = observation.pose
        step_m = p.episode.action_spec.forward_step_m
        ahead = (pose.x + step_m * math.cos(pose.yaw), pose.y + step_m * math.sin(pose.yaw))
        channel = nearest_channel(channels, ahead, self.settings.reach_m) or nearest_channel(
            channels, (pose.x, pose.y), self.settings.reach_m)
        if channel is None:
            self.stats["no_channel"] += 1
            return None
        error = heading_error(channel, pose.yaw)
        if error > math.radians(self.settings.max_heading_deg):
            self.stats["misaligned"] += 1
            self._event(observation.step, channel, error, "written: %.0f degrees off the axis" % math.degrees(error))
            return None
        step = int(observation.step)
        recent = [t for t in self._counts.get(channel.key(), ()) if step - t <= self.settings.window_actions]
        if len(recent) >= self.settings.scrapes_before_mark:
            self.stats["exhausted"] += 1
            self.stats["written"] += 1
            self._event(step, channel, error, "written: %d scrapes excused already" % len(recent))
            return None
        recent.append(step)
        self._counts[channel.key()] = recent
        self.stats["scrapes"] += 1
        verdict = dict(channel.as_dict(), heading_error_deg=round(math.degrees(error), 1), excused=len(recent),
                       step=step)
        self._event(step, channel, error, "scrape %d of %d excused" % (len(recent), self.settings.scrapes_before_mark))
        return verdict

    def _event(self, step, channel, error, detail):
        self.events.append({"step": int(step), "channel": channel.as_dict(),
                            "heading_error_deg": round(math.degrees(error), 1), "detail": detail})
        del self.events[:-EVENT_MEMORY]

    def diagnostics(self):
        return {"enabled": self.settings.enabled, "stats": dict(self.stats), "events": list(self.events)}
