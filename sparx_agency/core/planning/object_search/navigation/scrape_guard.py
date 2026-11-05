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
entry from another heading or offset. A place that refuses
``scrapes_before_mark`` alignments in ``window_actions`` is closed after all:
its next refusal is written as before, and so is any refusal there for the
next ``closed_memory_actions``.

The count is by PLACE -- the throat, or the refused step's end point, within
``same_place_m`` -- not by channel (since 2026-11-05). Every refusal clears
the route, and the route re-planned from the same spot names a throat a few
centimetres over: keyed by the throat rounded to half a metre, the count
started afresh with every re-plan, and 00843/000007 was refused at one spot
at four headings 90 degrees apart, each excused as the first scrape of a new
channel -- a hole in the mesh the depth camera sees nothing of, so nothing
but the refusals could ever close it on the map.
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
        self._scrapes = []                      # (action, throat_xy, refused_xy) of every excused scrape
        self._closed = []                       # (action, throat_xy, refused_xy) of every place that ran out of excuses
        self.events = []
        self.stats = {"refusals": 0, "scrapes": 0, "written": 0, "exhausted": 0, "closed": 0, "no_route": 0,
                      "no_channel": 0, "misaligned": 0}

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
        throat = tuple(channel.throat_xy)
        self._forget(step)
        closed = [c for c in self._closed if self._same_place(c, throat, ahead)]
        if closed:
            self.stats["closed"] += 1
            self.stats["written"] += 1
            self._event(step, channel, error, "written: the place was closed at action %d" % closed[-1][0])
            return None
        recent = [s for s in self._scrapes if step - s[0] <= self.settings.window_actions
                  and self._same_place(s, throat, ahead)]
        if len(recent) >= self.settings.scrapes_before_mark:
            self._closed.append((step, throat, ahead))
            self.stats["exhausted"] += 1
            self.stats["written"] += 1
            self._event(step, channel, error, "written: %d scrapes excused already" % len(recent))
            return None
        self._scrapes.append((step, throat, ahead))
        self.stats["scrapes"] += 1
        verdict = dict(channel.as_dict(), heading_error_deg=round(math.degrees(error), 1), excused=len(recent) + 1,
                       step=step)
        self._event(step, channel, error, "scrape %d of %d excused" % (len(recent) + 1, self.settings.scrapes_before_mark))
        return verdict

    def _same_place(self, entry, throat, refused):
        """Whether a remembered scrape or closure is at this place: its throat near this throat, or its refused step near this one."""
        near = self.settings.same_place_m
        return math.dist(entry[1], throat) <= near or math.dist(entry[2], refused) <= near

    def _forget(self, step):
        """Drop the scrapes outside the window and the closures outside their memory (0 keeps them for the episode)."""
        self._scrapes = [s for s in self._scrapes if step - s[0] <= self.settings.window_actions]
        horizon = self.settings.closed_memory_actions
        if horizon > 0:
            self._closed = [c for c in self._closed if step - c[0] <= horizon]

    def _event(self, step, channel, error, detail):
        self.events.append({"step": int(step), "channel": channel.as_dict(),
                            "heading_error_deg": round(math.degrees(error), 1), "detail": detail})
        del self.events[:-EVENT_MEMORY]

    def diagnostics(self):
        return {"enabled": self.settings.enabled, "stats": dict(self.stats), "events": list(self.events)}
