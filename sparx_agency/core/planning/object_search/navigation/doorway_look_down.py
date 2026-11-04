"""The look-down at a bottleneck: the camera pitched on the approach, the entry re-planned on what it saw.

The request is read by the camera controller (``CameraController.apply``,
owner ``DOORWAY``): a 30-degree pitch keeps the room ahead in the top of
the image and puts the floor from half a metre out in the bottom, so the
frames on the way to a doorway map its jambs where the body will pass them.
The frame after the camera has pitched is the first one taken with that
view; the follower drops its plan on it, and the next plan through the
doorway is made on the refreshed map -- the pre-entry correction, one turn
at most, before a step is committed.
"""
from __future__ import annotations

import math

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.object_search.navigation.bottlenecks import route_channels, route_position

#: How many of the latest look-downs are kept for the record.
EVENT_MEMORY = 40
#: How many actions without a walk (a hold, a glance's turn, a re-plan) a request survives before it is released.
#: The smoke run of 2026-10-28 (00832/000074) released and re-requested one doorway 89 times -- 178 of the
#: episode's 500 actions spent on LOOK_DOWN and LOOK_UP -- when every scan turn between two walking steps
#: released it at once.
ROUTE_GONE_GRACE_ACTIONS = 2
#: The agent must be facing the throat to within this angle for the approach to count: a route that begins by
#: turning round is not approaching the doorway yet.
APPROACH_HALF_ANGLE_RAD = math.radians(90.0)
#: Two throats closer than this are the same doorway: a route re-planned every step moves its throat by a few
#: cells, and a channel matched by an exact key read as a new one every time it crossed a rounding boundary.
SAME_THROAT_M = 0.6


class BottleneckLookDown:
    """Ask for the pitch on the approach to the next channel of the route, and say when the view has refreshed.

    Args:
        policy: The search policy (its planner's clearance field and settings are read).
        settings: :class:`~sparx_agency.core.planning.object_search.navigation.settings.LookDownSettings`.
    """

    def __init__(self, policy, settings):
        self.policy, self.settings = policy, settings
        self._active = None                    # {"throat", "channel", "since", "gone_since" (steps without a walk), ...}
        self._done = []                        # throats whose look-down is over: never requested again
        self._last_pitch = None
        self._channels_for = None              # (route key) -> channels, the route in force
        self.events = []
        self.stats = {"requests": 0, "released_past_exit": 0, "released_route_gone": 0, "released_hold": 0,
                      "refreshes": 0}

    def _channels(self, route, world):
        key = tuple(route)
        if self._channels_for is None or self._channels_for[0] != key:
            p = self.policy
            fields = p.planner.fields_for(world)
            doors = p.doors.confirmed() if getattr(p, "doors", None) is not None else ()
            self._channels_for = (key, route_channels(route, fields.clearance, world.params.origin_x,
                                                      world.params.origin_y, world.resolution,
                                                      p.planner_params.channel_half_width_m, doors=doors))
        return self._channels_for[1]

    def pitch_request(self, observation, world, route):
        """The pitch to hold this step, degrees, or None for the level view.

        Args:
            observation: The current observation.
            world: The map in force.
            route: The route about to be walked (``(x, y)`` pairs), or None when the step is no walk.
        """
        if not self.settings.enabled:
            return None
        step = int(observation.step)
        active = self._active
        if route is None or len(route) < 2 or world is None:
            if active is None:
                return None
            active["gone_since"] = (active["gone_since"] or 0) + 1       # consecutive steps without a walk
            if active["gone_since"] >= ROUTE_GONE_GRACE_ACTIONS:
                return self._release("released_route_gone", step)
            return self.settings.pitch_deg
        channels = self._channels(route, world)
        pose = observation.pose
        s_here = route_position(route, (pose.x, pose.y))
        if active is not None:
            active["gone_since"] = None
            channel = next((c for c in channels if math.dist(c.throat_xy, active["throat"]) <= SAME_THROAT_M), None)
            if channel is None:
                return self._release("released_route_gone", step)
            if s_here > channel.exit_s_m + self.settings.exit_margin_m:
                return self._release("released_past_exit", step)
            if step - active["since"] >= self.settings.max_hold_actions:
                return self._release("released_hold", step)
            return self.settings.pitch_deg
        for channel in channels:
            if any(math.dist(channel.throat_xy, done) <= SAME_THROAT_M for done in self._done):
                continue
            if s_here > channel.exit_s_m + self.settings.exit_margin_m:
                continue
            if channel.entry_s_m - s_here > self.settings.approach_m:
                break                                   # channels are in route order: the next one is farther still
            bearing = math.atan2(channel.throat_xy[1] - pose.y, channel.throat_xy[0] - pose.x)
            if abs(normalize_angle(bearing - pose.yaw)) > APPROACH_HALF_ANGLE_RAD:
                break                                   # not facing it yet: the turn comes first, the look after
            self._active = {"throat": channel.throat_xy, "channel": channel.as_dict(), "since": step, "gone_since": None,
                            "pitched": False, "refreshed": False}
            self.stats["requests"] += 1
            self._event(step, "requested", channel.as_dict())
            return self.settings.pitch_deg
        return None

    def _release(self, why, step):
        """End the request: a released channel is done, whatever ended it -- a second look costs two more actions."""
        active, self._active = self._active, None
        if active is not None:
            self._done.append(active["throat"])
            self.stats[why] += 1
            self._event(step, why, active["channel"])
        return None

    def refreshed(self, observation):
        """True once per look-down: on the first frame taken after the camera reached the requested pitch."""
        pitch = float(observation.pose.camera_pitch)
        previous, self._last_pitch = self._last_pitch, pitch
        active = self._active
        if active is None or active["refreshed"]:
            return False
        target = math.radians(self.settings.pitch_deg)
        if abs(pitch - target) > math.radians(10.0):
            return False
        if previous is not None and abs(previous - pitch) < 1e-6 and not active["pitched"]:
            # The camera was already there before the request: the view did not change.
            active["pitched"] = active["refreshed"] = True
            return False
        active["pitched"] = active["refreshed"] = True
        self.stats["refreshes"] += 1
        self._event(int(observation.step), "refreshed", active["channel"])
        return True

    @property
    def active(self):
        return self._active is not None

    def _event(self, step, kind, channel):
        self.events.append({"step": int(step), "event": kind, "channel": channel})
        del self.events[:-EVENT_MEMORY]

    def diagnostics(self):
        return {"enabled": self.settings.enabled, "stats": dict(self.stats), "active": self._active,
                "events": list(self.events)}
