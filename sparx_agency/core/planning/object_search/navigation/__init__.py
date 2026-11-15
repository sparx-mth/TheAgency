"""Narrow-passage navigation: the kinematic follower, the bottleneck geometry, the scrape guard and the look-down.

See ``README.md`` beside this file.
"""
from .bottlenecks import Channel, heading_error, nearest_channel, route_channels, route_position
from .doorway_look_down import BottleneckLookDown
from .kinematic_follower import KinematicFollower
from .scrape_guard import DoorwayScrapeGuard
from .settings import KinematicSettings, LookDownSettings, ScrapeSettings

__all__ = ["BottleneckLookDown", "Channel", "DoorwayScrapeGuard", "KinematicFollower", "KinematicSettings",
           "LookDownSettings", "ScrapeSettings", "heading_error", "nearest_channel", "route_channels", "route_position"]
