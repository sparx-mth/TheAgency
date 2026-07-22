"""Roll-assisted waypoint follower (waypoint navigation + cross-track ROLL).

Keeps the one-axis ``waypoint_follower`` fully in charge of navigation and only
adds a lateral ("ROLL", ``+vy`` = left) velocity that pulls the drone back onto
its trajectory when it drifts sideways — full while advancing, weak while
turning, small while holding. Drop-in compatible public API.
"""
from .params import CrossTrackRollParams
from .corrector import CrossTrackRollCorrector
from .follower import RollAssistFollower

__all__ = [
    "CrossTrackRollParams",
    "CrossTrackRollCorrector",
    "RollAssistFollower",
]
