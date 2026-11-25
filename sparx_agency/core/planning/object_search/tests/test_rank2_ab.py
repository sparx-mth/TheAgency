import unittest
from types import SimpleNamespace as NS

from sparx_agency.core.planning.object_search.closing.bearing_followup import BearingFollowup
from sparx_agency.core.planning.object_search.closing.target_closing import TargetClosing


class BearingTest(unittest.TestCase):
    def test_storey_change_or_second_abandon_never_arms_another_retry(self):
        obs = NS(step=8, pose=NS(x=0., y=0.))
        policy = NS(mapping=NS(floor_id=0), episode=NS(max_steps=500), route_memory=NS(kind=None))
        closing = NS(target_only=True, settings=NS(bearing_followup=True, confidence=.7),
                     bearing={"spot": (0., 0.), "yaw": 0., "conf": .9}, floor_id=0,
                     followup=None, bearing_abandoned=0, followups_started=0, policy=policy,
                     bearing_failures=[], _clear=lambda: None)
        TargetClosing._end_bearing(closing, obs, "no safe observed step on the bearing")
        self.assertIsInstance(closing.followup, BearingFollowup)
        self.assertFalse(closing.followup.displaced((.6, 0.)))
        self.assertTrue(closing.followup.displaced((0., .6)))
        policy.mapping.floor_id = 1
        TargetClosing._end_bearing(closing, obs, "storey changed")
        self.assertEqual((closing.followups_started, closing.bearing_abandoned), (1, 1))
