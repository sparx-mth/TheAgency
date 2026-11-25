import unittest
from types import SimpleNamespace as NS

from sparx_agency.core.planning.object_search.closing.stop_view import uncertain


class StopEvidenceTest(unittest.TestCase):
    def test_large_centred_box_alone_is_not_a_reason_to_delay(self):
        obs = NS(step=10, pose=NS(x=0., y=0.), camera=NS(intrinsics=NS(cx=320, width=640)))
        closing = NS(last_seen=10, conf=.99, box=(0, 5, 620, 465), xyz=(1., 0., .5),
                     settings=NS(border_margin_px=8, stop_confidence=.8),
                     policy=NS(episode=NS(action_spec=NS(forward_step_m=.25))))
        self.assertFalse(uncertain(closing, obs, .95))
        self.assertTrue(uncertain(closing, obs, .5))
        closing.last_seen = 7
        self.assertTrue(uncertain(closing, obs, None))
