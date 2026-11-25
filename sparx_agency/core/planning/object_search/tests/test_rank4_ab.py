import unittest
from types import SimpleNamespace as NS

from sparx_agency.core.planning.object_search.closing.release_view import ReleaseView


class ReleaseEvidenceTest(unittest.TestCase):
    def test_only_fresh_associated_geometry_from_another_angle_can_override_release(self):
        obs = NS(step=10, pose=NS(x=1., y=0.))
        retry = ReleaseView(NS(floor_id=1, label="plant", observed_xyz=(0., 0., .7)), obs, (1., 0.))
        retry.goal = (0., 1.)
        self.assertFalse(retry.new_evidence(obs, 1, "plant", (0., 0., .7), .4))
        obs.pose.x, obs.pose.y, obs.step = 0., 1., 14
        self.assertTrue(retry.new_evidence(obs, 1, "plant", (.1, 0., .7), .4))
        self.assertFalse(retry.new_evidence(obs, 0, "plant", (0., 0., .7), .4))
        self.assertFalse(retry.new_evidence(obs, 1, "plant", (1., 0., .7), .4))
        obs.step = 30
        self.assertFalse(retry.new_evidence(obs, 1, "plant", (0., 0., .7), .4))
