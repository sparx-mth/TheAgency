"""A discrete turn means a rotation, and the policy has to say so."""
import math

import numpy as np
import pytest

from sparx_agency.core.planning.vlas.internvla_n1 import geometry


class TestTurnDeltaRad:
    def test_left_is_positive_and_right_is_negative(self):
        assert geometry.turn_delta_rad(geometry.TURN_LEFT, 15.0) == pytest.approx(
            math.radians(15.0))
        assert geometry.turn_delta_rad(geometry.TURN_RIGHT, 15.0) == pytest.approx(
            math.radians(-15.0))

    @pytest.mark.parametrize("index", [geometry.STOP, geometry.FORWARD, -1, 5, 99])
    def test_everything_that_is_not_a_turn_is_none(self, index):
        # FORWARD especially: it is a translation, and answering it with a
        # rotation would leave the aircraft turning on the spot for ever while
        # the model asks it to advance.
        assert geometry.turn_delta_rad(index, 15.0) is None

    def test_it_agrees_with_the_bent_step_it_replaces(self):
        # The two renderings of the same action must not disagree about which
        # way round it is; a sign flip here mirrors every turn in the flight and
        # still looks like a working stack.
        for index in (geometry.TURN_LEFT, geometry.TURN_RIGHT):
            delta = geometry.turn_delta_rad(index, 15.0)
            step = geometry.trajectory_from_action(index, step_m=0.25, turn_deg=15.0)
            bearing = math.atan2(step[-1, 1], step[-1, 0])
            assert math.copysign(1.0, bearing) == math.copysign(1.0, delta)
            assert bearing == pytest.approx(delta, abs=1e-9)

    def test_the_angle_comes_from_the_caller_not_a_constant(self):
        assert geometry.turn_delta_rad(geometry.TURN_LEFT, 30.0) == pytest.approx(
            math.radians(30.0))


class TestPolicyReportsIt:
    """The metadata contract the SJTU runner branches on."""

    def _policy(self):
        from sparx_agency.core.planning.vlas.internvla_n1.policy import InternVLAN1Policy
        return InternVLAN1Policy.__new__(InternVLAN1Policy)

    def test_a_curve_reports_no_rotation(self):
        # A System-1 curve carries its own heading in column 2. Asking for a
        # rotation as well would turn one decision into two manoeuvres.
        raw = {"action": [{"action": [2],
                           "trajectory": [[0.0, 0.0], [0.5, 0.1], [1.0, 0.3]]}]}
        assert geometry.trajectory_from_response(raw) is not None

    def test_a_turn_response_carries_no_curve(self):
        raw = {"action": [{"action": [2], "trajectory": None}]}
        assert geometry.trajectory_from_response(raw) is None
