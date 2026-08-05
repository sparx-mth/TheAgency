"""The ground-truth pose PX4 is fed, and the loop that guarantees it keeps coming.

Two things can go wrong here and both are silent in flight: the frame conversion
(PX4 flies confidently in the wrong direction) and the cadence (EKF2 stops fusing
after a 400 ms gap and says nothing). Both are checked without MAVLink, Isaac or
Pegasus.
"""
from __future__ import annotations

import math

import pytest

from sparx_agency.tasks.planning.sim_flight_recording.px4_vision_pose import (
    VisionPoseSender, enu_flu_to_ned_frd, find_px4_backend,
)


class _State:
    def __init__(self, position=(0.0, 0.0, 0.0), attitude=(0.0, 0.0, 0.0, 1.0)):
        self.position = position
        self.attitude = attitude


class _Backend:
    """Stands in for Pegasus's PX4MavlinkBackend: only its clock is read."""

    def __init__(self, utime=0):
        self._current_utime = utime


class _Vehicle:
    def __init__(self, backends, state=None):
        self._backends = backends
        self.state = state or _State()


class _Link:
    """Records what would have gone on the wire."""

    def __init__(self):
        self.poses = []

    def send_vision_pose(self, north, east, down, roll, pitch, yaw, usec):
        self.poses.append((north, east, down, roll, pitch, yaw, usec))


# --- the frame conversion ----------------------------------------------------

def test_east_and_north_swap_and_up_becomes_down():
    north, east, down, _, _, _ = enu_flu_to_ned_frd((1.0, 2.0, 3.0), 0.0, 0.0, 0.0)
    assert (north, east, down) == (2.0, 1.0, -3.0)


def test_facing_east_in_enu_is_ninety_degrees_in_ned():
    """ENU yaw is CCW from +X (east); NED yaw is CW from north."""
    *_, yaw_ned = enu_flu_to_ned_frd((0.0, 0.0, 0.0), 0.0, 0.0, 0.0)
    assert yaw_ned == pytest.approx(math.pi / 2.0)


def test_facing_north_in_enu_is_zero_in_ned():
    *_, yaw_ned = enu_flu_to_ned_frd((0.0, 0.0, 0.0), 0.0, 0.0, math.pi / 2.0)
    assert yaw_ned == pytest.approx(0.0)


def test_pitch_flips_sign_and_roll_does_not():
    """FLU pitches about +y (left); FRD pitches about +y (right)."""
    _, _, _, roll, pitch, _ = enu_flu_to_ned_frd((0.0, 0.0, 0.0), 0.3, 0.2, 0.0)
    assert roll == pytest.approx(0.3)
    assert pitch == pytest.approx(-0.2)


# --- finding the backend ----------------------------------------------------

def test_the_backend_is_found_by_the_clock_the_sender_needs():
    backend = _Backend()
    assert find_px4_backend(_Vehicle([object(), backend])) is backend


def test_a_vehicle_with_no_px4_backend_reports_none():
    assert find_px4_backend(_Vehicle([object()])) is None
    assert find_px4_backend(_Vehicle([])) is None


def test_a_vehicle_with_no_px4_backend_is_refused_loudly():
    """Stamping a pose on a clock PX4 is not running on is the failure this
    class exists to prevent, so it is not something to degrade into."""
    with pytest.raises(ValueError, match="no PX4 MAVLink backend"):
        VisionPoseSender(_Link(), _Vehicle([]))


# --- the cadence ------------------------------------------------------------

def test_the_first_call_always_sends():
    link = _Link()
    sender = VisionPoseSender(link, _Vehicle([_Backend()]), rate_hz=10.0)
    assert sender.send(0.0) is True
    assert len(link.poses) == 1


def test_calls_inside_the_interval_are_dropped():
    link = _Link()
    sender = VisionPoseSender(link, _Vehicle([_Backend()]), rate_hz=10.0)
    sender.send(0.0)
    assert sender.send(0.05) is False
    assert len(link.poses) == 1


def test_the_configured_rate_is_honoured_in_simulated_time():
    link = _Link()
    sender = VisionPoseSender(link, _Vehicle([_Backend()]), rate_hz=50.0)
    for step in range(250):                      # 1 s at the 250 Hz physics rate
        sender.send(step * (1.0 / 250.0))
    assert len(link.poses) == pytest.approx(50, abs=1)


def test_the_default_rate_stays_well_inside_ekf2s_fusion_timeout():
    """EV_MAX_INTERVAL is 200 ms to start fusing and 400 ms to stop."""
    from sparx_agency.tasks.planning.sim_flight_recording import px4_vision_pose

    assert 1.0 / px4_vision_pose.SEND_RATE_HZ < 0.1


def test_the_pose_carries_the_backends_lockstep_timestamp():
    """Not wall time and not the loop's own clock: the estimator drops a sample
    stamped in PX4's future."""
    link = _Link()
    backend = _Backend(utime=1_234_000)
    sender = VisionPoseSender(link, _Vehicle([backend]))
    sender.send(0.0)
    assert link.poses[0][-1] == 1_234_000


def test_the_pose_sent_is_the_vehicles_ground_truth():
    link = _Link()
    vehicle = _Vehicle([_Backend()], _State(position=(3.0, -4.0, 1.5)))
    VisionPoseSender(link, vehicle).send(0.0)
    north, east, down = link.poses[0][:3]
    assert (north, east, down) == (-4.0, 3.0, -1.5)


def test_the_sent_count_is_available_for_the_health_check():
    link = _Link()
    sender = VisionPoseSender(link, _Vehicle([_Backend()]), rate_hz=10.0)
    for step in range(30):
        sender.send(step * 0.1)
    assert sender.sent == len(link.poses) == 30


# --- the loop is what makes "every step" true -------------------------------

class _Recorder:
    """A stand-in sender that only counts, and remembers when it was called."""

    def __init__(self):
        self.times = []

    def send(self, sim_time):
        self.times.append(sim_time)
        return True


class _LoopVehicle(_Vehicle):
    """A vehicle whose physics-callback methods record the order they ran in."""

    def __init__(self, calls):
        super().__init__([_Backend()])
        self._sim_running = True
        self._calls = calls

    def update_state(self, dt):
        self._calls.append("state")

    def update_sensors(self, dt):
        self._calls.append("sensors")

    def update_sim_state(self, dt):
        self._calls.append("sim_state")

    def update(self, dt):
        self._calls.append("backend")


class _World:
    def step(self, render=False):
        pass


def _loop(vision, vehicle):
    from sparx_agency.tasks.planning.sim_flight_recording.sim_loop import SimLoop

    loop = SimLoop(_World(), vehicle, px4=None, rate_hz=10.0, vision=vision)
    loop.start()
    return loop


def test_the_loop_sends_a_pose_on_every_step():
    """No flight script has to remember to: a gap stops fusion silently."""
    vision, vehicle = _Recorder(), _LoopVehicle([])
    loop = _loop(vision, vehicle)
    for _ in range(5):
        loop.step()
    assert len(vision.times) == 5


def test_the_pose_is_stamped_with_the_time_the_step_reached():
    vision, vehicle = _Recorder(), _LoopVehicle([])
    loop = _loop(vision, vehicle)
    loop.step()
    assert vision.times == [pytest.approx(loop.dt)]


def test_the_pose_goes_out_after_the_backend_advanced_px4s_clock():
    """Sent before ``update()``, the pose is stamped a step ahead of PX4 and dropped."""
    calls = []
    vision = _Recorder()

    class _Ordered(_Recorder):
        def send(self, sim_time):
            calls.append("vision")
            return super().send(sim_time)

    vision = _Ordered()
    loop = _loop(vision, _LoopVehicle(calls))
    loop.step()
    assert calls.index("backend") < calls.index("vision")


def test_a_loop_with_no_sender_still_steps():
    loop = _loop(None, _LoopVehicle([]))
    loop.step()
    assert loop.step_index == 1
