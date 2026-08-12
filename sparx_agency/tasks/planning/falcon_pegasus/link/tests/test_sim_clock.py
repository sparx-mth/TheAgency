"""The wall-clock-to-simulator map, and the failure it exists to prevent."""
import pytest

from sparx_agency.tasks.planning.falcon_pegasus.link.sim_clock import (
    MAX_FACTOR, MIN_FACTOR, SimClock,
)


def _run(clock, factor, seconds=4.0, step=0.004):
    """Tick both clocks for a while at a fixed real-time factor."""
    wall, sim = 1_700_000_000.0, 12.0
    ticks = int(seconds / step)
    for _ in range(ticks):
        wall += step
        sim += step * factor
        clock.update(wall, sim)
    return wall, sim


def test_the_factor_converges_on_a_simulator_running_slow():
    clock = SimClock()
    _run(clock, 0.66)
    assert clock.real_time_factor == pytest.approx(0.66, abs=0.01)


def test_a_real_time_simulator_leaves_the_factor_at_one():
    clock = SimClock()
    _run(clock, 1.0)
    assert clock.real_time_factor == pytest.approx(1.0, abs=0.01)


def test_now_converts_to_now():
    """The anchor is exact, whatever the factor: no drift accumulates at 'now'."""
    clock = SimClock()
    wall, sim = _run(clock, 0.5)
    assert clock.to_sim(wall) == pytest.approx(sim)


def test_a_future_instant_is_scaled_by_the_factor():
    clock = SimClock()
    wall, sim = _run(clock, 0.5)
    # Half a wall second ahead is a quarter of a simulated second ahead.
    assert clock.to_sim(wall + 0.5) == pytest.approx(sim + 0.25, abs=1e-3)


def test_a_trajectory_stamped_now_starts_now_however_slow_the_simulator():
    """The property the whole module exists for.

    A trajectory stamped at the current instant must begin at the current
    instant on the aircraft's clock. Handed the raw wall stamp instead, the
    tracker would compute an elapsed time of some 1.7 billion seconds and read
    every trajectory as finished long ago.
    """
    clock = SimClock()
    wall, sim = _run(clock, 0.3)
    assert clock.to_sim(wall) == pytest.approx(sim)
    assert abs(clock.to_sim(wall) - wall) > 1e8    # nothing like the wall stamp


def test_a_stalled_simulator_decays_toward_the_floor_but_never_below_it():
    """A frozen simulator must read as slow, and must stop at the floor.

    The previous version of this test could not fail: sim_step == 0 was skipped
    entirely, so the estimate never moved and asserting it stayed above
    MIN_FACTOR was vacuous. A stall is real information -- wall time is passing
    and no simulated time is -- so it now decays the estimate, and the clamp is
    what has to hold.
    """
    clock = SimClock(window_s=0.5)
    wall, sim = _run(clock, 1.0)
    assert clock.real_time_factor == pytest.approx(1.0, abs=0.05)
    for _ in range(10_000):                        # wall runs on, sim frozen
        wall += 0.004
        clock.update(wall, sim)
    assert clock.real_time_factor == pytest.approx(MIN_FACTOR)
    assert clock.real_time_factor >= MIN_FACTOR


def test_ticks_that_advanced_neither_clock_carry_no_information():
    """Polling faster than the physics steps must not read as a stall."""
    clock = SimClock()
    wall, sim = _run(clock, 0.8)
    settled = clock.real_time_factor
    for _ in range(500):
        clock.update(wall, sim)                    # the same instant, over and over
    assert clock.real_time_factor == pytest.approx(settled)


def test_the_factor_is_bounded_above():
    clock = SimClock()
    _run(clock, 5.0)
    assert clock.real_time_factor <= MAX_FACTOR


def test_converting_before_the_first_tick_raises():
    with pytest.raises(RuntimeError, match="before update"):
        SimClock().to_sim(1_700_000_000.0)


def test_a_non_positive_window_is_rejected():
    for bad in (0.0, -0.1):
        with pytest.raises(ValueError, match="window_s"):
            SimClock(window_s=bad)


def test_uneven_ticks_do_not_bias_the_factor():
    """Isaac's real cadence: 20 cheap physics ticks, then one expensive render.

    This is the bug the estimator was rewritten for. Averaging PER-TICK ratios
    weights the twenty cheap ticks the same as the one slow one and reports far
    too fast a simulator -- 0.75 against a true 0.61 on a real flight, with
    excursions above 1.0 on a machine that never reached real time. Summing
    simulated and wall time and dividing once weights each sample by its own
    wall step, which is the correct estimator.
    """
    clock = SimClock(window_s=2.0)
    wall, sim = 1_700_000_000.0, 0.0
    dt, cheap, render, every = 0.004, 0.0036, 0.0553, 21
    for step in range(20_000):
        wall += render if step % every == 0 else cheap
        sim += dt
        clock.update(wall, sim)
    true_rtf = (20_000 * dt) / (20_000 / every * render + 20_000 * (every - 1) / every * cheap)
    assert true_rtf == pytest.approx(0.66, abs=0.02)
    assert clock.real_time_factor == pytest.approx(true_rtf, rel=0.05)
    assert clock.real_time_factor < 1.0
