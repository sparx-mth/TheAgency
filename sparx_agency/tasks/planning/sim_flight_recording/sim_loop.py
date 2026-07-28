"""Advance the simulation by hand, one physics step at a time.

Everything in a PX4-in-the-loop flight hangs off this loop, and it is not
optional plumbing -- it is the fix for the bug that stopped any of this working.

Isaac Sim 6.0.1 stops dispatching ``World.add_physics_callback()``-registered
callbacks a couple of steps after ``world.reset()``, silently and with no
exception (see ``robots/PEGASUS/README.md`` for how that was confirmed). Pegasus
drives *everything* off those callbacks: the pose cache, the sensors, the
backend state push, and rotor force application -- which is also what sends
``HIL_SENSOR`` to PX4. PX4 is built with ``ENABLE_LOCKSTEP_SCHEDULER``, so with
no sensor data its clock never advances and it never even boots far enough to
emit a heartbeat.

:class:`~manual_physics_driver.ManualPhysicsDriver` calls those methods by hand.
This class is the loop that calls it, keeps the simulation clock, decides which
steps render, and polls the autopilot. Nothing here blocks: any blocking MAVLink
wait would stop the loop that produces the very data PX4 needs to answer it.
"""
from __future__ import annotations

import time

from sparx_agency.tasks.planning.sim_flight_recording import flight_session
from sparx_agency.tasks.planning.sim_flight_recording.manual_physics_driver import (
    ManualPhysicsDriver,
)


class SimLoop:
    """Steps the world, the vehicle and the autopilot together.

    Args:
        world: The Pegasus world.
        vehicle: The Pegasus ``Multirotor`` to drive.
        px4: A :class:`~px4_offboard.PX4Offboard` to poll each step, or None.
        rate_hz: Camera capture rate. Sets how often a step renders.
        dt: Physics timestep. Must match what ``world.step()`` actually advances
            -- see :mod:`flight_session`, which makes that a single fixed value.
        realtime: Throttle to wall-clock time so a live viewer can follow along.
            Off for collection: with warm GPU caches the simulation runs faster
            than real time, and there is no reason to wait.
        state_only: Drive only the vehicle's state cache, leaving force
            application to the caller. What a direct force-control script wants.
    """

    def __init__(self, world, vehicle, px4=None, rate_hz: float = 10.0,
                 dt: float = flight_session.PHYSICS_DT, realtime: bool = False,
                 state_only: bool = False):
        self.world = world
        self.vehicle = vehicle
        self.px4 = px4
        self.dt = dt
        self.realtime = realtime
        self.render_every = flight_session.render_every_n_steps(rate_hz, 1.0 / dt)
        self.driver = ManualPhysicsDriver(vehicle, state_only=state_only)
        self.sim_time = 0.0
        self.step_index = 0
        self._wall_start = None

    def start(self) -> None:
        """Run the simulation-start hooks the timeline callback may never fire."""
        self.driver.ensure_started()
        self._wall_start = time.monotonic()

    def set_realtime(self, enabled: bool) -> None:
        """Turn wall-clock pacing on or off mid-run, from *now*.

        Pacing is measured from a fixed origin, so simply flipping the flag on a
        loop that has already run is inert: simulated time is by then far behind
        ``_wall_start + sim_time``, and every subsequent step decides it is
        already late. Re-baselining is what makes a late switch mean anything.

        A run that must be paced usually should not pay for it during start-up --
        PX4's several minutes of warm-up are simulated seconds nobody watches,
        and there is no reason to spend real ones on them.

        Args:
            enabled: Whether following steps should be throttled to wall time.
        """
        self.realtime = bool(enabled)
        self._wall_start = time.monotonic() - self.sim_time

    def step(self) -> bool:
        """Advance one physics step.

        Returns:
            True if this step also rendered, and so produced a fresh camera
            frame.
        """
        render = self.step_index % self.render_every == 0
        self.world.step(render=render)
        self.driver.step(self.dt)
        if self.px4 is not None:
            self.px4.poll()
        self.step_index += 1
        self.sim_time += self.dt
        if self.realtime:
            self._pace()
        return render

    def run_for(self, seconds: float) -> None:
        """Step the simulation for ``seconds`` of simulated time, doing nothing else."""
        until = self.sim_time + seconds
        while self.sim_time < until:
            self.step()

    def warmup_camera(self, ticks: int = flight_session.CAMERA_WARMUP_RENDER_TICKS) -> None:
        """Step until the onboard camera has rendered enough to produce frames.

        ``MonocularCamera`` discards its first ~100 render callbacks, so
        ``capture_frame`` raises until this has run. Counted in *renders*, not
        steps, which is why it cannot just be a duration.
        """
        rendered = 0
        while rendered < ticks:
            if self.step():
                rendered += 1

    def _pace(self) -> None:
        """Sleep until wall-clock time has caught up with simulated time."""
        behind = self._wall_start + self.sim_time - time.monotonic()
        if behind > 0:
            time.sleep(behind)
