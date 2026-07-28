"""Drive a Pegasus vehicle's per-physics-step updates by hand.

Isaac Sim 6.0.1 stops dispatching ``World.add_physics_callback()``-registered
callbacks after ~2 calls following ``world.reset()`` (see
``robots/PEGASUS/README.md`` for how that was confirmed). Pegasus registers
*four* such callbacks per vehicle, and every one of them silently stops:

===========================  ================================================
callback                     what dies with it
===========================  ================================================
``update_state``             the Python-side pose/velocity cache goes stale
``update_sensors``           IMU/barometer/magnetometer/GPS stop producing
``update_sim_state``         backends stop being told the vehicle's state
``update``                   rotor forces stop being applied, **and**
                             ``backend.update()`` stops running -- which for
                             the PX4 backend is what sends ``HIL_SENSOR`` over
                             MAVLink and reads actuator commands back
===========================  ================================================

That last row is why the PX4 path never flew: PX4 SITL is built with
``ENABLE_LOCKSTEP_SCHEDULER``, so its entire internal clock is driven by
receiving sensor data from the simulator. No ``update`` callback means no
sensor data means PX4's clock never advances.

This driver sidesteps the broken dispatch entirely by calling those same four
methods, in the same order, once per ``world.step()`` from an ordinary Python
loop -- which does keep running reliably.
"""
from __future__ import annotations


class ManualPhysicsDriver:
    """Calls a Pegasus vehicle's physics-callback methods manually each step.

    Args:
        vehicle: A ``pegasus.simulator.logic.vehicles.Vehicle`` (in practice a
            ``Multirotor``).
        state_only: If True, only refresh the state cache
            (``Vehicle.update_state``) and leave force application and backend
            I/O to the caller. This is what a direct force-control script wants
            (see :mod:`fly_direct`); a PX4-in-the-loop script wants the full
            chain and must leave this False.
    """

    def __init__(self, vehicle, state_only: bool = False):
        self._vehicle = vehicle
        self._state_only = state_only

    def ensure_started(self) -> bool:
        """Run the simulation-start hooks if the timeline callback never fired.

        ``Vehicle.sim_start_stop`` -- registered via
        ``World.add_timeline_callback`` -- is what normally calls ``start()`` on
        every sensor and backend. For the PX4 backend that is what opens the
        MAVLink connection and sets ``_is_running``, without which
        ``PX4MavlinkBackend.update()`` returns immediately and no sensor data is
        ever sent. Timeline callbacks are a different dispatch path from physics
        callbacks, so this is belt-and-braces: it is a no-op when the timeline
        callback did fire.

        Returns:
            True if this call performed the start (i.e. the timeline callback
            had not fired), False if the vehicle was already running.
        """
        vehicle = self._vehicle
        if vehicle._sim_running:
            return False

        vehicle._sim_running = True
        for sensor in vehicle._sensors:
            sensor.start()
        for graphical_sensor in vehicle._graphical_sensors:
            graphical_sensor.start()
        for backend in vehicle._backends:
            backend.start()
        vehicle.start()
        return True

    def step(self, dt: float) -> None:
        """Run one physics step's worth of vehicle updates.

        The order matches the order Pegasus registers the callbacks in, which
        matters: sensors and backends both read the state cache that
        ``update_state`` refreshes, and ``update`` consumes the actuator
        commands that the backend's own ``update`` polls off the wire.

        Args:
            dt: Physics timestep, seconds. Must match the timestep
                ``world.step()`` actually advances, since PX4's lockstep clock
                is integrated from it.
        """
        vehicle = self._vehicle
        vehicle.update_state(dt)
        if self._state_only:
            return
        vehicle.update_sensors(dt)
        vehicle.update_sim_state(dt)
        vehicle.update(dt)
