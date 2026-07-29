"""Pegasus's PX4 sensor suite with the simulated noise turned off.

Pegasus models a real GPS receiver, IMU, magnetometer and barometer, noise and
biases included. Outdoors that is the right model. Indoors it is the difference
between a data-collection campaign that works and one that does not: PX4's
estimator wanders metres around each setpoint, which is more than the gap
between two office desks, and its attitude solution occasionally diverges
outright. A simulated flight has no reason to pay that cost -- the point of it
is a *clean* expert demonstration, not a study of sensor noise.

So every configurable noise term is set to zero, which turns PX4's estimator
input into ground truth. The correlation times are deliberately left alone: they
appear in divisors (``imu.py:111``/``:131``, ``magnetometer.py:118``,
``gps.py:130``) and zeroing them is a ``ZeroDivisionError``, not a quieter
sensor.

The barometer is the exception and needs a subclass -- see
:class:`NoiselessBarometer`.

Must run inside a live Isaac Sim process with the patched ``pegasus.simulator``
extension importable.
"""
from __future__ import annotations

import numpy as np

# Pegasus's own default is 250 Hz, i.e. one HIL_GPS per physics step -- 50x what
# any receiver produces, and past the ~90 Hz PX4's observation buffer can accept
# anyway (it drops the excess with "GPS data too fast"). 20 Hz is far more
# realistic while still giving the estimator a fix every few filter updates,
# which matters here because the barometer is out of the loop and GNSS is the
# only absolute reference left.
GPS_RATE_HZ = 20.0
SENSOR_RATE_HZ = 250.0


def _noiseless_barometer_class():
    """Build the :class:`NoiselessBarometer` type against the live Pegasus import.

    Defined lazily because subclassing requires ``pegasus.simulator``, which only
    exists inside a running Kit app -- and this module must stay importable
    outside one so the rest of the package can be read and tested.
    """
    from pegasus.simulator.logic.sensors import Barometer
    from pegasus.simulator.logic.sensors.sensor import Sensor

    class NoiselessBarometer(Barometer):
        """A barometer with Pegasus's hardcoded pressure noise removed.

        Unlike every other Pegasus sensor, ``Barometer.update`` has **no config
        key for its noise**: it unconditionally draws ~1 Pa of Gaussian
        pressure error, which at sea-level air density is 8.4 cm of altitude,
        injected on every update. That is the single largest error source in
        the default suite and it lands directly on the height PX4 holds. The
        only way to remove it is to override the method.

        Everything else -- the ISA pressure model, the ``sensor_type`` the PX4
        backend dispatches on -- is unchanged.
        """

        @Sensor.update_at_rate
        def update(self, state, dt: float):
            """Publish the exact ISA pressure and altitude for the current state."""
            if self._z_start is None:
                self._z_start = state.position[2]

            altitude_amsl = self._origin_alt + (state.position[2] - self._z_start)
            temperature_local = self._TEMPERATURE_MSL - self._LAPSE_RATE * altitude_amsl
            absolute_pressure = self._PRESSURE_MSL / np.power(
                self._TEMPERATURE_MSL / temperature_local, 5.2561,
            )
            self._state = {
                "absolute_pressure": absolute_pressure * 0.01,  # Pa -> hPa
                "pressure_altitude": altitude_amsl,
                "temperature": temperature_local + self._ABSOLUTE_ZERO_C,
            }
            return self._state

    return NoiselessBarometer


def noiseless_sensors(gps_rate_hz: float = GPS_RATE_HZ,
                      sensor_rate_hz: float = SENSOR_RATE_HZ) -> list:
    """The four sensors PX4 needs, with every simulated error source removed.

    Drop-in replacement for ``MultirotorConfig().sensors``. Each keeps its
    ``sensor_type`` string, which is what ``PX4MavlinkBackend.update_sensor``
    dispatches on, so PX4 sees the same message set it always did -- just
    without the noise.

    Args:
        gps_rate_hz: GPS update rate. A real receiver's rate, not Pegasus's
            250 Hz default.
        sensor_rate_hz: Update rate for the IMU, magnetometer and barometer.
            Should be at least the physics rate so PX4's lockstep clock is fed
            every step.

    Returns:
        ``[barometer, imu, magnetometer, gps]``.
    """
    from pegasus.simulator.logic.sensors import GPS, IMU, Magnetometer

    barometer_class = _noiseless_barometer_class()
    return [
        barometer_class({"update_rate": sensor_rate_hz, "drift_pa_per_sec": 0.0}),
        IMU({
            "update_rate": sensor_rate_hz,
            "gyroscope": {
                "noise_density": 0.0, "random_walk": 0.0,
                "bias_correlation_time": 1.0e3, "turn_on_bias_sigma": 0.0,
            },
            "accelerometer": {
                "noise_density": 0.0, "random_walk": 0.0,
                "bias_correlation_time": 300.0, "turn_on_bias_sigma": 0.0,
            },
        }),
        Magnetometer({
            "update_rate": sensor_rate_hz, "noise_density": 0.0,
            "random_walk": 0.0, "bias_correlation_time": 6.0e2,
        }),
        GPS({
            "update_rate": gps_rate_hz,
            "fix_type": 3, "sattelites_visible": 16,  # (sic) that is the upstream key
            "eph": 1, "epv": 1,
            "gps_xy_random_walk": 0.0, "gps_z_random_walk": 0.0,
            "gps_xy_noise_density": 0.0, "gps_z_noise_density": 0.0,
            "gps_vxy_noise_density": 0.0, "gps_vz_noise_density": 0.0,
            "gps_correlation_time": 60,  # a divisor -- must stay non-zero
        }),
    ]
