"""Turn ``config/airframe.yaml`` into the core control objects it describes.

The measured velocity response belongs to the *platform*, not to the controller.
Written as a constant inside a control module it would be silently wrong for
every other airframe that module ever flies, and there would be no honest place
to record where the numbers came from. Kept here it is data, versioned next to
the drone it was measured on, and
:class:`~sparx_agency.core.control.velocity_servo.plant.VelocityPlant` is built
from it rather than from a default.

That matters more on this platform than on most. ``libplugin_drone.so`` is a PID
cascade with a 0.181 s horizontal transport delay, and a velocity servo that
assumes the plant's stock defaults (0.18 s / 0.5 s / unity gain -- close, but
guessed) is inverting a plant it has not met.

Everything here reads a file next to the package; nothing takes an absolute
path. Every key is required -- a missing one raises rather than falling back to
the core defaults, because a plant that is half measured and half assumed is the
one failure mode that cannot be spotted in a flight log.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from sparx_agency.core.control.velocity_servo import AxisPlant, VelocityPlant
from sparx_agency.robots.SJTU.adapters.velocity_command import BodyVelocityLimits

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
AIRFRAME_CONFIG = CONFIG_DIR / "airframe.yaml"

_AXES = ("horizontal", "vertical", "yaw")


def load_airframe(path=None):
    # type: (Optional[Path]) -> Dict[str, Any]
    """Read ``config/airframe.yaml`` and hand back its raw contents.

    Args:
        path: Override the file, for a variant airframe or a test fixture.
            Defaults to :data:`AIRFRAME_CONFIG`.

    Returns:
        The parsed mapping.

    Raises:
        FileNotFoundError: If the file is missing, which on this platform means
            the package was copied without its config rather than that a default
            is wanted.
        ValueError: If the file does not parse to a mapping.
    """
    config_path = Path(path) if path is not None else AIRFRAME_CONFIG
    if not config_path.is_file():
        raise FileNotFoundError("SJTU airframe config not found: %s" % (config_path,))
    data = yaml.safe_load(config_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("%s must contain a mapping, got %s" % (config_path, type(data).__name__))
    return data


def velocity_plant(path=None):
    # type: (Optional[Path]) -> VelocityPlant
    """Build the measured :class:`VelocityPlant` for the SJTU sim drone.

    Args:
        path: Override the config file.

    Returns:
        The three-axis first-order-plus-delay model an outer loop inverts.

    Raises:
        KeyError: If any axis or any of its three numbers is missing.
    """
    plant = _section(load_airframe(path), "velocity_plant")
    axes = {}
    for axis in _AXES:
        axes[axis] = _axis_plant(_section(plant, axis, "velocity_plant"), axis)
    return VelocityPlant(horizontal=axes["horizontal"], vertical=axes["vertical"],
                         yaw=axes["yaw"])


def body_velocity_limits(path=None):
    # type: (Optional[Path]) -> BodyVelocityLimits
    """Build the command clamp from the plugin's own saturations.

    Args:
        path: Override the config file.

    Returns:
        The ceilings
        :func:`~sparx_agency.robots.SJTU.adapters.velocity_command.twist_fields`
        clamps against.

    Raises:
        KeyError: If any limit is missing.
    """
    limits = _section(load_airframe(path), "limits")
    return BodyVelocityLimits(
        max_speed_xy=_number(limits, "max_speed_xy", "limits"),
        max_speed_z=_number(limits, "max_speed_z", "limits"),
        max_yaw_rate=_number(limits, "max_yaw_rate", "limits"),
    )


def _axis_plant(axis_config, axis):
    # type: (Dict[str, Any], str) -> AxisPlant
    """One axis of the plant, with every number required.

    ``AxisPlant`` has defaults for all three fields, which is exactly why they
    are spelled out here: an axis missing ``delay_s`` would otherwise silently
    become a generic quadrotor's 0.15 s and the flight would merely be a little
    worse rather than a stated error.
    """
    where = "velocity_plant.%s" % (axis,)
    return AxisPlant(
        dc_gain=_number(axis_config, "dc_gain", where),
        time_constant_s=_number(axis_config, "time_constant_s", where),
        delay_s=_number(axis_config, "delay_s", where),
    )


def _section(config, key, where="airframe.yaml"):
    # type: (Dict[str, Any], str, str) -> Dict[str, Any]
    """Fetch a required sub-mapping, naming the file when it is absent."""
    if key not in config:
        raise KeyError("%s is missing '%s'" % (where, key))
    section = config[key]
    if not isinstance(section, dict):
        raise ValueError("%s.%s must be a mapping, got %s"
                         % (where, key, type(section).__name__))
    return section


def _number(config, key, where):
    # type: (Dict[str, Any], str, str) -> float
    """Fetch a required numeric value, naming where it should have been."""
    if key not in config:
        raise KeyError("%s is missing '%s'" % (where, key))
    return float(config[key])
