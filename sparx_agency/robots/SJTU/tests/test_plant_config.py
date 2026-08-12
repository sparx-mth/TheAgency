"""``config/airframe.yaml`` really does load into the plant that was measured.

Two failure modes are worth a test each. The first is drift: a number edited in
the YAML that nothing notices, which produces a slightly wrong lead term and a
flight that is merely a bit worse. The second is silence: a *missing* number
falling back to ``AxisPlant``'s generic-quadrotor defaults, which is the same
outcome with no edit to point at.
"""
from __future__ import annotations

import pytest
import yaml

from sparx_agency.core.control.velocity_servo import VelocityPlant
from sparx_agency.robots.SJTU.adapters import plant_config

# The measured step responses, restated from the campaign rather than read back
# out of the file under test.
MEASURED = {
    "horizontal": (0.998, 0.181, 0.510),
    "vertical": (1.024, 0.033, 0.409),
    "yaw": (0.999, 0.055, 0.477),
}


def test_config_ships_with_the_package():
    """The YAML is found relative to this file, never by absolute path."""
    assert plant_config.AIRFRAME_CONFIG.is_file()
    assert plant_config.AIRFRAME_CONFIG.name == "airframe.yaml"
    assert plant_config.AIRFRAME_CONFIG.parent.name == "config"


@pytest.mark.parametrize("axis", sorted(MEASURED))
def test_axis_carries_the_measured_response(axis):
    """Each axis is the fitted first-order-plus-delay model, to the digit."""
    plant = plant_config.velocity_plant()
    dc_gain, delay_s, time_constant_s = MEASURED[axis]
    measured = getattr(plant, axis)

    assert measured.dc_gain == pytest.approx(dc_gain)
    assert measured.delay_s == pytest.approx(delay_s)
    assert measured.time_constant_s == pytest.approx(time_constant_s)


def test_plant_is_the_core_type():
    """What comes back is what a core velocity servo takes, not a lookalike."""
    assert isinstance(plant_config.velocity_plant(), VelocityPlant)


def test_no_axis_silently_keeps_a_core_default():
    """Every axis differs from ``VelocityPlant()``'s stock values.

    The defaults are a representative indoor quadrotor and are close enough to
    the SJTU drone to be mistaken for it. If a future edit drops a key, the load
    must raise -- see :func:`test_a_missing_number_raises` -- and this test is
    what makes that assertion meaningful, by confirming the two are actually
    distinguishable today.
    """
    measured, stock = plant_config.velocity_plant(), VelocityPlant()
    for axis in MEASURED:
        assert getattr(measured, axis) != getattr(stock, axis)


def test_horizontal_is_the_slow_axis():
    """The property the controller's bandwidth is set by.

    Horizontal goes through tilt, so it answers roughly five times later than
    vertical (0.181 s against 0.033 s). A controller tuned on one lag for all
    three axes is sluggish in two of them and aggressive in the third.
    """
    plant = plant_config.velocity_plant()
    assert plant.horizontal.delay_s > plant.vertical.delay_s
    assert plant.horizontal.delay_s > plant.yaw.delay_s
    assert plant.feedforward_lead_s == pytest.approx(plant.horizontal.delay_s)


def test_limits_are_the_plugin_saturations():
    """The clamp is configured from the platform, not typed into a control file."""
    limits = plant_config.body_velocity_limits()
    assert limits.max_speed_xy == pytest.approx(2.0)
    assert limits.max_speed_z == pytest.approx(2.0)
    assert limits.max_yaw_rate == pytest.approx(1.5)


def test_a_missing_number_raises(tmp_path):
    """A half-measured plant is an error, not a quietly defaulted one."""
    config = yaml.safe_load(plant_config.AIRFRAME_CONFIG.read_text())
    del config["velocity_plant"]["horizontal"]["delay_s"]
    path = tmp_path / "airframe.yaml"
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(KeyError):
        plant_config.velocity_plant(path)


def test_a_missing_axis_raises(tmp_path):
    """Same for a whole axis, which is the easier edit to make by accident."""
    config = yaml.safe_load(plant_config.AIRFRAME_CONFIG.read_text())
    del config["velocity_plant"]["yaw"]
    path = tmp_path / "airframe.yaml"
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(KeyError):
        plant_config.velocity_plant(path)


def test_a_missing_file_raises(tmp_path):
    """Naming a config that is not there must not fall back to the shipped one."""
    with pytest.raises(FileNotFoundError):
        plant_config.load_airframe(tmp_path / "nope.yaml")
