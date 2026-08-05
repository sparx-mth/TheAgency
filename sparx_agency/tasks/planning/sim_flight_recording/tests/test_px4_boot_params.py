"""The pre-boot parameter channel, which is the only way to set a reboot-only one.

PX4 accepts a ``@reboot_required`` parameter over MAVLink, acknowledges it, saves
it and then ignores it -- so the difference between the two channels is invisible
from the outside and has to be enforced here. These check the split and the text
that gets written; nothing launches PX4.
"""
from __future__ import annotations

import pytest

from sparx_agency.tasks.planning.sim_flight_recording import px4_launch, px4_params


# --- which parameter goes on which channel -----------------------------------

def test_reboot_only_parameters_never_reach_the_mavlink_push():
    """The guard that makes the split enforceable rather than a convention."""
    for vision in (True, False):
        pushed = set(px4_params.all_params(vision))
        assert not pushed & set(px4_params.REBOOT_REQUIRED)


def test_every_reboot_only_parameter_vision_needs_is_in_the_boot_set():
    boot = set(px4_params.boot_params(vision=True))
    needed = set(px4_params.VISION_ESTIMATOR) & set(px4_params.REBOOT_REQUIRED)
    assert needed, "the vision set is supposed to contain reboot-only parameters"
    assert needed <= boot


def test_vision_turns_off_every_competing_aiding_source():
    """A second, worse opinion can only pull the estimate off the truth."""
    boot = px4_params.boot_params(vision=True)
    assert boot["EKF2_GPS_CTRL"] == 0
    assert boot["EKF2_BARO_CTRL"] == 0
    assert boot["EKF2_MAG_TYPE"] == 5      # None
    assert boot["SYS_HAS_MAG"] == 0
    assert boot["EKF2_HGT_REF"] == 3       # vision


def test_vision_fuses_position_and_yaw_but_not_velocity():
    """VISION_POSITION_ESTIMATE has no velocity field; asking for it stops fusion."""
    control = px4_params.boot_params(vision=True)["EKF2_EV_CTRL"]
    assert control & 1, "horizontal position"
    assert control & 2, "vertical position"
    assert control & 8, "yaw"
    assert not control & 4, "3D velocity -- the message cannot supply it"


def test_ev_quality_minimum_stays_zero():
    """PX4 never fills odom.quality for this message, so any minimum blocks fusion."""
    assert px4_params.boot_params(vision=True)["EKF2_EV_QMIN"] == 0


def test_vision_angle_noise_respects_the_parameters_own_floor():
    """EKF2_EVA_NOISE has @min 0.05; anything smaller is silently clamped up."""
    assert px4_params.boot_params(vision=True)["EKF2_EVA_NOISE"] >= 0.05
    assert px4_params.boot_params(vision=True)["EKF2_EVP_NOISE"] >= 0.01


def test_gps_aiding_is_configured_only_when_something_fuses_gps():
    assert "EKF2_GPS_P_NOISE" in px4_params.all_params(vision=False)
    assert "EKF2_GPS_P_NOISE" not in px4_params.all_params(vision=True)


def test_gps_flights_need_nothing_before_boot():
    assert px4_params.boot_params(vision=False) == {}


def test_the_flight_envelope_is_pushed_either_way():
    for vision in (True, False):
        assert px4_params.all_params(vision)["MPC_XY_VEL_MAX"] == 2.0


# --- how a value becomes a `param set` line ----------------------------------

def test_integers_are_written_without_a_decimal_point():
    """PX4 infers the type from the text, and rejects 1.0 for an INT32."""
    assert px4_launch.format_param_value(11) == "11"
    assert px4_launch.format_param_value(0) == "0"


def test_floats_always_keep_a_decimal_point():
    assert px4_launch.format_param_value(0.0) == "0.0"
    assert px4_launch.format_param_value(2.0) == "2.0"
    assert px4_launch.format_param_value(0.02) == "0.02"


def test_booleans_are_refused_rather_than_written_as_True():
    with pytest.raises(TypeError, match="no boolean parameter type"):
        px4_launch.format_param_value(True)


def test_a_string_is_refused():
    with pytest.raises(TypeError):
        px4_launch.format_param_value("5")


# --- the script itself -------------------------------------------------------

def test_the_script_sets_every_parameter_it_was_given(tmp_path):
    script = px4_launch.write_boot_parameters(
        tmp_path, 0, {"EKF2_EV_CTRL": 11, "EKF2_EVP_NOISE": 0.02})

    lines = script.read_text().splitlines()
    assert "param set EKF2_EV_CTRL 11" in lines
    assert "param set EKF2_EVP_NOISE 0.02" in lines


def test_the_script_lands_where_px4_will_look_for_it(tmp_path):
    """rcS resolves `. px4-rc.params` through PATH, and launch_px4 puts the
    instance's working directory first on it."""
    script = px4_launch.write_boot_parameters(tmp_path, 2, {"EKF2_EV_CTRL": 11})
    assert script.name == px4_launch.BOOT_PARAM_SCRIPT
    assert script.parent == px4_launch.working_dir(tmp_path, 2)


def test_the_script_is_a_shell_script_px4_can_source(tmp_path):
    script = px4_launch.write_boot_parameters(tmp_path, 0, {"SYS_HAS_MAG": 0})
    assert script.read_text().startswith("#!/bin/sh")


def test_each_instance_gets_its_own_script(tmp_path):
    px4_launch.write_boot_parameters(tmp_path, 0, {"EKF2_EV_CTRL": 11})
    px4_launch.write_boot_parameters(tmp_path, 1, {"EKF2_EV_CTRL": 0})

    first = px4_launch.working_dir(tmp_path, 0) / px4_launch.BOOT_PARAM_SCRIPT
    second = px4_launch.working_dir(tmp_path, 1) / px4_launch.BOOT_PARAM_SCRIPT
    assert "EKF2_EV_CTRL 11" in first.read_text()
    assert "EKF2_EV_CTRL 0" in second.read_text()


def test_writing_nothing_removes_a_script_from_an_earlier_run(tmp_path):
    """The same trap as PX4's persisted parameters.bson: a run that dropped the
    flag must not still be configured by it."""
    px4_launch.write_boot_parameters(tmp_path, 0, {"EKF2_EV_CTRL": 11})
    assert px4_launch.write_boot_parameters(tmp_path, 0, {}) is None
    assert not (px4_launch.working_dir(tmp_path, 0)
                / px4_launch.BOOT_PARAM_SCRIPT).exists()


def test_clearing_saved_parameters_also_clears_the_boot_script(tmp_path):
    px4_launch.write_boot_parameters(tmp_path, 0, {"EKF2_EV_CTRL": 11})
    px4_launch.clear_saved_parameters(tmp_path, 0)
    assert not (px4_launch.working_dir(tmp_path, 0)
                / px4_launch.BOOT_PARAM_SCRIPT).exists()


def test_a_bad_value_fails_before_px4_is_launched(tmp_path):
    with pytest.raises(TypeError):
        px4_launch.write_boot_parameters(tmp_path, 0, {"EKF2_EV_CTRL": "eleven"})


def test_the_real_vision_set_writes_cleanly(tmp_path):
    """Every value in the shipped set has to survive being turned into text."""
    script = px4_launch.write_boot_parameters(
        tmp_path, 0, px4_params.boot_params(vision=True))
    written = [line for line in script.read_text().splitlines()
               if line.startswith("param set ")]
    assert len(written) == len(px4_params.VISION_ESTIMATOR)
