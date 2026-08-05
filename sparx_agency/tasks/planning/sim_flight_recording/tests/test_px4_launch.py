"""Per-instance PX4 identity: the thing that has to be right before running two.

Two PX4 instances that share a port, a lock file or -- worst and least
obviously -- a working directory will corrupt each other. These check the
derivations, not the launching.
"""
from __future__ import annotations

import pytest

from sparx_agency.tasks.planning.sim_flight_recording import px4_launch


def test_ports_are_distinct_per_instance():
    offboard = [px4_launch.offboard_port(i) for i in range(px4_launch.MAX_INSTANCES)]
    hil = [px4_launch.hil_port(i) for i in range(px4_launch.MAX_INSTANCES)]
    assert len(set(offboard)) == px4_launch.MAX_INSTANCES
    assert len(set(hil)) == px4_launch.MAX_INSTANCES
    assert not set(offboard) & set(hil)


def test_ports_match_px4s_own_formulas():
    assert px4_launch.offboard_port(0) == 14540
    assert px4_launch.offboard_port(7) == 14547
    assert px4_launch.hil_port(0) == 4560
    assert px4_launch.hil_port(7) == 4567


def test_instances_past_px4s_ceiling_are_rejected():
    """PX4 gives every instance from 10 up the same offboard port, so they collide."""
    with pytest.raises(ValueError, match="could not be told apart"):
        px4_launch.offboard_port(px4_launch.MAX_INSTANCES)
    with pytest.raises(ValueError):
        px4_launch.hil_port(-1)


def test_each_instance_gets_its_own_working_directory(tmp_path):
    directories = {px4_launch.working_dir(tmp_path, i) for i in range(4)}
    assert len(directories) == 4
    for directory in directories:
        assert directory.parent.name == "px4_sitl_default"


def test_clearing_saved_parameters_only_touches_that_instance(tmp_path):
    for instance in (0, 1):
        directory = px4_launch.working_dir(tmp_path, instance)
        directory.mkdir(parents=True)
        (directory / "parameters.bson").write_bytes(b"x")
        (directory / "parameters_backup.bson").write_bytes(b"x")

    px4_launch.clear_saved_parameters(tmp_path, 0)

    assert not (px4_launch.working_dir(tmp_path, 0) / "parameters.bson").exists()
    assert not (px4_launch.working_dir(tmp_path, 0) / "parameters_backup.bson").exists()
    assert (px4_launch.working_dir(tmp_path, 1) / "parameters.bson").exists()


def test_clearing_saved_parameters_is_fine_when_there_are_none(tmp_path):
    px4_launch.clear_saved_parameters(tmp_path, 3)  # must not raise


def test_launching_without_a_build_says_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="make px4_sitl_default none"):
        px4_launch.launch_px4(tmp_path, instance=0)


def test_terminating_nothing_is_a_no_op():
    px4_launch.terminate_px4(None, instance=0)
