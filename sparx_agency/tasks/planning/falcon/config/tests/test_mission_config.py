"""Tests for the object mission's YAML config reader.

The contract locked in here:
  * every section/key of the real, shipped ``mission.yaml`` is valid, and its
    ``launch:`` keys are all args the real ``object_mission.launch`` declares
    (so the shipped config cannot rot away from the launch file);
  * an unknown section / key / launch arg is a HARD ERROR, never a silent no-op --
    a typo must not leave a parameter quietly at its default;
  * YAML booleans reach roslaunch as ``true``/``false``, not Python's ``True``;
  * the emitted shell is quoted, so a path with a space cannot split.
"""
import pathlib
import subprocess
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve()
_CONFIG_DIR = _HERE.parents[1]
_FALCON = _CONFIG_DIR.parent
_LAUNCH = _FALCON / "adapter" / "launch" / "object_mission.launch"
_SHIPPED = _CONFIG_DIR / "mission.yaml"

sys.path.insert(0, str(_CONFIG_DIR))
import mission_config as mc  # noqa: E402


# ── The shipped config must stay valid against the real launch file ──────────
def test_shipped_config_loads_against_the_real_launch_file():
    """Assert the SHAPE, never the tuned values: map/nav_mode/model are the operator's
    to change, and a test that pins them just breaks every time the config is tuned."""
    variables, launch_args = mc.load(_SHIPPED, _LAUNCH)
    assert {"MAP", "NAV_MODE", "MODEL", "OBJECTS_DIR"} <= set(variables)
    assert variables["NAV_MODE"] in {"fallback", "hybrid", "astar", "combination", "navdp"}
    assert variables["MODEL"] in {"s", "m", "l", "x"}
    assert launch_args, "shipped config should set some launch args"


def test_shipped_config_launch_keys_all_exist_in_the_launch_file():
    declared = mc.launch_arg_names(_LAUNCH)
    _, launch_args = mc.load(_SHIPPED, _LAUNCH)
    for arg in launch_args:
        assert arg.split(":=")[0] in declared


def test_launch_arg_names_finds_real_args():
    names = mc.launch_arg_names(_LAUNCH)
    assert {"nav_mode", "land_range_m", "arrive_radius_m", "viewer"} <= names


# ── Unknown things are hard errors, with a suggestion ────────────────────────
def test_unknown_section_raises():
    with pytest.raises(mc.ConfigError, match="unknown section"):
        mc.parse({"missionn": {"map": "office"}})


def test_unknown_key_raises_and_suggests():
    with pytest.raises(mc.ConfigError) as e:
        mc.parse({"mission": {"nav_mod": "astar"}})
    assert "nav_mode" in str(e.value)          # the suggestion


def test_unknown_launch_arg_raises():
    with pytest.raises(mc.ConfigError, match="not an arg"):
        mc.parse({"launch": {"land_rang_m": 1.0}}, {"land_range_m", "viewer"})


def test_launch_args_unchecked_when_no_launch_file_given():
    _, args = mc.parse({"launch": {"anything_at_all": 1}})
    assert args == ["anything_at_all:=1"]


def test_non_mapping_section_raises():
    with pytest.raises(mc.ConfigError, match="must be a mapping"):
        mc.parse({"mission": ["map"]})


def test_non_scalar_value_raises():
    with pytest.raises(mc.ConfigError, match="single value"):
        mc.parse({"mission": {"map": ["office", "hospital"]}})


def test_missing_file_raises():
    with pytest.raises(mc.ConfigError, match="cannot read config"):
        mc.load(_CONFIG_DIR / "no_such_config.yaml")


def test_invalid_yaml_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("mission: {map: office\n")     # unbalanced brace
    with pytest.raises(mc.ConfigError, match="not valid YAML"):
        mc.load(bad)


# ── Empty / absent config is legal: everything falls back to built-ins ───────
def test_empty_config_sets_nothing():
    assert mc.parse(None) == ({}, [])


def test_empty_section_sets_nothing():
    assert mc.parse({"mission": None}) == ({}, [])


# ── Value rendering ─────────────────────────────────────────────────────────
def test_yaml_booleans_render_the_way_roslaunch_expects():
    _, args = mc.parse({"launch": {"viewer": True, "publish_overlay": False}},
                       {"viewer", "publish_overlay"})
    assert "viewer:=true" in args and "publish_overlay:=false" in args


def test_numbers_render_without_quotes_in_the_arg():
    _, args = mc.parse({"launch": {"land_range_m": 1.0, "n_confirm": 3}},
                       {"land_range_m", "n_confirm"})
    assert set(args) == {"land_range_m:=1.0", "n_confirm:=3"}


def test_negative_seed_survives():
    variables, _ = mc.parse({"mission": {"seed": -1}})
    assert variables["SEED"] == "-1"


# ── Emitted shell ───────────────────────────────────────────────────────────
def test_emit_quotes_paths_with_spaces():
    out = mc.emit(*mc.parse({"paths": {"objects_dir": "/a dir/with spaces"}}))
    assert "MISSION_CFG_OBJECTS_DIR=" in out
    # Round-trip through the shell: the value must arrive as ONE word.
    got = subprocess.run(
        ["bash", "-c", '%s\nprintf "%%s" "$MISSION_CFG_OBJECTS_DIR"' % out],
        capture_output=True, text=True, check=True).stdout
    assert got == "/a dir/with spaces"


def test_emit_always_defines_the_launch_args_array():
    out = mc.emit(*mc.parse({}))
    assert "MISSION_CFG_LAUNCH_ARGS=()" in out


def test_emitted_launch_array_round_trips_through_bash():
    out = mc.emit(*mc.parse({"launch": {"viewer": False, "land_range_m": 0}},
                            {"viewer", "land_range_m"}))
    got = subprocess.run(
        ["bash", "-c", '%s\nprintf "%%s\\n" "${MISSION_CFG_LAUNCH_ARGS[@]}"' % out],
        capture_output=True, text=True, check=True).stdout.split()
    assert sorted(got) == ["land_range_m:=0", "viewer:=false"]


# ── CLI ─────────────────────────────────────────────────────────────────────
def test_cli_emits_shell_for_the_shipped_config(capsys):
    assert mc.main([str(_SHIPPED), "--launch-file", str(_LAUNCH)]) == 0
    out = capsys.readouterr().out
    assert "MISSION_CFG_NAV_MODE=" in out and "MISSION_CFG_LAUNCH_ARGS=(" in out


def test_cli_returns_nonzero_on_a_bad_key(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("mission:\n  nav_mod: astar\n")
    assert mc.main([str(bad)]) == 1
    assert "[ERROR]" in capsys.readouterr().err
