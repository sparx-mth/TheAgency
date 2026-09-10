"""The episode's two ends -- the episode an agent is given and the measurement the harness receives -- accept what they promise and refuse what they forbid.

A blank identifier merges result rows that are not the same, and a STOP flag
that contradicts its termination grants or denies success wrongly. Neither
crashes anything downstream -- they move SR and SPL -- so each documented
refusal of ``ObjNavEpisode`` and ``EpisodeMeasurement`` is exercised here,
with the error class the docstring names, and each documented acceptance too
(an infinite final distance is a real outcome, not an adapter bug).

Python 3.8 syntax; numpy arrives only through the types.
"""
from __future__ import annotations

from types import MappingProxyType

import pytest

from sparx_agency.core.planning.objnav.errors import (
    EnvContractError,
    ObjNavError,
)
from sparx_agency.core.planning.objnav.tests.helpers import (
    INF,
    NAN,
    camera,
    intrinsics,
)
from sparx_agency.core.planning.objnav.types.actions import (
    ALL_ACTIONS,
    DiscreteActionSpec,
)
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.measurement import (
    NATIVE_KEYS,
    TERMINATION_STEP_LIMIT,
    TERMINATION_STOP,
    TERMINATIONS,
    EpisodeMeasurement,
)


def episode(**overrides):
    """A valid episode, with any field replaced."""
    fields = dict(episode_id="ep0", scene_id="scene0", benchmark="fake",
                  split="test", target_category="chair", camera=camera(),
                  action_spec=DiscreteActionSpec(), max_steps=500)
    fields.update(overrides)
    return ObjNavEpisode(**fields)


# -- ObjNavEpisode --------------------------------------------------------------------

def test_a_valid_episode_gets_its_own_metadata():
    """A shared default dict would leak one episode's extras into the next."""
    first, second = episode(), episode()
    first.metadata["note"] = 1
    assert second.metadata == {}


@pytest.mark.parametrize("field", ["episode_id", "scene_id", "benchmark", "split",
                                   "target_category"])
@pytest.mark.parametrize("value", ["", None, 7])
def test_a_blank_episode_identifier_is_refused(field, value):
    """Results are keyed on these; a blank one merges rows that are not the same."""
    with pytest.raises(ObjNavError, match=field):
        episode(**{field: value})


@pytest.mark.parametrize("field", ["episode_id", "scene_id", "benchmark", "split",
                                   "target_category"])
@pytest.mark.parametrize("value", ["   ", "\t\n"], ids=["spaces", "tab-newline"])
def test_an_identifier_of_only_whitespace_is_refused(field, value):
    """Non-empty but blank: rows keyed on it merge, and the harness refused it only at the summary."""
    with pytest.raises(ObjNavError, match=field):
        episode(**{field: value})


@pytest.mark.parametrize("metadata", [None, [("floor", 1)], "floor 1"],
                         ids=["none", "pairs", "string"])
def test_metadata_that_is_not_a_mapping_is_refused(metadata):
    """The policy reads it as a dict; None or pairs would fail inside the policy, far from the adapter."""
    with pytest.raises(ObjNavError, match="metadata"):
        episode(metadata=metadata)


def test_metadata_is_stored_as_a_dict_copy_of_any_mapping():
    """The adapter may reuse its dict for the next episode; this one must not change with it."""
    source = {"floor": 1}
    run = episode(metadata=MappingProxyType(source))
    source["floor"] = 2
    assert type(run.metadata) is dict and run.metadata == {"floor": 1}


def test_a_camera_or_action_spec_of_the_wrong_type_is_refused():
    """The agent reads both before its first step; a wrong type fails late and far away."""
    with pytest.raises(ObjNavError, match="camera"):
        episode(camera=intrinsics())
    with pytest.raises(ObjNavError, match="action_spec"):
        episode(action_spec=ALL_ACTIONS)


@pytest.mark.parametrize("max_steps", [0, -1, True, 500.0])
def test_a_non_positive_or_non_integer_step_budget_is_refused(max_steps):
    """The budget ends the episode; ``True`` would silently allow one step."""
    with pytest.raises(ObjNavError, match="max_steps"):
        episode(max_steps=max_steps)


# -- EpisodeMeasurement ------------------------------------------------------------------------

def measurement(**overrides):
    """A successful STOP episode, with any field replaced."""
    fields = dict(success=True, stop_called=True, termination=TERMINATION_STOP,
                  steps=42, shortest_path_m=5.0, start_distance_to_goal_m=5.0,
                  final_distance_to_goal_m=0.05, path_length_m=6.0)
    fields.update(overrides)
    return EpisodeMeasurement(**fields)


def test_measurements_for_both_terminations_are_accepted():
    """STOP and the step budget are the only two ways an environment ends an episode."""
    assert TERMINATIONS == (TERMINATION_STOP, TERMINATION_STEP_LIMIT)
    assert measurement().termination == TERMINATION_STOP
    ran_out = measurement(success=False, stop_called=False,
                          termination=TERMINATION_STEP_LIMIT, steps=500)
    assert ran_out.native_metrics == {} and ran_out.info == {}


@pytest.mark.parametrize("stop_called,termination", [
    (True, TERMINATION_STEP_LIMIT), (False, TERMINATION_STOP)])
def test_a_stop_flag_that_contradicts_the_termination_is_refused(stop_called,
                                                                 termination):
    """Success needs STOP; a flag that disagrees with the ending grants or denies it wrongly."""
    with pytest.raises(EnvContractError, match="contradicts"):
        measurement(stop_called=stop_called, termination=termination)


def test_an_unknown_termination_is_refused():
    """The summary counts terminations by name; a new one would vanish from the table."""
    with pytest.raises(EnvContractError, match="termination"):
        measurement(termination="timeout")


@pytest.mark.parametrize("field", ["success", "stop_called"])
@pytest.mark.parametrize("value", [1, "true", None])
def test_a_flag_that_is_not_a_real_bool_is_refused(field, value):
    """``bool("false")`` is True; a string flag would credit a failure."""
    with pytest.raises(EnvContractError, match=field):
        measurement(**{field: value})


@pytest.mark.parametrize("steps", [-1, True, 42.0])
def test_a_negative_or_non_integer_step_count_is_refused(steps):
    """The runner cross-checks steps against the actions it sent."""
    with pytest.raises(EnvContractError, match="steps"):
        measurement(steps=steps)


@pytest.mark.parametrize("field", ["shortest_path_m", "start_distance_to_goal_m",
                                   "path_length_m"])
@pytest.mark.parametrize("value", [-0.1, NAN, INF, True])
def test_a_negative_or_non_finite_length_is_refused(field, value):
    """SPL divides by these; a NaN makes the mean NaN and an inf makes it zero."""
    with pytest.raises(EnvContractError, match=field):
        measurement(**{field: value})


def test_an_infinite_final_distance_is_accepted():
    """Ending where no goal is reachable is a real outcome, not an adapter bug."""
    ended = measurement(success=False, final_distance_to_goal_m=INF,
                        native_metrics={"distance_to_goal": INF})
    assert ended.final_distance_to_goal_m == INF


@pytest.mark.parametrize("value", [NAN, -0.1, True])
def test_a_nan_or_negative_final_distance_is_refused(value):
    """Only ``inf`` has a meaning beyond the finite distances."""
    with pytest.raises(EnvContractError, match="final_distance_to_goal_m"):
        measurement(final_distance_to_goal_m=value)


def test_every_canonical_native_key_is_accepted():
    """The harness cross-checks each of these against its own arithmetic."""
    native = {"success": 1.0, "spl": 0.8, "soft_spl": 0.9, "distance_to_goal": 0.05}
    assert set(measurement(native_metrics=native).native_metrics) == set(NATIVE_KEYS)


@pytest.mark.parametrize("native,message", [
    ({"softspl": 0.9}, "unknown key"), ({"SPL": 0.8}, "unknown key"),
    ({"spl": NAN}, "must be a number"), ({"success": True}, "must be a number"),
    ({"spl": "0.8"}, "must be a number"),
], ids=["habitat-softspl", "upper-case", "nan", "bool", "string"])
def test_an_unknown_native_key_or_a_non_number_is_refused(native, message):
    """A typo'd key would skip its cross-check silently; Habitat's ``softspl`` must be renamed."""
    with pytest.raises(EnvContractError, match=message):
        measurement(native_metrics=native)


@pytest.mark.parametrize("field", ["native_metrics", "info"])
@pytest.mark.parametrize("value", [None, [("spl", 1.0)]], ids=["none", "pairs"])
def test_native_metrics_or_info_that_is_not_a_mapping_is_refused(field, value):
    """None used to raise AttributeError, which never named the contract the adapter broke."""
    with pytest.raises(EnvContractError, match=field):
        measurement(**{field: value})


def test_native_metrics_and_info_are_stored_as_dict_copies():
    """An adapter reusing its dicts must not rewrite a measurement already scored."""
    source = {"spl": 0.8}
    ended = measurement(native_metrics=MappingProxyType(source),
                        info=MappingProxyType({"goal": "g1"}))
    source["spl"] = 0.1
    assert type(ended.native_metrics) is dict
    assert ended.native_metrics == {"spl": 0.8}
    assert type(ended.info) is dict and ended.info == {"goal": "g1"}
