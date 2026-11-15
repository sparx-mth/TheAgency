"""Each clause of the environment contract, broken on purpose by a corridor built to break exactly that clause.

An adapter bug does not crash a benchmark; it produces a number. So every
check the runner makes must raise and name the clause that broke, before the
agent or the score sees the bad value. How the agent moved is its own clause,
pinned in ``test_kinematics.py``.
"""
from __future__ import annotations

import dataclasses
import re

import pytest

from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.tasks.planning.objnav_benchmark.errors import ScoringError
from sparx_agency.tasks.planning.objnav_benchmark.tests.corridor import (
    CorridorEnv,
    F,
    ScriptedAgent,
    one_episode,
)

ENV_FLAWS = [
    ("reset_wrong_id", "returned episode 'other'"),
    ("reset_step_one", "is step 0"),
    ("reset_wrong_target", "is for target 'bed'"),
    ("reset_wrong_camera", "another camera"),
    ("over_at_reset", "already over right after reset"),
    ("step_miscount", "step count must equal the actions sent"),
    ("step_wrong_target", "after action 1 .* target 'bed'"),
    ("step_wrong_camera", "after action 1 .* another camera"),
    ("stop_does_not_end", "still running after STOP"),
    ("ends_early", "ended after 2 actions without STOP"),
    ("measure_miscount", "counts 6 steps but the runner sent 5"),
    ("measure_wrong_stop", "stop_called=False, but its last action was STOP"),
]


class MetadataCorridor(CorridorEnv):
    """The corridor, its episode carrying ``metadata`` as an adapter might copy it from a dataset row."""

    def __init__(self, metadata):
        super().__init__({"e1": 1.0})
        self.metadata = metadata

    def reset(self, episode_id):
        episode, observation = super().reset(episode_id)
        return dataclasses.replace(episode, metadata=self.metadata), observation


@pytest.mark.parametrize("flaw,message", ENV_FLAWS, ids=[f for f, _ in ENV_FLAWS])
def test_each_broken_clause_of_the_environment_contract_names_itself(flaw, message):
    """Each would otherwise become a plausible number; each must raise and say which clause broke."""
    with pytest.raises(EnvContractError, match=message):
        one_episode(CorridorEnv({"e1": 1.0}, flaws=[flaw]))


def test_an_environment_that_runs_past_its_budget_gets_no_action_past_it():
    """Without the check an agent that never stops would run past what the benchmark allows."""
    env = CorridorEnv({"e1": 1.0}, max_steps=5, flaws=["runs_past_budget"])
    with pytest.raises(EnvContractError, match=r"after 5 actions.*max_steps=5"):
        one_episode(env, ScriptedAgent((F,)))
    assert env.steps == 5


def test_a_path_length_in_the_wrong_unit_fails_against_the_observed_poses():
    """The runner's own trace of p is what exposes an adapter reporting centimetres."""
    with pytest.raises(ScoringError, match="unit or accounting bug") as caught:
        one_episode(CorridorEnv({"e1": 1.0}, flaws=["path_in_cm"]))
    # A chord sum is blind to a rotated or mirrored frame; the message must not claim it.
    assert "Y-up" not in str(caught.value)
    record = one_episode(CorridorEnv({"e1": 1.0}, flaws=["path_in_cm"]),
                         path_tolerance_m=None)
    assert (record.path_length_m, record.observed_path_length_m) == (100.0, 1.0)


def test_a_native_metric_the_harness_computes_differently_fails_the_episode():
    """The simulator's own DTG disagreeing with ours is an adapter bug, not a result."""
    with pytest.raises(ScoringError, match="'distance_to_goal' disagrees"):
        one_episode(CorridorEnv({"e1": 1.0}, flaws=["wrong_native_distance"]))


@pytest.mark.parametrize("metadata,key", [
    ({"shortest_path_length": 5.57}, "metadata['shortest_path_length']"),
    ({"extras": [{"Goal_Position": [1.0, 2.0, 3.0]}]},
     "metadata['extras'][0]['Goal_Position']"),
    ({"task_info": {"distance_to_target": 3.0}},
     "metadata['task_info']['distance_to_target']"),
    ({"viewPoints": []}, "metadata['viewPoints']"),
    ({"geodesic_distance": 4.2}, "metadata['geodesic_distance']"),
], ids=["robothor_row", "nested_in_a_list", "allenact_task_info", "camel_case",
        "habitat_geodesic"])
def test_metadata_naming_privileged_information_is_refused_at_reset(metadata, key):
    """The agent receives the metadata at reset; a goal or a shortest path there makes SR and SPL meaningless."""
    with pytest.raises(EnvContractError,
                       match=re.escape(key) + ", a key that names privileged"):
        one_episode(MetadataCorridor(metadata))


def test_metadata_without_privileged_keys_passes_the_tripwire():
    """Public extras are what metadata is for; the tripwire must not refuse them."""
    record = one_episode(MetadataCorridor(
        {"floor_count": 2, "scene_dataset": "hm3d", "notes": ["x", {"lighting": "day"}]}))
    assert record.success
