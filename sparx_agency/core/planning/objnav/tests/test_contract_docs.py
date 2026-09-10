"""The contract docstrings promise only what the code does.

An adapter or method author reads these docstrings as the specification, so a
false promise in one is a bug no behavioural test catches: per-step info said
to be logged by the harness (it is returned to the caller of ``act()`` and
never stored), episode ids with no word about uniqueness (Habitat's repeat
across scenes), a no-dithering claim that holds only under exact execution.
Each promise that was wrong is pinned here in its corrected form, with the
docstring's line breaks collapsed.

Python 3.8 syntax; numpy arrives only through the modules under test.
"""
from __future__ import annotations

import pytest

import sparx_agency.core.planning.objnav.action_converter.action_choice as action_choice
import sparx_agency.core.planning.objnav.action_converter.ladder as ladder
import sparx_agency.core.planning.objnav.action_converter.types as converter_types
import sparx_agency.core.planning.objnav.agent.episode_log as episode_log
from sparx_agency.core.planning.objnav.interfaces.env import ObjNavEnv
from sparx_agency.core.planning.objnav.types.command import NavigationCommand
from sparx_agency.core.planning.objnav.types.decision import AgentDecision

#: Where per-step information is documented. The harness never stores it.
PER_STEP_DOCS = {
    "AgentDecision": AgentDecision.__doc__,
    "NavigationCommand": NavigationCommand.__doc__,
    "action_converter.types": converter_types.__doc__,
    "agent.episode_log": episode_log.__doc__,
}


def text(doc) -> str:
    """A docstring with its line breaks and indentation collapsed to single spaces."""
    return " ".join((doc or "").split())


@pytest.mark.parametrize("where", sorted(PER_STEP_DOCS))
def test_per_step_info_is_documented_as_returned_to_the_caller_not_logged(where):
    """The runner keeps only episode_info(); a promised per-step log sends a debugger to a file that does not exist."""
    doc = text(PER_STEP_DOCS[where])
    for promise in ("Logged by the harness", "Logged, never interpreted",
                    "read back from the log"):
        assert promise not in doc, where
    assert "caller of ``act()``" in doc, where
    assert "episode_info()" in doc, where


def test_episode_ids_are_documented_as_unique_within_benchmark_and_split():
    """Habitat ObjectNav numbers episodes from 0 in every scene; passed through raw, reset(id) is ambiguous."""
    doc = text(ObjNavEnv.episode_ids.__doc__)
    assert "unique within ``(benchmark, split)``" in doc
    assert "Scene-qualify" in doc and "Habitat" in doc


@pytest.mark.parametrize("module", [ladder, action_choice],
                         ids=["ladder", "action_choice"])
def test_the_no_dithering_claim_is_qualified_by_exact_execution(module):
    """AI2-THOR's turn noise does undo an overshooting turn now and then; the docs must not deny it."""
    doc = text(module.__doc__)
    assert "under exact execution" in doc
    assert "turn noise" in doc and "never a livelock" in doc
