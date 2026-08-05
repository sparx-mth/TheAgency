"""Finding recordings on disk, whatever shape the campaign left them in.

Every one of these layouts is produced by something in this repo, and getting
any of them wrong is silent: :func:`discover` returning fewer directories than
exist does not raise, it just trains on less data than the operator believes
they collected. The nested case is the one that matters most —
``campaign_supervisor.py`` cannot avoid producing it, because a relaunched
worker must not be given a directory that already holds flights.
"""
from __future__ import annotations

import numpy as np
import pytest

from sparx_agency.tasks.planning.vlas.navdp.finetune.world_goal.sources import discover


def _recording(directory):
    """The minimum on disk that makes a directory a recording."""
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "poses.npy", np.zeros((4, 4), dtype=np.float32))
    (directory / "rgb").mkdir(exist_ok=True)
    (directory / "depth").mkdir(exist_ok=True)
    return directory


@pytest.fixture()
def campaign(tmp_path):
    """One root holding all four layouts this repo can produce."""
    flat = tmp_path / "flat"                       # a hand-run collect.py
    _recording(flat / "office_w0_e000")
    _recording(flat / "office_w0_e001")

    nested = tmp_path / "nested"                   # campaign_supervisor.py
    _recording(nested / "w0_c001" / "office_w0_e000")
    _recording(nested / "w0_c002" / "office_w0_e000")
    _recording(nested / "w1_c001" / "office_w1_e000")

    falcon = tmp_path / "falcon"                   # a FALCON exploration run
    _recording(falcon / "run_a" / "recording")

    return tmp_path


def test_finds_recordings_directly_inside_a_campaign(campaign):
    """The layout a hand-run ``collect.py`` writes."""
    assert len(discover([str(campaign / "flat")])) == 2


def test_finds_recordings_nested_under_per_launch_directories(campaign):
    """The layout ``campaign_supervisor.py`` writes, and must not miss.

    Each worker launch gets its own directory so a recycled worker cannot
    overwrite its earlier flights, which puts every recording two levels below
    the campaign root the operator names on the command line.
    """
    assert len(discover([str(campaign / "nested")])) == 3


def test_finds_a_falcon_run_in_its_recording_subdirectory(campaign):
    """FALCON keeps the recording one level in, under ``recording/``."""
    found = discover([str(campaign / "falcon")])
    assert len(found) == 1
    assert found[0].name == "recording"


def test_walks_a_parent_of_several_campaigns(campaign):
    """Naming the parent finds every campaign below it."""
    assert len(discover([str(campaign)])) == 6


def test_expands_an_absolute_glob(campaign):
    """An absolute pattern is the normal case and must not raise.

    ``Path().glob`` rejects one outright with ``NotImplementedError``, so the
    pattern has to be expanded against the filesystem instead.
    """
    assert len(discover([str(campaign / "*")])) == 6


def test_expands_a_tilde_in_a_glob(monkeypatch, campaign):
    """``~`` has to be resolved before globbing, not after."""
    monkeypatch.setenv("HOME", str(campaign))
    assert len(discover(["~/nested/w*"])) == 3


def test_the_same_recording_reached_two_ways_is_returned_once(campaign):
    """Overlapping roots are deduplicated by resolved path."""
    both = discover([str(campaign / "nested"), str(campaign)])
    assert len(both) == 6


def test_does_not_descend_into_a_recordings_own_frame_directories(campaign):
    """The walk stops at the recording.

    Descending would read every one of a campaign's hundreds of thousands of
    image files to discover nothing, which is the difference between discovery
    taking a moment and taking minutes.
    """
    deep = campaign / "flat" / "office_w0_e000" / "rgb" / "surprise"
    _recording(deep)
    assert deep not in discover([str(campaign / "flat")])


def test_a_missing_root_is_not_an_error(tmp_path):
    """A root that does not exist yields nothing rather than raising."""
    assert discover([str(tmp_path / "absent")]) == []
