"""The editor's behaviour when the screen it is showing changes underneath it.

These need a display, so they skip where there is none (a headless CI box);
everything they cover is a real bug that reached a running window.
"""
import os

import pytest

from sparx_agency.tasks.common.launch_params.editor import ParameterEditor
from sparx_agency.tasks.common.launch_params.spec import ParamSet, ParamSpec

tk = pytest.importorskip("tkinter")
pytestmark = pytest.mark.skipif(not os.environ.get("DISPLAY"),
                                reason="needs an X display")


@pytest.fixture
def editor():
    root = tk.Tk()
    widget = ParameterEditor(root)
    widget.pack()
    yield widget
    root.destroy()


def make(names, section="S"):
    return ParamSet([ParamSpec(n, "0", section=section) for n in names])


def visible(editor):
    return [row.param.name for row in editor._rows if row.widgets[0].winfo_ismapped()]


def test_a_filter_still_applies_after_switching_to_another_command(editor):
    """It used to reset silently: the box still said "Changed only" while the
    new screen showed all of its parameters."""
    editor.show(make(["alpha", "beta"]))
    editor._changed_only.set(True)
    editor.update()
    assert visible(editor) == []

    editor.show(make(["gamma", "delta"]))
    editor.update()
    assert visible(editor) == [], "the filter the operator left set still holds"


def test_a_search_term_survives_switching_command(editor):
    editor.show(make(["alpha", "beta"]))
    editor._search.set("alph")
    editor.update()
    editor.show(make(["alphabet", "zulu"]))
    editor.update()
    assert visible(editor) == ["alphabet"]


def test_sections_keep_their_order_through_a_filter_round_trip(editor):
    """Re-packing only the sections that reappeared shuffled them behind the
    ones that never left, which on a 292-parameter form loses your place."""
    params = ParamSet([ParamSpec("a", "0", section="One"),
                       ParamSpec("b", "0", section="Two"),
                       ParamSpec("c", "0", section="Three")])
    editor.show(params)
    editor.update()

    def packed_sections():
        return [frame.cget("text")
                for frame in editor._scroller.body.pack_slaves()]

    assert packed_sections() == ["One", "Two", "Three"]

    editor._search.set("b")      # hides One and Three
    editor.update()
    editor._search.set("")       # brings them back
    editor.update()
    assert packed_sections() == ["One", "Two", "Three"]
