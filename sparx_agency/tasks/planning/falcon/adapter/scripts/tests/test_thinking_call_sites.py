"""Static audit of every ``thinker.say()`` call site in the adapter nodes.

``say()`` deliberately RAISES on a mislabelled thought -- an unknown category or
level is a bug, not something to quietly mis-colour in the operator's log. The
sharp edge is *when* it would raise: a typo in a rare branch ("no route", "target
lost", "I am stuck") stays dormant until the drone reaches that branch -- i.e.
until it is already in trouble, mid-flight, which is the worst possible moment
and the hardest to reproduce on the bench.

So the labels are audited statically instead of waiting for the branch to run.
The nodes cannot be imported (they need a live ROS graph), so the check reads the
source with ``ast``: it needs no rospy, and it sees dead and rare branches alike.

Also pinned: ``say(text)`` takes ONE positional argument. ``say("wp %d", n)`` --
the %-style habit every rospy logger encourages -- would silently narrate the
literal "wp %d" rather than fail, so it is caught here as an arity error.

SCOPE, honestly: this audits the labels it can SEE. A handful of sites pass
``level`` as a variable (a funnel's parameter, or a lookup in a small narration
table beside the code it narrates); their labels are literals at the caller or in
the table, but not at the ``say()`` call, so the checks below skip them rather
than guess. ``say()``'s own runtime raise remains the backstop there. This is a
net, not a proof.
"""
import ast
import pathlib

import pytest

from sparx_agency.core.common.thought_message import CATEGORIES, LEVELS

_SCRIPTS = pathlib.Path(__file__).resolve().parents[1]


def _say_calls():
    """Every ``*.say(...)`` call in the adapter scripts, as (file, node, lineno)."""
    found = []
    for path in sorted(_SCRIPTS.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "say"):
                found.append((path.name, node, node.lineno))
    return found


def _kwarg(call, name):
    """The literal string value of keyword ``name``, or None if absent/dynamic."""
    for kw in call.keywords:
        if kw.arg == name and isinstance(kw.value, ast.Constant):
            return kw.value.value
    return None


SAY_CALLS = _say_calls()


def test_the_nodes_actually_narrate():
    # A guard on the guard: if the AST walk silently found nothing, every
    # assertion below would vacuously pass and this file would prove nothing.
    assert len(SAY_CALLS) > 20, "expected the nav stack to narrate; found %d" % len(
        SAY_CALLS)


def test_every_call_site_uses_a_known_category():
    bad = ["%s:%d category=%r" % (f, ln, _kwarg(c, "category"))
           for f, c, ln in SAY_CALLS
           if _kwarg(c, "category") is not None
           and _kwarg(c, "category") not in CATEGORIES]
    assert not bad, "unknown thought categories (say() would raise here): %s" % bad


def test_every_call_site_uses_a_known_level():
    bad = ["%s:%d level=%r" % (f, ln, _kwarg(c, "level"))
           for f, c, ln in SAY_CALLS
           if _kwarg(c, "level") is not None
           and _kwarg(c, "level") not in LEVELS]
    assert not bad, "unknown thought levels (say() would raise here): %s" % bad


def test_no_call_site_passes_a_lazily_formatted_message():
    # say("waypoint %d", n) would narrate the literal "waypoint %d".
    bad = ["%s:%d (%d positional args)" % (f, ln, len(c.args))
           for f, c, ln in SAY_CALLS if len(c.args) > 1]
    assert not bad, ("say() takes one pre-formatted text; use "
                     'say("wp %%d" %% n): %s' % bad)


def test_no_call_site_narrates_an_empty_literal():
    bad = ["%s:%d" % (f, ln) for f, c, ln in SAY_CALLS
           if c.args and isinstance(c.args[0], ast.Constant)
           and isinstance(c.args[0].value, str) and not c.args[0].value.strip()]
    assert not bad, "empty thought text (say() would raise here): %s" % bad


@pytest.mark.parametrize("expected", ["nav", "plan", "object", "sensor", "map",
                                      "mission"])
def test_the_stack_narrates_every_subsystem(expected):
    # The operator asked to see the whole train of thought, not one node's slice.
    # A subsystem that never speaks is a blind spot in the log.
    used = {_kwarg(c, "category") or "nav" for _, c, _ in SAY_CALLS}
    assert expected in used, "no node narrates category %r" % expected
