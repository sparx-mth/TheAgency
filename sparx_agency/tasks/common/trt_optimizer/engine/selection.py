"""The file that says which engines are blessed to run, and the rule for writing it.

A directory of engines is not a decision. Something has to record *which*
precision passed its gate on this device, and the runtime has to be able to read
that without re-running a benchmark. That record is ``selected.json``.

The rule that matters is when NOT to write it. Selection is a side effect of
evaluation -- you learn which precision wins by building and gating each one --
so a crash in the middle, or a race where nothing passed, must never leave the
file pointing at an engine that was never blessed. :func:`clear` exists for
exactly that, and every caller that writes a selection must be prepared to call
it from an ``except BaseException`` handler. The alternative is a server quietly
loading a gate-failed engine, which is the worst outcome this package can
produce.

The schema is self-describing on purpose: a consumer should not have to know the
model's geometry to load it correctly.
"""
from __future__ import annotations

import json
from pathlib import Path

#: Name of the selection file inside an engine directory.
SELECTED = "selected.json"


def write(engine_dir, precision, engines, extra=None):
    """Record the blessed precision and its engine files.

    Args:
        engine_dir: the per-target engine directory.
        precision: the precision that won its gate.
        engines: mapping of engine key -> engine filename (not a full path, so
            the directory can be relocated).
        extra: any self-describing geometry a consumer needs to load correctly
            (sample counts, step counts, horizons). Recording it here means a
            loader never has to guess.

    Returns:
        pathlib.Path: the written file.
    """
    payload = {"precision": precision, "engines": dict(engines)}
    payload.update(extra or {})
    path = Path(engine_dir) / SELECTED
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def read(engine_dir):
    """Load the selection.

    Raises:
        FileNotFoundError: with the reason it is usually absent -- the directory
            was never benchmarked, or a gate failed and the selection was
            correctly withdrawn.
    """
    path = Path(engine_dir) / SELECTED
    if not path.exists():
        raise FileNotFoundError(
            "%s has no %s: this engine directory was never benchmarked, or a "
            "gate failed and the selection was withdrawn. Run the bench stage."
            % (engine_dir, SELECTED))
    return json.loads(path.read_text())


def clear(engine_dir):
    """Remove the selection, leaving nothing blessed.

    Call this from an ``except BaseException`` handler around any selection
    process. Leaving a stale selection behind is worse than leaving none.
    """
    Path(engine_dir, SELECTED).unlink(missing_ok=True)
