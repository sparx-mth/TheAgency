#!/usr/bin/env python3
"""The on-disk run folder for ``nav_debug_recorder_node``: layout + guarded writes.

One class owns where a recording lands and how it is written, so the node owns
only ROS wiring. Two rules are enforced here rather than at every call site:

  * **Both clocks on every row.** Rows go through
    ``nav_debug.schema.row(t, wall, ...)``, so the ROS1 recording joins the ROS2
    one (written in another container, on another ROS version) on ``wall``.
  * **A diagnostic never takes the flight down.** Every write is guarded and
    reports through a caller-supplied ``warn``; a full disk costs a debug row,
    not a mission.

The schema import is itself guarded. The repo is mounted read-only at
``/opt/sparx_agency`` with ``PYTHONPATH=/opt`` (see ``run_falcon.sh``), but a
recorder that refuses to start because a diagnostic contract module is missing
is exactly the failure this file exists to avoid, so the layout falls back to
its literal names.

Python 3.8 compatible (runs in the FALCON Noetic container).
"""
import json
import os
import time

import numpy as np

from thought_journal import LOG_DIR_ENV     # sibling: shared $FALCON_LOG_DIR resolution

try:
    from sparx_agency.tasks.planning.nav_debug import schema
except Exception:                           # noqa: BLE001 - never block a recording
    schema = None


def _const(name, default):
    """The schema's value for ``name``, or ``default`` if it is not importable."""
    return getattr(schema, name, default) if schema is not None else default


SCHEMA_VERSION = _const("SCHEMA_VERSION", 2)
TELEMETRY_FILE = _const("TELEMETRY_FILE", "telemetry.jsonl")
REFERENCE_FILE = _const("REFERENCE_FILE", "reference.jsonl")
CONTROL_FILE = _const("CONTROL_FILE", "control.jsonl")
EVENTS_FILE = _const("EVENTS_FILE", "events.jsonl")
MAPPING_FILE = _const("MAPPING_FILE", "mapping.jsonl")
MANIFEST_FILE = _const("MANIFEST_FILE", "manifest.json")
ROUTES_DIR = _const("ROUTES_DIR", "routes")
BEV_DIR = _const("BEV_DIR", "bev")
BEV_CONF_DIR = _const("BEV_CONF_DIR", "bev_conf")
CONTROL_TRACE_TOPIC = _const("CONTROL_TRACE_TOPIC", "/nav_debug/control_trace")


def row(t, wall, **fields):
    """One record carrying both clocks, via the schema when it is importable."""
    if schema is not None:
        return schema.row(t, wall, **fields)
    out = {"t": round(float(t), 3), "wall": round(float(wall), 3)}
    out.update(fields)
    return out


def default_out_dir(now=None):
    """Where this recording lands when ``~out_dir`` is unset.

    ``run_falcon.sh`` sets ``FALCON_RUN_DIR`` to the shared, single-stamp run
    folder so the thought journal, the certainty CSV and this recording all land
    together. Honour it; fall back to our own timestamped folder standalone.
    """
    run = os.environ.get("FALCON_RUN_DIR")
    if run:
        return run
    base = (os.environ.get(LOG_DIR_ENV)
            or os.path.join(os.path.expanduser("~"), ".ros", "falcon"))
    stamp = time.strftime("%Y%m%d_%H%M%S", now or time.localtime())
    return os.path.join(base, "nav_debug_%s" % stamp)


class RunFolder(object):
    """One recording on disk: the schema's layout, written guardedly.

    Args:
        out_dir: The run folder; created along with its subdirectories.
        warn: Called with a message when a write fails. Defaults to silence so
            the class stays usable (and testable) without ROS.
    """

    def __init__(self, out_dir, warn=None):
        self.out_dir = out_dir
        self.counts = {}
        self._warn = warn if warn is not None else _silent
        self._files = {}
        for sub in (BEV_DIR, BEV_CONF_DIR, ROUTES_DIR):
            _mkdir(os.path.join(out_dir, sub))

    def emit(self, name, t, wall, **fields):
        """Append one row to the jsonl file ``name``, flushed.

        Returns:
            bool: True if the row reached the file.
        """
        try:
            fh = self._files.get(name)
            if fh is None:
                fh = self._files[name] = open(os.path.join(self.out_dir, name), "a")
            fh.write(json.dumps(row(t, wall, **fields)) + "\n")
            fh.flush()
        except (OSError, IOError, ValueError, TypeError) as e:
            self._warn("%s write failed (%s)" % (name, e))
            return False
        self._bump(name)
        return True

    def save_bev(self, name, grid, geometry, conf=None):
        """Save one BEV snapshot: the grid, its geometry sidecar and confidence."""
        try:
            np.save(os.path.join(self.out_dir, BEV_DIR, name + ".npy"), grid)
            _write_json(os.path.join(self.out_dir, BEV_DIR, name + ".json"), geometry)
            if conf is not None:
                np.save(os.path.join(self.out_dir, BEV_CONF_DIR, name + ".npy"), conf)
        except (OSError, IOError, ValueError, TypeError) as e:
            self._warn("BEV save failed (%s)" % e)
            return False
        self._bump(BEV_DIR)
        return True

    def save_routes(self, name, payload):
        """Save one route-layer snapshot as ``routes/<name>``."""
        try:
            _write_json(os.path.join(self.out_dir, ROUTES_DIR, name), payload)
        except (OSError, IOError, ValueError, TypeError) as e:
            self._warn("routes save failed (%s)" % e)
            return False
        self._bump(ROUTES_DIR)
        return True

    def write_manifest(self, payload):
        """Write the run manifest, stamped with the schema version."""
        payload = dict(payload)
        payload["schema_version"] = SCHEMA_VERSION
        try:
            _write_json(os.path.join(self.out_dir, MANIFEST_FILE), payload)
        except (OSError, IOError, ValueError, TypeError) as e:
            self._warn("manifest write failed (%s)" % e)

    def summary(self):
        """Counts so far as a compact ``name=n`` string, for a heartbeat line."""
        return " ".join("%s=%d" % (k.split(".")[0], v)
                        for k, v in sorted(self.counts.items()))

    def close(self):
        """Close every open writer; safe to call twice."""
        for fh in self._files.values():
            try:
                fh.close()
            except (OSError, IOError):
                pass
        self._files = {}

    def _bump(self, name):
        self.counts[name] = self.counts.get(name, 0) + 1


def _silent(_message):
    """Default ``warn``: a run folder used without ROS says nothing."""


def _mkdir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def _write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh)
