"""Run-folder output for the ROS 2 half of a nav-debug recording.

Owns three things the recorder node should not have to think about: where a run
folder lives when nobody said, how a jsonl line is appended without ever being
able to take the process down, and the manifest that makes a silent QoS mismatch
detectable after the fact.

**Where the files land.** The recorder must run inside the ``it`` container --
that is the only place the vendor message packages are built -- and ``it``'s own
filesystem is not the host's: a recording written to the container's ``/tmp``
exists nowhere else until someone remembers ``docker cp``, and is destroyed
outright when ``it`` is recreated (which some Sphera restarts do). So the default
base directory is the container's ``workspace`` bind mount, which is a
read-write bind of a host directory, and every byte lands on the host as it is
recorded. ``$ROOSTER_WORKSPACE`` overrides the mount point, ``$NAV_DEBUG_RUN_DIR``
overrides the whole run folder -- that is how a launcher points the ROS 1 and
ROS 2 recorders at the same run stamp -- and the ``~out_dir`` rosparam overrides
everything. Outside the container the mount does not exist and the fallback is
the user cache dir, so the node still runs on a dev box.

Python 3.8 compatible: this runs under ROS 2 Foxy in the vendor container.
"""
from __future__ import annotations

import json
import os
import time

from sparx_agency.robots.ROBOTICAN.nav_debug_ros2_imports import schema

#: Env var naming the shared run folder both recorders write into.
RUN_DIR_ENV = "NAV_DEBUG_RUN_DIR"
#: Env var overriding the vendor container's bind-mounted workspace.
WORKSPACE_ENV = "ROOSTER_WORKSPACE"
#: Mount point of that workspace inside ``it`` (a rw bind of a host directory).
DEFAULT_WORKSPACE = os.path.join(os.sep, "home", "rooster", "workspace")
LOGS_SUBDIR = "nav_debug_logs"
RUN_PREFIX = "nav_debug_"

#: Sits inside ``<run_dir>/ros2`` beside the streams it describes, so the whole
#: directory copies out of the container as one self-describing unit.
MANIFEST_ROS2_FILE = "manifest_ros2.json"


def default_base_dir():
    """Directory new run folders are created under.

    Returns:
        The container workspace's log dir when that bind mount exists, else a
        path under the user's cache dir so the node still works off-container.
    """
    workspace = os.environ.get(WORKSPACE_ENV) or DEFAULT_WORKSPACE
    if os.path.isdir(workspace):
        return os.path.join(workspace, LOGS_SUBDIR)
    return os.path.join(os.path.expanduser("~"), ".cache", "sparx_agency",
                        "falcon_nav_logs", "sphera_ros2")


def resolve_run_dir(out_dir=""):
    """Pick the run folder: ``~out_dir``, then the env var, then a fresh stamp.

    Args:
        out_dir: The ``~out_dir`` rosparam; empty means "not set".

    Returns:
        Absolute path of the run folder (not yet created).
    """
    if out_dir:
        return os.path.abspath(out_dir)
    shared = os.environ.get(RUN_DIR_ENV)
    if shared:
        return os.path.abspath(shared)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(default_base_dir(), RUN_PREFIX + stamp)


class RunWriter(object):
    """Append-only jsonl files under ``<run_dir>/ros2``, plus the manifest.

    Every write is guarded and counted rather than raised: this is a diagnostic
    riding along with a live flight, and a full disk must cost the recording,
    not the flight. ``errors`` being non-zero next to a healthy ``rows`` count is
    how a partially-written run announces itself.

    Attributes:
        ros2_dir: Where the stream files are written.
        rows: Rows successfully written, per filename.
        errors: Failed writes since start.
    """

    def __init__(self, run_dir):
        """Create the run folder and its ``ros2`` subdirectory.

        Args:
            run_dir: Run folder; created with parents if missing.

        Raises:
            OSError: If the directory cannot be created -- there is nothing to
                record into, so this one failure is worth refusing to start on.
        """
        self.run_dir = run_dir
        self.ros2_dir = os.path.join(run_dir, schema.ROS2_DIR)
        os.makedirs(self.ros2_dir, exist_ok=True)
        self._handles = {}
        self.rows = {}
        self.errors = 0

    def write(self, filename, obj):
        """Append one jsonl row, opening the file on first use.

        Args:
            filename: A ``schema`` stream filename, e.g. ``ACTUATOR_FILE``.
            obj: A ``schema.row()`` dict.

        Returns:
            True if the row reached the file.
        """
        try:
            handle = self._handles.get(filename)
            if handle is None:
                # Append: a recorder restarted mid-flight must never truncate.
                handle = open(os.path.join(self.ros2_dir, filename), "a")
                self._handles[filename] = handle
            handle.write(json.dumps(obj) + "\n")
            handle.flush()
            self.rows[filename] = self.rows.get(filename, 0) + 1
            return True
        except (OSError, IOError, ValueError, TypeError):
            self.errors += 1
            return False

    def write_manifest(self, payload):
        """Write ``manifest_ros2.json``, overwriting the previous copy.

        Called at startup, periodically and at shutdown, so even a hard-killed
        recorder leaves near-final per-stream counts behind.

        Args:
            payload: Manifest dict; ``rows``/``write_errors`` are filled in here.

        Returns:
            True if the manifest reached disk.
        """
        payload = dict(payload)
        payload["rows"] = dict(self.rows)
        payload["write_errors"] = self.errors
        path = os.path.join(self.ros2_dir, MANIFEST_ROS2_FILE)
        try:
            with open(path, "w") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
            return True
        except (OSError, IOError, ValueError, TypeError):
            self.errors += 1
            return False

    def close(self):
        """Close every open stream file. Each line was already flushed."""
        for handle in self._handles.values():
            try:
                handle.close()
            except (OSError, IOError):
                pass
        self._handles = {}
