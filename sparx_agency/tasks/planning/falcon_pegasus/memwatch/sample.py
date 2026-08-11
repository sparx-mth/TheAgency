"""Read how much memory the exploration node is actually holding.

Per-process rather than whole-container, because the question is specifically
what the voxel map costs, and the container also runs the trajectory server, the
bridge, the recorder and roscore. The container total is recorded alongside so
the two can be compared.

Nothing here needs a Python inside the container: ``docker exec`` plus ``/proc``
is enough, and ``/proc`` is the same number ``ps`` would report without the
parsing risk.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

DEFAULT_CONTAINER = "falcon-pegasus"
DEFAULT_PROCESS = "exploration_node"


@dataclass(frozen=True)
class Sample:
    """One reading.

    Attributes:
        elapsed_s: Seconds since sampling began.
        rss_bytes: Resident set size of the watched process, or None if it is
            not running yet — the container takes a few seconds to get there.
        container_bytes: Resident total across every process in the container.
    """

    elapsed_s: float
    rss_bytes: Optional[int]
    container_bytes: Optional[int]

    def csv_row(self) -> str:
        """One line for the CSV, blank where a reading was unavailable."""
        return "{:.2f},{},{}".format(
            self.elapsed_s,
            "" if self.rss_bytes is None else self.rss_bytes,
            "" if self.container_bytes is None else self.container_bytes,
        )


CSV_HEADER = "elapsed_s,rss_bytes,container_bytes"


def parse_vmrss_bytes(status_text: str) -> Optional[int]:
    """Pull ``VmRSS`` out of a ``/proc/<pid>/status`` dump.

    Args:
        status_text: The file's contents.

    Returns:
        Resident bytes, or None if the field is absent — which happens when the
        process exits between listing it and reading it.
    """
    for line in status_text.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return None


def parse_proc_dump(dump: str) -> Tuple[Optional[int], Optional[int]]:
    """Read one ``docker exec`` dump into a watched RSS and a container total.

    The dump is every process's ``VmRSS`` line, each prefixed with a marker when
    it belongs to the watched process, produced by :func:`_probe_script`. Doing
    it in one shell round trip rather than one per process keeps the sampler
    cheap enough to run at 1 Hz beside a flight.

    Args:
        dump: Standard output of the probe.

    Returns:
        ``(watched_bytes, container_bytes)``; either may be None.
    """
    watched = None
    total = 0
    saw_any = False

    for line in dump.splitlines():
        line = line.strip()
        if not line:
            continue
        marked = line.startswith("*")
        value = parse_vmrss_bytes(line.lstrip("*").strip())
        if value is None:
            continue
        saw_any = True
        total += value
        if marked:
            # The LARGEST match, not the sum. More than one process carries the
            # node's name on its command line -- roslaunch launched it, so its
            # own cmdline contains it too -- and summing them silently folds a
            # supervisor's few megabytes into the figure, then reports the
            # supervisor alone once the node itself has gone.
            watched = value if watched is None else max(watched, value)

    return (watched, total if saw_any else None)


def _probe_script(process: str) -> str:
    """The shell run inside the container to dump every process's RSS.

    Marks the watched process's lines with a leading ``*``. Reads ``cmdline``
    rather than ``comm`` because ``comm`` is truncated to fifteen characters and
    ``exploration_node`` is sixteen.

    Args:
        process: Substring to look for in the command line.

    Returns:
        A ``sh`` script.
    """
    return (
        'for d in /proc/[0-9]*; do '
        '  [ -r "$d/status" ] || continue; '
        '  rss=$(grep "^VmRSS:" "$d/status" 2>/dev/null) || continue; '
        '  if tr "\\0" " " < "$d/cmdline" 2>/dev/null | grep -q "%s"; '
        '  then echo "*$rss"; else echo "$rss"; fi; '
        'done' % process
    )


def sample_once(
    container: str = DEFAULT_CONTAINER,
    process: str = DEFAULT_PROCESS,
    timeout_s: float = 15.0,
) -> Tuple[Optional[int], Optional[int]]:
    """Take one reading from a running container.

    Args:
        container: Container name.
        process: Substring identifying the process to watch.
        timeout_s: Give up on the ``docker exec`` after this long.

    Returns:
        ``(watched_bytes, container_bytes)``, both None if the container is gone.
    """
    try:
        completed = subprocess.run(
            ["docker", "exec", container, "sh", "-c", _probe_script(process)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
        )
    except (OSError, subprocess.TimeoutExpired):
        return (None, None)
    if completed.returncode != 0:
        return (None, None)
    return parse_proc_dump(completed.stdout.decode("utf-8", "replace"))


def container_is_running(container: str = DEFAULT_CONTAINER) -> bool:
    """Whether the container exists and is up.

    Args:
        container: Container name.

    Returns:
        True if docker reports it running.
    """
    try:
        completed = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.stdout.decode("utf-8", "replace").strip() == "true"


def read_csv(text: str) -> List[Sample]:
    """Parse a CSV this module wrote, for re-summarising a finished run.

    Args:
        text: File contents, including the header.

    Returns:
        The samples, in order.
    """
    samples: List[Sample] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("elapsed_s"):
            continue
        fields = line.split(",")
        if len(fields) < 3:
            continue
        samples.append(
            Sample(
                elapsed_s=float(fields[0]),
                rss_bytes=int(fields[1]) if fields[1] else None,
                container_bytes=int(fields[2]) if fields[2] else None,
            )
        )
    return samples
