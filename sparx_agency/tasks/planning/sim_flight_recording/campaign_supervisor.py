"""Keep a collection campaign flying for hours without anyone watching it.

:mod:`collect` is deliberately willing to stop. Three failed episodes in a row
means the aircraft is wedged against something it cannot right itself from, and
carrying on would only write more bad data — so the worker exits. That is the
correct behaviour for one campaign and the wrong behaviour for an overnight
one, where a wedge at episode 20 of 400 would cost the other 380.

This supervisor closes that gap. It runs ``run_collection.sh``'s worker command
directly, one subprocess per worker, and relaunches any worker that exits until
a campaign-wide target is met. A wedged aircraft therefore costs one Kit boot —
about three and a half minutes — instead of a worker's whole remaining share.

**Every launch gets its own output directory**, and it has to. ``collect.py``
names recordings ``<scene>_w<worker>_e<index>`` with the index restarting at
zero each launch, so a relaunched worker writing to the directory it used
before would overwrite its own earlier flights one by one, silently, and the
campaign would never grow. Launch *n* of worker *w* writes to ``w<w>_c<n>/``
instead, which also gives each launch its own campaign manifest.

Stopping, in the order the supervisor checks them:

``--episodes``    total finished flights across every worker and launch
``--max-bytes``   campaign size on disk; the run stops rather than filling it
``--hours``       wall-clock budget
``--stop-file``   ``touch`` it to bring the campaign down within seconds,
                  leaving every finished recording valid -- at most the one
                  flight per worker still in the air is lost

Run it from the host, with the container already up::

    python3 sparx_agency/tasks/planning/sim_flight_recording/campaign_supervisor.py \\
        --scene office --workers 4 --episodes 2000 --max-bytes 400e9 \\
        --host-dir ~/data/sim/office --container-dir /data/office
"""
from __future__ import annotations

import argparse
import json
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sparx_agency.tools.campaign_monitor import collection, resources  # noqa: E402

STAGGER_S = 45.0
"""Seconds between worker starts.

Kit's start-up is the heaviest moment of a worker's life, and overlapping two of
them contends for the GPU hard enough to crash the RTX shader compiler. This is
the same interval ``run_collection.sh`` uses, for the same reason.
"""

RELAUNCH_BACKOFF_S = 20.0
"""Pause before restarting a worker, so a worker failing instantly cannot spin."""


@dataclass
class WorkerState:
    """One worker slot: at most one live process, plus what it has cost so far."""

    index: int
    scene: str = "office"
    launches: int = 0
    process: Optional[subprocess.Popen] = None
    launched_at: float = 0.0
    ready_at: float = 0.0
    exits: List[int] = field(default_factory=list)

    @property
    def alive(self) -> bool:
        """Whether this slot currently holds a running worker."""
        return self.process is not None and self.process.poll() is None


class Supervisor:
    """Runs ``--workers`` collection processes and keeps relaunching them."""

    def __init__(self, args) -> None:
        self.args = args
        self.host_dir = Path(args.host_dir).expanduser()
        self.host_dir.mkdir(parents=True, exist_ok=True)
        # The container writes here as uid 1234 and this supervisor as the host
        # user, and neither is in the other's group. Whoever creates the
        # directory first would otherwise lock the other out of it.
        try:
            self.host_dir.chmod(0o2777)
        except OSError:
            pass
        self.started = time.time()
        self.stop_requested = False
        # Scenes are dealt round-robin across worker slots rather than flown as
        # one campaign after another, so a run that is cut short still holds
        # every building instead of only the first ones. A policy meant to fly
        # any indoor space is limited by how many *buildings* it has seen far
        # more than by how many frames of one it has.
        scenes = args.scenes or [args.scene]
        self.workers = [
            WorkerState(index=index, scene=scenes[index % len(scenes)],
                        launches=self._launches_so_far(index))
            for index in range(args.workers)
        ]
        self.log_path = self.host_dir / "supervisor.log"
        self.state_path = self.host_dir / "supervisor.json"

    def _launches_so_far(self, index: int) -> int:
        """Highest launch number worker ``index`` has already used here.

        A campaign is routinely restarted — to change the worker count, after a
        reboot, or to resume one that was stopped. Numbering from zero again
        would point the new launch at a directory that already holds flights,
        and ``collect.py`` would overwrite them one by one as its episode index
        counted back up from ``e000``. Continuing the sequence keeps every
        earlier flight, which matters because a crashed episode is still
        training data here: the label comes from the map, not from the flight.
        """
        highest = 0
        # Launch directories are named "<scene>_w<index>_c<launch>", and the
        # scene may differ between runs, so match on the worker index rather
        # than on a fixed prefix -- a resumed campaign that deals scenes
        # differently must still not reuse a directory that holds flights.
        for existing in self.host_dir.glob(f"*w{index}_c*"):
            suffix = existing.name.split("_c")[-1]
            if suffix.isdigit():
                highest = max(highest, int(suffix))
        return highest

    # ----------------------------------------------------------------- logging
    def say(self, message: str) -> None:
        """One timestamped line, to the console and to the campaign's own log."""
        stamped = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(stamped, flush=True)
        with self.log_path.open("a") as handle:
            handle.write(stamped + "\n")

    def save_state(self, progress: collection.CollectionProgress, used_bytes: int) -> None:
        """Publish the numbers the dashboard reads, so it need not re-scan."""
        payload = {
            "scenes": sorted({w.scene for w in self.workers}),
            "started": self.started,
            "updated": time.time(),
            "target_episodes": self.args.episodes,
            "max_bytes": self.args.max_bytes,
            "used_bytes": used_bytes,
            "done": progress.done,
            "landed": progress.landed,
            "frames": progress.frames,
            "outcomes": progress.outcomes,
            "workers": [
                {"index": worker.index, "scene": worker.scene, "launches": worker.launches,
                 "alive": worker.alive, "exits": worker.exits}
                for worker in self.workers
            ],
            "stopping": self.stop_requested,
        }
        self.state_path.write_text(json.dumps(payload, indent=2))

    # --------------------------------------------------------------- launching
    def _in_container(self, script: str) -> None:
        """Run a shell snippet inside the container, ignoring its exit status."""
        subprocess.run(["docker", "exec", self.args.container, "bash", "-c", script],
                       check=False, capture_output=True)

    def kill_stale_px4(self, instance: int) -> None:
        """Kill a PX4 left behind by an abruptly-terminated worker.

        A killed worker never reaches ``px4_launch.terminate_px4``, so its PX4
        outlives it holding TCP 4560+N. The next worker on that instance then
        fails to bind, and PX4 says nothing useful about why. Instances are told
        apart by working directory rather than by scanning for ``-i N`` in a
        command line, because that is what ``px4_launch.working_dir`` actually
        guarantees is unique — and it must never match a *live* sibling worker.
        """
        self._in_container(
            f"for pid in $(pgrep -f 'bin/px4' 2>/dev/null); do "
            f"  target=$(readlink /proc/$pid/cwd 2>/dev/null); "
            f"  case \"$target\" in *instance_{instance}) kill -9 $pid 2>/dev/null;; esac; "
            f"done; "
            f"pkill -9 -f 'rcS {instance}$' 2>/dev/null; "
            f"pkill -9 -f 'simulator_mavlink --instance {instance} ' 2>/dev/null; true"
        )

    def launch(self, worker: WorkerState) -> None:
        """Start one ``collect.py`` inside the container, in a fresh directory."""
        worker.launches += 1
        # The scene is in the directory name because the dataset builder infers
        # nothing from the tree -- every stage downstream is told which scene a
        # recording belongs to, and a human reading the campaign directory
        # should be able to see the mix at a glance.
        name = f"{worker.scene}_w{worker.index}_c{worker.launches:03d}"
        container_out = f"{self.args.container_dir}/{name}"
        host_out = self.host_dir / name
        seed = self.args.seed + worker.index * 1000 + worker.launches

        self.kill_stale_px4(worker.index)
        self._in_container(
            f"rm -f /tmp/px4_lock-{worker.index} /tmp/px4-sock-{worker.index}; "
            f"mkdir -p '{container_out}'; chmod 2777 '{container_out}'"
        )

        collect_args = [
            "--scene", worker.scene,
            "--out-dir", container_out,
            "--episodes", str(self.args.episodes_per_launch),
            "--altitude", str(self.args.altitude),
            "--rate-hz", str(self.args.rate_hz),
            "--worker", str(worker.index),
            "--seed", str(seed),
            "--pegasus-root", "/tmp/dev/PegasusSimulator/extensions/pegasus.simulator",
            "--px4-dir", "/tmp/dev/PX4-Autopilot",
        ] + list(self.args.extra)

        quoted = " ".join(f"'{value}'" for value in collect_args)
        command = (
            f"cd /tmp/dev/repo && /isaac-sim/python.sh "
            f"sparx_agency/tasks/planning/sim_flight_recording/collect.py {quoted} "
            f"> '{container_out}/worker{worker.index}.log' 2>&1"
        )
        worker.process = subprocess.Popen(
            ["docker", "exec", self.args.container, "bash", "-c", command],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        worker.launched_at = time.time()
        self.say(f"worker {worker.index} launch {worker.launches} -> {host_out.name} (seed {seed})")

    def stop_all(self) -> None:
        """Ask every worker to finish, then make sure none is left holding the GPU.

        Terminating the ``docker exec`` client does not reliably reach the
        process inside the container, and Kit ignores SIGTERM outright while it
        is compiling shaders. So the polite request is followed by an
        unconditional sweep of both Kit and every PX4 it started — leaving
        either behind costs the *next* campaign its GPU memory and its ports.
        """
        for worker in self.workers:
            if worker.alive:
                worker.process.terminate()
        deadline = time.time() + 60.0
        while time.time() < deadline and any(worker.alive for worker in self.workers):
            time.sleep(1.0)
        self._in_container("pkill -9 -f collect.py; true")
        for worker in self.workers:
            self.kill_stale_px4(worker.index)

    # ------------------------------------------------------------------- limits
    def limit_reached(self, progress: collection.CollectionProgress,
                      used_bytes: int) -> Optional[str]:
        """The first stopping condition that is now true, if any."""
        if self.args.episodes and progress.done >= self.args.episodes:
            return f"episode target reached ({progress.done}/{self.args.episodes})"
        if self.args.max_bytes and used_bytes >= self.args.max_bytes:
            return f"disk cap reached ({used_bytes / 1e9:.0f} GB)"
        if self.args.hours and time.time() - self.started >= self.args.hours * 3600:
            return f"time budget reached ({self.args.hours} h)"
        if self.args.stop_file and Path(self.args.stop_file).exists():
            return "stop file present"
        free_gb = shutil.disk_usage(self.host_dir).free / 1e9
        if free_gb < self.args.min_free_gb:
            return f"only {free_gb:.0f} GB left on the filesystem"
        return None

    # --------------------------------------------------------------------- run
    def run(self) -> int:
        """Launch, supervise and relaunch until a stopping condition is met."""
        size = resources.CachedDirectorySize(self.host_dir, interval_s=60.0)
        self.say(f"campaign start: scenes={[w.scene for w in self.workers]} workers={self.args.workers} "
                 f"target={self.args.episodes} cap={self.args.max_bytes / 1e9:.0f} GB")

        for index, worker in enumerate(self.workers):
            if self.stop_requested:
                break
            self.launch(worker)
            if index + 1 < len(self.workers):
                time.sleep(STAGGER_S)

        last_report = 0.0
        while True:
            time.sleep(5.0)
            progress = collection.scan(self.host_dir)
            used = size.get()

            reason = self.limit_reached(progress, used)
            if reason and not self.stop_requested:
                self.stop_requested = True
                # Stop now rather than waiting out the rest of every worker's
                # launch, which is up to --episodes-per-launch more flights and
                # can be the best part of an hour. Nothing is lost by being
                # abrupt: FlightRecorder writes each episode's images as they
                # are captured and closes poses/meta before the next episode
                # starts, so at most the one flight in the air is discarded --
                # and a directory without poses.npy is skipped by the trainer's
                # discovery rather than half-ingested.
                self.say(f"stopping: {reason}")
                self.stop_all()
                self.save_state(progress, used)
                self.say(f"campaign finished: {progress.done} flights, "
                         f"{progress.landed} landed, {used / 1e9:.1f} GB")
                return 0

            for worker in self.workers:
                if worker.alive:
                    continue
                if worker.process is not None:
                    code = worker.process.returncode
                    worker.exits.append(code if code is not None else -1)
                    self.say(f"worker {worker.index} exited ({code}) after "
                             f"{(time.time() - worker.launched_at) / 60:.0f} min")
                    worker.process = None
                if self.stop_requested:
                    continue
                time.sleep(RELAUNCH_BACKOFF_S)
                self.launch(worker)

            self.save_state(progress, used)
            if time.time() - last_report >= 300:
                last_report = time.time()
                rate = progress.rate_per_hour()
                pace = f", {rate:.0f} flights/h" if rate else ""
                self.say(f"{progress.done} flights ({progress.landed} landed), "
                         f"{used / 1e9:.1f} GB{pace}")

            if self.stop_requested and not any(worker.alive for worker in self.workers):
                self.say(f"campaign finished: {progress.done} flights, "
                         f"{progress.landed} landed, {used / 1e9:.1f} GB")
                return 0


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="office",
                        help="single scene to fly; ignored when --scenes is given")
    parser.add_argument("--scenes", nargs="+", default=None,
                        help="fly several buildings at once, dealt round-robin "
                             "across worker slots. Each must already be surveyed")
    parser.add_argument("--workers", type=int, default=4,
                        help="concurrent Isaac Sim processes; PX4 caps this at 10")
    parser.add_argument("--episodes", type=int, default=0,
                        help="campaign-wide finished-flight target; 0 for no limit")
    parser.add_argument("--episodes-per-launch", type=int, default=60,
                        help="flights one worker attempts before it is recycled")
    parser.add_argument("--max-bytes", type=float, default=0.0,
                        help="stop once the campaign directory reaches this size")
    parser.add_argument("--hours", type=float, default=0.0, help="wall-clock budget")
    parser.add_argument("--min-free-gb", type=float, default=200.0,
                        help="stop rather than leave the filesystem with less than this")
    parser.add_argument("--stop-file", default=None,
                        help="touch this path to stop cleanly at the next episode boundary")
    parser.add_argument("--host-dir", required=True,
                        help="where the recordings land on the host")
    parser.add_argument("--container-dir", required=True,
                        help="the same directory as the container sees it")
    parser.add_argument("--container", default="isaac-sim")
    parser.add_argument("--altitude", type=float, default=1.5)
    parser.add_argument("--rate-hz", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("extra", nargs="*", default=[],
                        help="further arguments passed straight to collect.py")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """Entry point: supervise a campaign until one of its limits is reached."""
    args = _parse_args(argv)
    if not 1 <= args.workers <= 10:
        raise SystemExit("--workers must be 1..10 (PX4 gives instances >=10 the same UDP port)")

    supervisor = Supervisor(args)

    def request_stop(_signum, _frame):
        supervisor.stop_requested = True
        supervisor.say("signal received; stopping after the current flights")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        return supervisor.run()
    finally:
        supervisor.stop_all()


if __name__ == "__main__":
    raise SystemExit(main())
