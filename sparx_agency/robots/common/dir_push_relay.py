#!/usr/bin/env python3
"""dir_push_relay.py

Watches a local directory for new files and pushes each one to a remote
host's directory via rsync over a persistent SSH connection (ControlMaster),
so a producer's local file becomes visible on a machine that doesn't share
its filesystem.

Motivating case: rooster_frame_dir_publisher.py writes JPEGs on the host
(it needs the UDP port the ROBOTICAN container's network_mode: host
exposes), but depth_processor_node.py/localization_node.py need the DA3
TensorRT engine and only run on the Jetson. This relay is the missing link
between the two, reusable for any similar host/Jetson split.

Single responsibility: detect new files, push them. Does not touch what
wrote them or what will read them on the other end — pair with
dir_watch_path_publisher.py running on the destination machine.

New files are detected by mtime, not filename, so it's robust to a
producer's sequence counter resetting across restarts. rsync's default
temp-file-then-rename behavior (not disabled here — no --inplace) means
the destination side never sees a partially-transferred file.
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def _new_files(watch_dir: Path, pattern: str, since_mtime: float) -> list[Path]:
    files = [
        p for p in watch_dir.glob(pattern)
        if p.is_file() and p.stat().st_mtime > since_mtime
    ]
    files.sort(key=lambda p: p.stat().st_mtime)
    return files


class DirPushRelay:
    def __init__(
        self,
        watch_dir: str,
        pattern: str,
        remote_host: str,
        remote_dir: str,
        poll_interval: float = 0.05,
        max_in_flight: int = 4,
        include_existing: bool = False,
        control_path: str = "/tmp/ssh-dir-push-relay-%r@%h:%p",
    ):
        self.watch_dir = Path(watch_dir).expanduser().resolve()
        self.pattern = pattern
        self.remote_host = remote_host
        self.remote_dir = remote_dir
        self.poll_interval = poll_interval
        self.max_in_flight = max_in_flight
        # ControlMaster: the first ssh/rsync call opens the connection and
        # backgrounds it; every later call reuses it instead of paying a
        # fresh TCP+SSH handshake — the difference between ~100ms+ and a
        # few ms per push, which matters at 10Hz.
        self.ssh_opts = (
            f"ssh -o ControlMaster=auto -o ControlPersist=60s "
            f"-o ControlPath={control_path} -o ConnectTimeout=5"
        )
        self._last_mtime = 0.0 if include_existing else time.time()
        self._in_flight: list[subprocess.Popen] = []

        self._ensure_remote_dir()

    def _ensure_remote_dir(self) -> None:
        subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", self.remote_host, f"mkdir -p {self.remote_dir}"],
            check=True,
        )

    def _reap_in_flight(self) -> None:
        still_running = []
        for proc in self._in_flight:
            if proc.poll() is None:
                still_running.append(proc)
            elif proc.returncode != 0:
                stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
                print(f"[dir_push_relay] rsync failed (exit {proc.returncode}): {stderr.strip()}")
        self._in_flight = still_running

    def _push(self, path: Path) -> None:
        if len(self._in_flight) >= self.max_in_flight:
            print(f"[dir_push_relay] {len(self._in_flight)} pushes already in flight, dropping {path.name}")
            return
        remote = f"{self.remote_host}:{self.remote_dir}/"
        proc = subprocess.Popen(
            ["rsync", "-e", self.ssh_opts, str(path), remote],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        self._in_flight.append(proc)

    def run_forever(self) -> None:
        print(f"[dir_push_relay] watching {self.watch_dir} -> {self.remote_host}:{self.remote_dir}")
        while True:
            self._reap_in_flight()
            for path in _new_files(self.watch_dir, self.pattern, self._last_mtime):
                self._push(path)
                self._last_mtime = path.stat().st_mtime
            time.sleep(self.poll_interval)


def parse_args():
    p = argparse.ArgumentParser(
        description="Watch a directory and push new files to a remote host via rsync/SSH.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--watch-dir", required=True)
    p.add_argument("--pattern", default="*.jpg")
    p.add_argument("--remote-host", required=True, help="e.g. user@192.0.0.89")
    p.add_argument("--remote-dir", required=True)
    p.add_argument("--poll-interval", type=float, default=0.05)
    p.add_argument("--max-in-flight", type=int, default=4,
                    help="Cap on concurrent rsync pushes; newer frames are dropped beyond this")
    p.add_argument("--include-existing", action="store_true",
                    help="Push files already present in --watch-dir at startup (default: only new arrivals)")
    return p.parse_args()


def main():
    args = parse_args()
    relay = DirPushRelay(
        watch_dir=args.watch_dir,
        pattern=args.pattern,
        remote_host=args.remote_host,
        remote_dir=args.remote_dir,
        poll_interval=args.poll_interval,
        max_in_flight=args.max_in_flight,
        include_existing=args.include_existing,
    )
    try:
        relay.run_forever()
    except KeyboardInterrupt:
        print("\n[dir_push_relay] stopped by user")


if __name__ == "__main__":
    main()
