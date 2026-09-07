#!/usr/bin/env python3
"""Direct MAVLink acceptance check for Robotican's new Orin-NX architecture.

The new architecture wires the Orin NX to the flight controller over a
direct USB/MAVLink link (confirmed from Robotican's block diagram). This
script talks to that link DIRECTLY via pymavlink -- it does not go through
ROS2, rclpy, or any of Robotican's own bridge software (fcu_driver /
rooster_manager / rooster_handler). That's deliberate: it isolates the
physical link + MAVLink protocol itself from their ROS2 bridge layer. If
this script passes but orin_nx_flight_interface_test.py's ROS2-level checks
fail, that tells you the link is fine and the bug is in their bridge
software, not the hardware -- a real diagnostic distinction the ROS2-only
test can't make on its own.

Unlike the ROS2-layer StatusText check (whose topic name is an unconfirmed
guess -- Robotican's own custom message, not a standard one), STATUSTEXT
here is the real, standard MAVLink message (ID 253, MAV_SEVERITY has 8
levels: EMERGENCY/ALERT/CRITICAL/ERROR/WARNING/NOTICE/INFO/DEBUG) that
every MAVLink-based autopilot sends -- nothing to confirm with the vendor,
it's part of the protocol itself. Same for the IMU messages checked here
(RAW_IMU / SCALED_IMU* / HIGHRES_IMU) and HEARTBEAT.

Checks, in order:
  1. HEARTBEAT arrives at all -- is anything speaking MAVLink on this link.
  2. An IMU-family message arrives at a sane rate.
  3. STATUSTEXT is observed over the run (informational -- event-driven,
     so silence during a clean run is the expected good outcome, not a
     failure; any WARNING/ERROR/CRITICAL/etc. severities are surfaced).

Requires: pymavlink (`pip install pymavlink`). Not part of this repo's
package -- like opencv/tensorrt for the other acceptance scripts, this is
a third-party dependency, not something from sparx_agency.

Usage:
    python3 orin_nx_mavlink_direct_test.py --connection /dev/ttyACM0
    # or, if it's presented as a network endpoint instead of serial:
    python3 orin_nx_mavlink_direct_test.py --connection udpin:0.0.0.0:14550
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

IMU_MESSAGE_TYPES = {"RAW_IMU", "SCALED_IMU", "SCALED_IMU2", "SCALED_IMU3", "HIGHRES_IMU"}

# MAV_SEVERITY, for readable output -- see module docstring.
SEVERITY_NAMES = {
    0: "EMERGENCY", 1: "ALERT", 2: "CRITICAL", 3: "ERROR",
    4: "WARNING", 5: "NOTICE", 6: "INFO", 7: "DEBUG",
}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def decode_maybe_bytes(value) -> str:
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)


def run(args: argparse.Namespace) -> int:
    try:
        from pymavlink import mavutil
    except ImportError:
        print("[mavlink-direct-test] FAIL: pymavlink not installed (`pip install pymavlink`).")
        return 1

    print(f"[mavlink-direct-test] connection : {args.connection}")
    print(f"[mavlink-direct-test] baud       : {args.baud}")

    checks: list[Check] = []

    try:
        conn = mavutil.mavlink_connection(args.connection, baud=args.baud)
    except Exception as e:
        print(f"[mavlink-direct-test] FAIL: could not open connection: {e}")
        return 1

    print(f"[mavlink-direct-test] waiting up to {args.heartbeat_timeout_sec}s for HEARTBEAT...")
    hb = conn.wait_heartbeat(timeout=args.heartbeat_timeout_sec)
    checks.append(Check(
        "heartbeat_received",
        hb is not None,
        f"system={conn.target_system} component={conn.target_component}" if hb is not None
        else f"no HEARTBEAT within {args.heartbeat_timeout_sec}s -- nothing is speaking "
             f"MAVLink on {args.connection}",
    ))
    if hb is None:
        return report(checks)

    imu_count = 0
    imu_first_t = None
    imu_last_t = None
    status_texts: list[tuple[int, str]] = []

    print(f"[mavlink-direct-test] observing for {args.observe_sec}s...")
    deadline = time.monotonic() + args.observe_sec
    while time.monotonic() < deadline:
        msg = conn.recv_match(blocking=True, timeout=0.2)
        if msg is None:
            continue
        kind = msg.get_type()
        if kind in IMU_MESSAGE_TYPES:
            now = time.monotonic()
            if imu_first_t is None:
                imu_first_t = now
            imu_last_t = now
            imu_count += 1
        elif kind == "STATUSTEXT":
            severity = int(getattr(msg, "severity", -1))
            text = decode_maybe_bytes(getattr(msg, "text", ""))
            status_texts.append((severity, text))
            label = SEVERITY_NAMES.get(severity, str(severity))
            print(f"[mavlink-direct-test] STATUSTEXT [{label}] {text}")

    imu_rate = 0.0
    if imu_count > 1 and imu_first_t is not None and imu_last_t is not None:
        elapsed = imu_last_t - imu_first_t
        imu_rate = (imu_count - 1) / elapsed if elapsed > 0 else 0.0
    imu_ok = imu_count > 0 and imu_rate >= args.min_imu_hz
    checks.append(Check(
        "mav_imu_rate",
        imu_ok,
        f"count={imu_count} rate={imu_rate:.1f}Hz (min {args.min_imu_hz}Hz), "
        f"types seen from {IMU_MESSAGE_TYPES}",
    ))

    by_severity: dict[int, int] = {}
    for sev, _text in status_texts:
        by_severity[sev] = by_severity.get(sev, 0) + 1
    severe = [t for sev, t in status_texts if sev <= 3]  # EMERGENCY..ERROR
    detail = f"{len(status_texts)} STATUSTEXT message(s) over the run"
    if by_severity:
        detail += " (" + ", ".join(
            f"{SEVERITY_NAMES.get(sev, sev)}={n}" for sev, n in sorted(by_severity.items())
        ) + ")"
    if severe:
        detail += f" -- SEVERE: {'; '.join(severe[:5])}"
    checks.append(Check("mav_statustext_channel", True, detail))

    return report(checks)


def report(checks: list[Check]) -> int:
    print("\n[mavlink-direct-test] ---- results ----")
    all_passed = True
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        all_passed &= c.passed
        print(f"[mavlink-direct-test] {status:4s} {c.name:24s} {c.detail}")
    print(f"[mavlink-direct-test] {'PASS' if all_passed else 'FAIL'}")
    return 0 if all_passed else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Direct MAVLink check (no ROS2, no Robotican bridge software): "
        "HEARTBEAT, IMU rate, STATUSTEXT, straight from the FCU."
    )
    p.add_argument("--connection", required=True,
                    help="pymavlink connection string, e.g. /dev/ttyACM0 (serial) or "
                         "udpin:0.0.0.0:14550 (network) -- confirm the right one with Robotican, "
                         "this is not something we can infer.")
    p.add_argument("--baud", type=int, default=57600,
                    help="Serial baud rate (ignored for network connections).")
    p.add_argument("--heartbeat-timeout-sec", type=float, default=10.0)
    p.add_argument("--observe-sec", type=float, default=15.0,
                    help="How long to listen for IMU/STATUSTEXT messages after the heartbeat.")
    p.add_argument("--min-imu-hz", type=float, default=5.0)
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(run(parse_args()))
