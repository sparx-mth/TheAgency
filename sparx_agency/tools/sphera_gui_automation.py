"""xdotool-driven GUI automation for re-entering a Sphera scenario.

Sphera has no CLI/service hook past the container restart (confirmed live,
`LESSONS.md`'s 2026-08-13 entry) — getting a flyable drone back after
``sphera-restart.sh`` requires a fixed sequence of clicks inside its own
(Unreal) window: Continue -> choose role (Manager) -> Start -> Standalone ->
assign the drone (Rooster_1) -> Play. This module drives that sequence with
``xdotool``, using proportional (fraction-of-window) coordinates measured
against the window's live geometry, not fixed screen pixels — robust to
Sphera's window sitting at a different position/size than when these
fractions were measured, without needing a fresh screenshot per click.

Assumes (2026-08-17, by explicit user request) that Sphera's window is not
moved/resized and nothing else is driving its UI concurrently while this
runs. The fallback for "something changed anyway" is structural, not
pixel-level: this module polls for the window to exist before clicking, and
the caller (``sphera_battery_watchdog.py``) confirms success afterward via
``docker ps``/battery telemetry, not by reading pixels — a wrong click (e.g.
an unexpected dialog swallowing it) shows up as "R1 never appeared" rather
than a silent false success.
"""

from __future__ import annotations

import subprocess
import time

SPHERA_WINDOW_NAME = "Sphera"

# Click points as (fraction_x, fraction_y) of the window's current geometry.
# Measured against a live 1920x1043 Sphera window, confirmed stable across
# 3 separate reboots (2026-08-17) before being hardcoded here.
_CONTINUE = (0.4974, 0.6309)          # "Welcome to SPHERA" -> Continue
_MANAGER_ROLE = (0.2417, 0.4506)      # "Choose your role" -> Manager hexagon
_START_MENU = (0.4776, 0.2867)        # "Manager mode" -> Start
_STANDALONE_MENU = (0.4995, 0.3634)   # "Manager mode" -> Standalone
_ROOSTER1_ROW = (0.5026, 0.1735)      # Drones Assignment panel -> Rooster_1 row
_PLAY_BUTTON = (0.5974, 0.0508)       # green Play button, top bar

# Seconds to wait after the window first appears before clicking anything --
# the X window exists well before Sphera is actually rendering/interactive
# (see enter_scenario()). Empirically tuned (2026-08-17), not a documented
# constant from Sphera itself -- raise it if re-entry keeps failing on slow
# hardware.
_WINDOW_WARMUP_SEC = 10.0


def _run(*args: str, timeout: float = 5.0) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)


def find_content_window() -> int | None:
    """Returns Sphera's actual content window id, or None if not found.

    A plain `xdotool search --name Sphera` also matches anything else with
    "Sphera" in its title (e.g. a browser tab) — filtered out here by
    requiring an exact (whitespace-trimmed) name match. Among the exact
    matches, Sphera's window manager always reports two: an outer decorated
    frame and its inner content. The content window is reliably the
    smaller-by-area one (confirmed live across 3 reboots) — the outer frame
    is always slightly larger by its title bar/border chrome.
    """
    result = _run("xdotool", "search", "--name", SPHERA_WINDOW_NAME)
    if result.returncode != 0:
        return None
    candidates = []
    for wid in result.stdout.split():
        name = _run("xdotool", "getwindowname", wid).stdout.strip()
        if name != SPHERA_WINDOW_NAME:
            continue
        geom = _run("xdotool", "getwindowgeometry", "--shell", wid).stdout
        dims = dict(line.split("=", 1) for line in geom.splitlines() if "=" in line)
        try:
            area = int(dims["WIDTH"]) * int(dims["HEIGHT"])
        except (KeyError, ValueError):
            continue
        candidates.append((area, wid))
    if not candidates:
        return None
    candidates.sort()
    return int(candidates[0][1])


def wait_for_window(timeout_sec: float = 90.0, poll_sec: float = 1.0) -> int | None:
    """Polls `find_content_window()` until found or `timeout_sec` elapses —
    faster than a fixed sleep when Sphera boots quickly, and more reliable
    than one when it boots slowly."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        wid = find_content_window()
        if wid is not None:
            return wid
        time.sleep(poll_sec)
    return find_content_window()


#: How long a found window may report no geometry before we call it gone.
#: It is a settling race, not a missing window -- see ``_click_fraction``.
GEOMETRY_GRACE_SEC = 20.0
GEOMETRY_POLL_SEC = 0.5


class ClickTargetGone(RuntimeError):
    """The window we were about to click no longer has a geometry."""


def _click_fraction(window_id: int, fx: float, fy: float) -> None:
    """Clicks a point given as a fraction of the window's CURRENT geometry
    (queried fresh each call — one cheap `xdotool` call, not cached), so
    this adapts to minor size differences instead of assuming a fixed pixel
    position."""
    # Tolerate a briefly-unmapped window before giving up. `find_content_window`
    # only returns ids that HAD geometry, so an empty answer here is a
    # time-of-check/time-of-use race: Sphera is still settling and remaps its
    # window under us. Measured 2026-08-22 on cycle 386 -- the window was found
    # in 28 s, reported no geometry, and the whole re-entry was abandoned; the
    # campaign then spent eleven minutes restarting Sphera to reach a state it
    # was already moments away from. Polling costs nothing when the window is
    # healthy, because the first read succeeds.
    deadline = time.time() + GEOMETRY_GRACE_SEC
    while True:
        geom = _run("xdotool", "getwindowgeometry", "--shell",
                    str(window_id)).stdout
        dims = dict(line.split("=", 1) for line in geom.splitlines() if "=" in line)
        if "WIDTH" in dims and "HEIGHT" in dims:
            break
        if time.time() >= deadline:
            break
        time.sleep(GEOMETRY_POLL_SEC)
    if "WIDTH" not in dims or "HEIGHT" not in dims:
        # A window id can outlive the window, and one that is unmapped answers
        # with no geometry at all. Raising ClickTargetGone lets enter_scenario
        # return False and the caller retry, which is what it already does for a
        # window that never appeared; the bare KeyError instead escaped as a
        # traceback and stalled the campaign for eleven minutes on 2026-08-20.
        raise ClickTargetGone(
            "window %s still reports no geometry (%r) after %.0fs -- it is gone, "
            "not merely unmapped" % (window_id, geom.strip()[:120],
                                     GEOMETRY_GRACE_SEC))
    x = round(fx * int(dims["WIDTH"]))
    y = round(fy * int(dims["HEIGHT"]))
    _run("xdotool", "mousemove", "--window", str(window_id), str(x), str(y))
    _run("xdotool", "click", "1")


def enter_scenario(window_wait_sec: float = 90.0) -> bool:
    """Drives Continue -> Manager -> Start -> Standalone -> select Rooster_1
    -> Play. Returns False (never raises) if Sphera's window doesn't
    reappear within `window_wait_sec` — this module has no battery/docker
    knowledge of its own (see module docstring); the caller decides how to
    react and how to verify the click sequence actually worked.
    """
    window_id = wait_for_window(window_wait_sec)
    if window_id is None:
        print(f"[gui] Sphera window did not reappear within {window_wait_sec:.0f}s", flush=True)
        return False

    try:
        return _enter_scenario_clicks(window_id)
    except ClickTargetGone as exc:
        # Contract of this function: never raise. The caller retries.
        print(f"[gui] {exc}", flush=True)
        return False


def _enter_scenario_clicks(window_id: int) -> bool:
    """The click sequence itself, once a window has been found."""
    print(f"[gui] found Sphera content window {window_id}, entering scenario", flush=True)
    _run("xdotool", "windowactivate", str(window_id))

    # The X window exists well before Sphera/Unreal is actually rendering
    # and accepting input -- confirmed live (2026-08-17): firing the click
    # sequence right after the window appears landed every click on a still-
    # loading "Welcome" screen, and the whole run silently failed (stuck on
    # screen 1, R1 never spawned). This warmup, plus a redundant idempotent
    # second Continue click below (harmless no-op if the first already
    # landed -- it hits empty background on the next screen), replaces that
    # failure with a normal successful run.
    time.sleep(_WINDOW_WARMUP_SEC)

    _click_fraction(window_id, *_CONTINUE)
    time.sleep(1.0)
    _click_fraction(window_id, *_CONTINUE)  # idempotent safety net, see above
    time.sleep(0.6)  # "Choose your role" is a plain menu, renders fast

    _click_fraction(window_id, *_MANAGER_ROLE)
    time.sleep(1.0)  # "Manager mode" also fetches the scenario list + thumbnail

    _click_fraction(window_id, *_START_MENU)
    time.sleep(0.3)  # same screen, just a selection highlight

    _click_fraction(window_id, *_STANDALONE_MENU)
    time.sleep(1.5)  # heaviest transition: loads the actual 3D scenario level

    _click_fraction(window_id, *_ROOSTER1_ROW)
    time.sleep(0.4)  # same screen, just updates the assignment label

    _click_fraction(window_id, *_PLAY_BUTTON)
    print("[gui] Play clicked", flush=True)
    return True
