"""Shared BGR colour palette for the planning HUDs.

One place for the colours both the object-approach HUD and the nav-debug view
use, so a green means the same green in both. All values are OpenCV **BGR**
tuples (not RGB). Kept as plain module constants -- a palette is data, not
behaviour.
"""
from __future__ import annotations

# ── neutral chrome ───────────────────────────────────────────────────────────
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
PANEL_BG = (24, 24, 24)         # dark side-panel background
TEXT = (235, 235, 235)          # primary panel text
MUTED = (150, 150, 150)         # secondary / label text
HLINE = (60, 60, 60)            # thin section divider

# ── status semantics (shared by the lock indicator and the nav banner) ───────
GREEN = (60, 200, 60)           # good / confident / detector sees it
ORANGE = (0, 140, 255)          # working / tracking-only / approaching
RED = (40, 40, 220)             # lost / blocked / re-searching
GRAY = (150, 150, 150)          # searching / idle / context
AMBER = (0, 215, 255)           # terminal action (land / reached)
CYAN = (250, 190, 60)           # secondary command channel (converter -> drone)
