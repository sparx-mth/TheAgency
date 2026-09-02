"""The Rooster's measured horizontal stick-response curve (Sphera, ManualControl).

One curve serves forward, backward and lateral: the calibration campaign of
2026-08-25..31 (eight hand-flown sessions, 20-46 s holds, walls avoided; logs
under ``runs/manual_axiscal_*``) measured the three directions identical within
~3% at every shared level. Points are pooled per-level medians over all 69
settled straight holds, cross-checked by two independent re-derivations.

Shape: a transmitter-style expo curve (~counts^3.5) with NO dead band -- 250
counts still moves the aircraft -- rolling off to ~linear growth above ~750
counts. The last point, 900 counts -> 1.566 m/s, is treated as the platform
ceiling: 900 is also where the airframe stops behaving (pitch p90 jumps past
20 deg beyond it), so nothing may command more counts than that.

Yaw is deliberately NOT here: it follows a genuinely different law (linear with
a real ~100-count dead band, rate = 0.00285*(counts-100)), measured equally
well and already encoded by ``feedforward_axis`` in the twist adapter.
"""

from __future__ import annotations

from sparx_agency.core.control.axis_response_curve import AxisResponseCurve

#: Pooled per-level medians, (axis counts, steady-state m/s).
ROOSTER_HORIZONTAL_POINTS = [
    (0, 0.0),
    (250, 0.0261),
    (300, 0.0404),
    (350, 0.0636),
    (400, 0.0993),
    (450, 0.1509),
    (500, 0.2205),
    (550, 0.3057),
    (600, 0.4280),
    (650, 0.5905),
    (700, 0.7922),
    (750, 1.0247),
    (800, 1.1882),
    (850, 1.3724),
    (900, 1.5659),
]

#: The curve both horizontal axes fly and the analyzer reads logs back through.
ROOSTER_HORIZONTAL_CURVE = AxisResponseCurve(ROOSTER_HORIZONTAL_POINTS)
