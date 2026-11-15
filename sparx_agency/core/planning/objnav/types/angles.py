"""The largest angle the ObjectNav types accept: a finite angle can still hang the process.

Every angle in this layer is wrapped sooner or later with
``core.common.types.normalize_angle``, a loop that adds or subtracts one full
turn until the angle lies in ``(-pi, pi]``. It costs one iteration per turn: a
yaw of 1e9 rad takes about 1.6e8 iterations per call. From about 7e16 rad on,
one full turn is below the float spacing, so subtracting it leaves the value
unchanged and the loop never ends. A mis-scaled or never-wrapped simulator
yaw, or a policy's garbage heading, would then hang a benchmark sweep instead
of raising. So the pose and the command refuse any angle beyond
:data:`MAX_ANGLE_RAD`, and the message says to wrap it where it is made.

Python 3.8 syntax, standard library only.
"""

#: The largest magnitude an angle field may have, radians. About 160,000 full
#: turns: far beyond anything real (a 500-step episode of 30-degree turns
#: accumulates at most 262 rad), and still a few milliseconds to wrap.
MAX_ANGLE_RAD = 1e6
