"""The recorder's pure message->record conversion, without a ROS master.

``nav_debug_sources`` is deliberately ROS-free so it can be exercised headless.
The case that matters here is the executed path: FALCON's ``traj_server`` appends
a point per 100 Hz tick for the whole flight and republishes the WHOLE vector,
so an unbounded copy of it, re-serialized into every route snapshot, made the
recording grow with the square of the flight length.
"""
import pathlib
import sys
import types

_SCRIPTS = pathlib.Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import nav_debug_sources as sources        # noqa: E402


def _path(n):
    """A ``nav_msgs/Path``-shaped stub with ``n`` poses along +x."""
    def pose(i):
        position = types.SimpleNamespace(x=float(i), y=0.0, z=0.0)
        return types.SimpleNamespace(pose=types.SimpleNamespace(position=position))
    return types.SimpleNamespace(poses=[pose(i) for i in range(n)])


def test_path_is_verbatim_when_under_the_cap():
    assert sources.path_xy(_path(10), 600) == [[float(i), 0.0] for i in range(10)]


def test_path_is_verbatim_when_uncapped():
    assert len(sources.path_xy(_path(5000), 0)) == 5000


def test_long_path_is_decimated_to_the_cap():
    """A flight-long path must cost O(1) per snapshot, not O(flight)."""
    for n in (12_000, 120_000):
        out = sources.path_xy(_path(n), 600)
        assert len(out) <= 601, "n=%d gave %d points" % (n, len(out))


def test_decimation_keeps_the_true_endpoints():
    """The head and the tip must be exact: the tip is where the aircraft is."""
    out = sources.path_xy(_path(12_000), 600)
    assert out[0] == [0.0, 0.0]
    assert out[-1] == [11_999.0, 0.0]
