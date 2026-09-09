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


def test_control_row_passes_every_trace_section_through_untouched():
    """The recorder must not whitelist sections, or a new one silently vanishes.

    The follower shapes its own trace (``reference`` / ``state`` / ``tracking`` /
    ``terms`` / ``command`` / ``command_requested`` / ``gate``); the recorder only
    lifts out the two clocks. ``state`` is the newest of those and reaches the
    replay's reference-vs-actual table through this function.
    """
    import json

    payload = {
        "t": 12.0, "wall": 34.0,
        "state": {"x": 1.0, "y": 2.0, "z": 1.5, "yaw": 0.4,
                  "vx": 0.7, "vy": -0.1, "vz": 0.0, "wz": 0.02},
        "command": {"vx": 0.6, "vy": 0.0, "wz": 0.1, "vz": 0.0},
        "command_requested": {"vx": 0.63, "vy": 0.0, "wz": 0.1, "vz": 0.0},
        "tracking": None, "gate": {"reason": "driving"},
    }
    t, wall, fields = sources.control_row(json.dumps(payload), 99.0, 98.0)

    assert (t, wall) == (12.0, 34.0)          # the publisher's stamps, not ours
    assert fields["state"] == payload["state"]
    assert fields["command"] == payload["command"]
    assert fields["command_requested"] == payload["command_requested"]
    # An explicit null must survive as null: it means "the tracker did not run",
    # which is not the same as "this section was never written".
    assert "tracking" in fields and fields["tracking"] is None


def test_the_state_section_matches_the_replays_dataclass():
    """The writer's key names are the reader's field names -- that is the contract.

    ``records.build`` is reflective, so a key the follower renames is not an
    error anywhere: it is silently dropped and the panel quietly reads '--'.
    """
    from sparx_agency.tasks.planning.nav_debug import records
    from sparx_agency.tasks.planning.nav_debug.frame import DroneState

    emitted = {"x": 1.0, "y": 2.0, "z": 1.5, "yaw": 0.4,
               "vx": 0.7, "vy": -0.1, "vz": 0.0, "wz": 0.02}
    state = records.build(DroneState, emitted)
    assert state is not None
    for key, value in emitted.items():
        assert getattr(state, key) == value, key
