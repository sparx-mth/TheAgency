"""A controller that moves the way AI2-THOR moves, in AI2-THOR's own frame.

The point of this fixture is that it knows nothing about ENU. It keeps the
agent in AI2-THOR's Y-up, bearing-clockwise-from-+Z frame and applies the
LoCoBot's own rules -- ``MoveAhead`` along ``(sin, cos)`` of the bearing,
``RotateRight`` *increasing* the bearing, a ``LookUp``/``LookDown`` that fails
outright at +/-30 degrees and spends the step without moving the camera. So a
frame conversion that is mirrored, transposed the wrong way or signed the
wrong way round comes out as a motion the harness's own
:func:`~sparx_agency.tasks.planning.objnav_benchmark.kinematics.check_motion`
refuses, which is the only thing that can catch it: a mirrored frame keeps
every path length, so the scoring cross-checks never see it.

No ``ai2thor`` import, no Unity, no GPU.

Python 3.8 syntax.
"""
from __future__ import annotations

import math

import numpy as np

#: The LoCoBot's hard horizon clamp; a LOOK past it is refused, not clipped.
MIN_HORIZON_DEG = -30.0
MAX_HORIZON_DEG = 30.0


class FakeEvent:
    """What ``controller.step`` returns: frames plus the metadata dict."""

    def __init__(self, metadata, frame, depth_frame):
        self.metadata = metadata
        self.frame = frame
        self.depth_frame = depth_frame


class FakeThorController:
    """AI2-THOR's kinematics and metadata, without AI2-THOR.

    Args:
        width: Frame width, pixels.
        height: Frame height, pixels.
        grid_size: What ``MoveAhead`` advances, metres.
        rotate_step_degrees: What one rotation turns, degrees.
        camera_offset_m: How far the camera sits below the reported origin.
        origin_height_m: How far the reported origin sits above the floor.
        blocked: Bearings-and-positions predicate ``(x, z) -> bool`` marking
            where a ``MoveAhead`` fails. The default never blocks.
        depth_value: Constant metric depth to render.
    """

    def __init__(self, width=64, height=48, grid_size=0.25,
                 rotate_step_degrees=30.0, camera_offset_m=0.0312,
                 origin_height_m=0.9, blocked=None, depth_value=2.0):
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.rotate_step_degrees = rotate_step_degrees
        self.camera_offset_m = camera_offset_m
        self.origin_height_m = origin_height_m
        self.blocked = blocked or (lambda x, z: False)
        self.depth_value = depth_value
        self.scene = None
        self.stopped = False
        self.scene_resets = 0
        self.position = {"x": 0.0, "y": origin_height_m, "z": 0.0}
        self.rotation = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.horizon = 0.0
        self.last_action_success = True
        self.objects = []
        #: ``allowed_error -> bool``: which rungs of the tolerance ladder the
        #: navmesh refuses. The default navmesh always finds a path.
        self.navmesh_unreachable = lambda allowed_error: False

    def reset(self, scene):
        self.scene = scene
        self.scene_resets += 1
        return self._event()

    def stop(self):
        self.stopped = True

    def step(self, action=None, **kwargs):
        """Execute one AI2-THOR action, named the way AI2-THOR names it."""
        if isinstance(action, dict):
            name = action["action"]
            if name == "GetShortestPath":
                return self._shortest_path(action)
            if name != "TeleportFull":
                raise AssertionError("unexpected dict action %r" % (name,))
            self.position = {"x": float(action["x"]), "y": float(action["y"]),
                             "z": float(action["z"])}
            self.rotation = dict(action["rotation"])
            self.horizon = float(action["horizon"])
            self.last_action_success = True
            return self._event()
        self.last_action_success = True
        if action == "MoveAhead":
            bearing = math.radians(self.rotation["y"])
            x = self.position["x"] + self.grid_size * math.sin(bearing)
            z = self.position["z"] + self.grid_size * math.cos(bearing)
            if self.blocked(x, z):
                self.last_action_success = False
            else:
                self.position = dict(self.position, x=x, z=z)
        elif action in ("RotateLeft", "RotateRight"):
            sign = -1.0 if action == "RotateLeft" else 1.0
            self.rotation = dict(
                self.rotation,
                y=(self.rotation["y"] + sign * self.rotate_step_degrees) % 360.0)
        elif action in ("LookUp", "LookDown"):
            sign = -1.0 if action == "LookUp" else 1.0
            horizon = self.horizon + sign * self.rotate_step_degrees
            if horizon < MIN_HORIZON_DEG - 1e-9 or horizon > MAX_HORIZON_DEG + 1e-9:
                self.last_action_success = False
            else:
                self.horizon = horizon
        elif action == "Stop":
            pass
        else:
            raise AssertionError("unknown AI2-THOR action %r" % (action,))
        return self._event()

    def place_object(self, object_type, position, visible=False, distance=9.0):
        """Add one goal-shaped object to the metadata the evaluator reads."""
        self.objects.append({
            "objectId": "%s|%+.2f|%+.2f|%+.2f"
                        % (object_type, position["x"], position["y"], position["z"]),
            "objectType": object_type, "position": dict(position),
            "visible": visible, "distance": distance})

    def _shortest_path(self, query):
        """Answer ``GetShortestPath`` with a straight run to the nearest instance.

        A query is a ``controller.step`` like any other: it returns an event
        and replaces ``last_event``, but it must not move the agent. The
        ``navmesh_unreachable`` predicate lets a test exercise the tolerance
        ladder and the unreachable branch.
        """
        start = dict(query["position"])
        goals = [o["position"] for o in self.objects
                 if o["objectId"].split("|")[0] == query["objectType"]]
        event = self._event()
        allowed = float(query.get("allowedError", 0.0))
        if not goals or self.navmesh_unreachable(allowed):
            event.metadata["lastActionSuccess"] = False
            event.metadata["errorMessage"] = "Could not find a path"
            return event
        nearest = min(goals, key=lambda g: (g["x"] - start["x"]) ** 2
                      + (g["z"] - start["z"]) ** 2)
        event.metadata["lastActionSuccess"] = True
        event.metadata["actionReturn"] = {"corners": [
            {"x": start["x"], "y": 0.0103, "z": start["z"]},
            {"x": nearest["x"], "y": 0.0103, "z": nearest["z"]}]}
        return event

    def _event(self):
        frame = np.zeros((self.height, self.width, 3), np.uint8)
        depth = np.full((self.height, self.width), self.depth_value, np.float32)
        metadata = {
            "agent": {"position": dict(self.position),
                      "rotation": dict(self.rotation),
                      "cameraHorizon": self.horizon},
            "cameraPosition": dict(self.position,
                                   y=self.position["y"] - self.camera_offset_m),
            "lastActionSuccess": self.last_action_success,
            "errorMessage": "",
            "objects": [dict(o) for o in self.objects],
            "sceneName": self.scene,
        }
        return FakeEvent(metadata, frame, depth)
