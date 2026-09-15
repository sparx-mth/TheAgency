"""Optional reusable AI2-THOR RGB-D bridge. No task goals or metric access.

The mirror of :mod:`..habitat.simulator` for the other simulator family:
teleport to the published start, execute the benchmark's discrete actions, and
hand back RGB, metric depth and a ground-truth pose already converted into the
repo's world ENU frame. It loads published scenes by name and never moves an
episode's start.

Import is GPU-free: ``ai2thor`` is imported inside :meth:`reset`, so the
lightweight test environment can import this module, and the tests can drive
the whole bridge through a fake controller.

**The frame conversion, and why each sign is what it is.** AI2-THOR is Y-up
(``x`` right, ``y`` up, ``z`` forward) and reports ``agent.rotation.y`` as a
compass bearing -- degrees *clockwise* from ``+z``. Two anchors fix the whole
mapping: at ``rotation.y == 0`` the agent faces ``+z``, and ``RotateRight``
*increases* ``rotation.y``. Sending ``+y`` to ENU north and ``+x`` to ENU east
gives a heading of ``90 - rotation.y`` degrees measured counter-clockwise from
``+x``, so ``RotateRight`` lowers the yaw (turns clockwise) and ``RotateLeft``
raises it -- which is what
:func:`~sparx_agency.core.planning.objnav.action_converter.transition.apply_action`
models and what ``kinematics.check_motion`` refuses to let drift. Nothing is
negated: swapping AI2-THOR's ``y`` and ``z`` is a single transposition, and it
is the transposition that fixes the handedness. A mirrored frame keeps every
path length, so ``cross_checks.py`` can never see it -- only the turn check
can, which is why :func:`thor_pose` is pinned by its own tests.

**Camera pitch is not flipped.** ``agent.cameraHorizon`` is already REP-103 --
positive looks down -- so it is converted from degrees and left alone. The
Habitat bridge negates its sensor pitch because habitat-sim signs it the other
way; copying that negation here would reintroduce the bug.

**Heights.** AI2-THOR reports the agent's *transform origin*, which sits a
body height above the floor (0.901 in every published RoboTHOR episode), while
:attr:`~sparx_agency.core.planning.objnav.types.pose.AgentPose.z` is the floor
under the agent. The bridge subtracts a declared ``origin_height_m`` to reach
the floor, and checks the camera's own offset against the declared mount
height on every reset rather than trusting a constant read out of a Unity
source tree that is not the pinned build.

Python 3.8 syntax.
"""
from __future__ import annotations

import math

import numpy as np

from sparx_agency.core.common.types import normalize_angle
from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.pose import AgentPose

#: ``DiscreteAction`` to the AI2-THOR action name, by name and never by index.
#: ``Stop`` is a NOOP on the LoCoBot (``base.Pass()``): it exists so that the
#: evaluator can tell the episode ended, and it leaves the pose untouched,
#: which is what the shared STOP contract requires.
THOR_ACTIONS = {
    DiscreteAction.STOP: "Stop",
    DiscreteAction.MOVE_FORWARD: "MoveAhead",
    DiscreteAction.TURN_LEFT: "RotateLeft",
    DiscreteAction.TURN_RIGHT: "RotateRight",
    DiscreteAction.LOOK_UP: "LookUp",
    DiscreteAction.LOOK_DOWN: "LookDown",
}


def thor_pose(position, rotation_y_deg, camera_horizon_deg,
              origin_height_m) -> AgentPose:
    """AI2-THOR (+Y up, bearing clockwise from +Z) -> right-handed Z-up ENU.

    Args:
        position: The agent's ``{"x", "y", "z"}`` transform origin, metres.
        rotation_y_deg: ``agent.rotation.y``, degrees clockwise from ``+z``.
        camera_horizon_deg: ``agent.cameraHorizon``, degrees, already positive
            downward.
        origin_height_m: How far the reported origin sits above the floor,
            metres, so that the pose's ``z`` is the floor.

    Returns:
        The pose in world ENU: ``x`` east, ``y`` north, ``z`` the floor,
        yaw counter-clockwise from ``+x``, pitch REP-103.

    Raises:
        EnvContractError: The agent is not upright -- a non-zero roll or
            pitch on the *body* means the pose is not a yaw-only ObjectNav
            pose and every map built from it would be skewed.
    """
    return AgentPose(
        x=float(position["x"]),
        y=float(position["z"]),
        z=float(position["y"]) - float(origin_height_m),
        yaw=normalize_angle(math.radians(90.0 - float(rotation_y_deg))),
        camera_pitch=math.radians(float(camera_horizon_deg)))


def check_upright(rotation) -> None:
    """Refuse a body rotation that is not yaw-only.

    Raises:
        EnvContractError: ``rotation.x`` or ``rotation.z`` is more than a
            tenth of a degree from level. AI2-THOR tilts a body that has been
            knocked over, and a tilted body makes every back-projected point
            wrong without changing a single path length.
    """
    for axis in ("x", "z"):
        angle = abs(normalize_angle(math.radians(float(rotation[axis]))))
        if angle > math.radians(0.1):
            raise EnvContractError(
                "ObjectNav base must be upright (yaw-only); AI2-THOR reports "
                "rotation.%s = %g deg" % (axis, float(rotation[axis])))


def metric_depth(raw, camera, *, ray_distance=False):
    """Decode an AI2-THOR depth frame into the repo's metric convention.

    ``event.depth_frame`` is a ``(H, W)`` float32 array already in metres,
    clipped by the Unity camera's near and far planes. This applies the
    repo's clip semantics (:mod:`~sparx_agency.core.planning.objnav.types.camera`):
    a reading at or inside ``min_depth_m`` becomes ``NaN``, one at or beyond
    ``max_depth_m`` becomes ``+inf``. Left as numbers they would read as real
    surfaces sitting exactly on the clip planes.

    Args:
        raw: The simulator's depth array.
        camera: The :class:`~sparx_agency.core.planning.objnav.types.camera.CameraSpec`
            the frame was captured with.
        ray_distance: Whether the simulator returns the length of the ray to
            the surface rather than optical-frame ``z``. AI2-THOR returns
            optical-frame ``z``, so this is ``False``: its depth shader is
            ``Linear01Depth(UNITY_SAMPLE_DEPTH(...))`` over Unity's
            ``_CameraDepthTexture``, which stores the perspective-projected
            eye-space ``z`` -- the perpendicular distance to the image plane,
            not the ray length. (Read from the pinned build's own
            ``ImageSynthesis/Shaders/Depth.shader``; ai2thor 5.0.0's differs
            only in how it packs the result.) The flag exists so that a build
            which ever does otherwise is corrected here, in one place, rather
            than bowing every back-projected surface outward from the image
            centre.

    Returns:
        A new ``(H, W)`` float32 array; the caller's is not modified.
    """
    depth = np.asarray(raw, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[..., 0]
    depth = depth.copy()
    if ray_distance:
        depth = depth * _planar_from_ray(camera).astype(np.float32)
    near = (depth <= camera.min_depth_m) | (depth <= 0)
    far = depth >= camera.max_depth_m
    depth[near] = np.nan
    depth[far] = np.inf
    return depth


def _planar_from_ray(camera):
    """Per-pixel ``cos`` between the ray and the optical axis, for ray depth."""
    k = camera.intrinsics
    u = (np.arange(k.width, dtype=np.float64) - k.cx) / k.fx
    v = (np.arange(k.height, dtype=np.float64) - k.cy) / k.fy
    return 1.0 / np.sqrt(1.0 + u[None, :] ** 2 + v[:, None] ** 2)


class AI2ThorRGBDSimulator:
    """One AI2-THOR controller behind reset/step/observe. No scoring, no goals.

    The controller's initialisation is the benchmark's, passed in whole by the
    adapter: this class does not invent a camera, a step size or a build. It
    keeps the controller alive across episodes in one scene and calls
    ``reset`` only when the scene changes, which is what makes a 1,800-episode
    sweep finish in hours rather than days.

    Args:
        camera: Shared CameraSpec; its intrinsics must agree with the
            controller's ``width``/``height`` and field of view.
        actions: Shared DiscreteActionSpec, mapped to AI2-THOR by name.
        radius_m: Embodiment radius, recorded and cross-checked, never used to
            re-derive the simulator's own collider.
        height_m: Physical body height, independent of the camera mount.
        origin_height_m: How far above the floor AI2-THOR reports the agent's
            position, metres.
        initialize: The controller's ``initialize`` block, verbatim from the
            benchmark configuration (``gridSize``, ``rotateStepDegrees``,
            ``visibilityDistance``, ``fieldOfView``, ``agentMode``, ...).
        commit_id: The Unity build to run. Explicit, never the package
            default: the build, not the pip version, fixes the physics,
            visibility and rendering.
        width: Frame width, pixels.
        height: Frame height, pixels.
        platform: An ``ai2thor.platform`` class (``CloudRendering``), or None
            to let ai2thor choose -- which on a PRIME-offload laptop silently
            picks the integrated GPU.
        ray_distance_depth: Passed to :func:`metric_depth`.
        far_plane_sentinel_m: What the build returns for a pixel with no
            geometry -- a finite number, typically ``far - near``, never
            ``inf``. Given, the camera's ``max_depth_m`` is checked to sit
            below it so that empty sky is clipped to ``+inf`` instead of
            mapped as a surface.
        camera_offset_tolerance_m: How far the live camera offset may differ
            from ``camera.height_m`` before a reset is refused.
        controller_factory: Builds the controller; the tests pass a fake. The
            default imports ``ai2thor`` lazily.

    Raises:
        ValueError: On a non-positive radius or height, or an action spec
            holding an action AI2-THOR does not name.
    """

    def __init__(self, camera, actions, radius_m, *, height_m,
                 origin_height_m, initialize, commit_id, width, height,
                 platform=None, ray_distance_depth=False,
                 far_plane_sentinel_m=None, camera_offset_tolerance_m=0.01,
                 controller_factory=None):
        for name, value in (("radius_m", radius_m), ("height_m", height_m)):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "Explicit positive finite %s is required, got %r"
                    % (name, value))
        if isinstance(origin_height_m, bool) or not math.isfinite(origin_height_m):
            raise ValueError("origin_height_m must be a finite number")
        missing = [a.name for a in actions.actions if a not in THOR_ACTIONS]
        if missing:
            raise ValueError(
                "AI2-THOR has no action named %s" % ", ".join(missing))
        if (camera.intrinsics.width, camera.intrinsics.height) != (width, height):
            raise ValueError(
                "Camera intrinsics are %dx%d but the controller renders %dx%d"
                % (camera.intrinsics.width, camera.intrinsics.height,
                   width, height))
        # AI2-THOR reports no geometry as a finite far-plane value, not inf or
        # NaN. Every such pixel must land in the camera's out-of-range band, or
        # the sky maps as a wall standing at the far plane.
        if far_plane_sentinel_m is not None and camera.max_depth_m >= far_plane_sentinel_m:
            raise ValueError(
                "The camera measures to %.2f m but AI2-THOR reports an empty "
                "pixel as %.2f m; a max_depth_m at or above the sentinel would "
                "map the sky as a surface"
                % (camera.max_depth_m, far_plane_sentinel_m))
        self.camera = camera
        self.actions = actions
        self.radius_m = radius_m
        self.height_m = height_m
        self.origin_height_m = origin_height_m
        self.initialize = dict(initialize)
        self.commit_id = commit_id
        self.width = width
        self.height = height
        self.platform = platform
        self.ray_distance_depth = ray_distance_depth
        self.far_plane_sentinel_m = far_plane_sentinel_m
        self.camera_offset_tolerance_m = camera_offset_tolerance_m
        self._controller_factory = controller_factory or _default_controller
        self._controller = None
        self._scene = None
        self._event = None

    def controller_kwargs(self):
        """Exactly what the controller is built with, for the run manifest."""
        kwargs = {"commit_id": self.commit_id, "width": self.width,
                  "height": self.height, "renderDepthImage": True}
        kwargs.update(self.initialize)
        if self.platform is not None:
            kwargs["platform"] = getattr(self.platform, "__name__", self.platform)
        return kwargs

    def reset(self, scene, position, rotation_y_deg, horizon_deg):
        """Load ``scene`` if needed and teleport ONLY to the published start.

        Mirrors ``robothor_challenge``'s reset: ``controller.reset(scene)``
        then one ``TeleportFull`` carrying the episode's own position,
        orientation and horizon. The start is never snapped or re-sampled.

        Raises:
            EnvContractError: The teleport failed, or the live camera mount
                offset disagrees with the declared one.
        """
        if self._controller is None:
            self._controller = self._controller_factory(self)
        if self._scene != scene:
            self._controller.reset(scene)
            self._scene = scene
        event = self._controller.step(action={
            "action": "TeleportFull",
            "x": float(position["x"]), "y": float(position["y"]),
            "z": float(position["z"]),
            "rotation": {"x": 0.0, "y": float(rotation_y_deg), "z": 0.0},
            "horizon": float(horizon_deg),
            "standing": True,
        })
        if not event.metadata.get("lastActionSuccess", False):
            raise EnvContractError(
                "Could not teleport to the published start of %s: %s"
                % (scene, event.metadata.get("errorMessage", "")))
        self._event = event
        self._check_camera_mount(event)
        return self.observe()

    def _check_camera_mount(self, event):
        """Refuse a build whose camera does not sit where the protocol says."""
        camera_y = event.metadata.get("cameraPosition", {}).get("y")
        if camera_y is None:
            return
        offset = float(camera_y) - (float(event.metadata["agent"]["position"]["y"])
                                    - self.origin_height_m)
        if abs(offset - self.camera.height_m) > self.camera_offset_tolerance_m:
            raise EnvContractError(
                "This build mounts the camera %.4f m above the floor but the "
                "protocol declares %.4f m; every back-projected point would be "
                "offset by the difference" % (offset, self.camera.height_m))

    def step(self, action):
        """Execute one benchmark action; a refused one still returns a frame.

        AI2-THOR reports a collision or a refused LOOK as
        ``lastActionSuccess == False`` and leaves the agent where it was. That
        is not an error here: the benchmark spends the step either way, and
        the harness's own pose comparison is what notices a blocked move.

        Raises:
            EnvContractError: The simulator was never reset, or the action is
                not one the episode allows.
        """
        if self._controller is None or not self.actions.allows(action):
            raise EnvContractError("Simulator not reset, or unsupported action")
        self._event = self._controller.step(action=THOR_ACTIONS[action])
        return self.observe()

    @property
    def last_action_succeeded(self) -> bool:
        """Whether AI2-THOR executed the last action, for evaluator telemetry."""
        if self._event is None:
            return False
        return bool(self._event.metadata.get("lastActionSuccess", False))

    @property
    def controller(self):
        """The live controller, for evaluator-only navmesh queries.

        ``GetShortestPath`` is issued as a ``controller.step`` and replaces the
        controller's ``last_event``. Nothing here reads ``last_event`` -- the
        bridge holds its own reference to the frame it last rendered -- so a
        distance query can never become the agent's observation.

        Raises:
            EnvContractError: The simulator was never reset.
        """
        if self._controller is None:
            raise EnvContractError("Simulator not reset")
        return self._controller

    @property
    def metadata(self):
        """The last event's raw metadata. Evaluator-only; never an observation."""
        if self._event is None:
            raise EnvContractError("Simulator not reset")
        return self._event.metadata

    def observe(self):
        """``(rgb, depth_m, pose)`` from the last event, in the repo's frames."""
        event = self._event
        agent = event.metadata["agent"]
        check_upright(agent["rotation"])
        pose = thor_pose(agent["position"], agent["rotation"]["y"],
                         agent["cameraHorizon"], self.origin_height_m)
        rgb = np.asarray(event.frame)[..., :3].copy()
        if rgb.dtype != np.uint8:
            raise EnvContractError(
                "AI2-THOR returned a %s frame; RGB must be uint8" % rgb.dtype)
        depth = event.depth_frame
        if depth is None:
            raise EnvContractError(
                "No depth frame: build the controller with renderDepthImage=True")
        return rgb, metric_depth(depth, self.camera,
                                 ray_distance=self.ray_distance_depth), pose

    def close(self):
        """Release the Unity process before another scene or run takes the GPU."""
        if self._controller is not None:
            self._controller.stop()
        self._controller = None
        self._scene = None
        self._event = None


def _default_controller(bridge):
    """Build the real controller. Imported here so the module stays GPU-free."""
    import ai2thor.controller

    kwargs = {"commit_id": bridge.commit_id, "width": bridge.width,
              "height": bridge.height, "renderDepthImage": True}
    kwargs.update(bridge.initialize)
    if bridge.platform is not None:
        kwargs["platform"] = bridge.platform
    return ai2thor.controller.Controller(**kwargs)
