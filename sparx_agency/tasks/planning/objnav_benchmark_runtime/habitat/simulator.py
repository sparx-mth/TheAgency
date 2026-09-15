"""Optional reusable habitat-sim RGB-D bridge. No task goals or metric access.

Uses existing navmeshes, never recomputes them or snaps episode starts. Import
is GPU-free; a rendering context is created only by reset(). The same bridge
can serve Gibson, HM3D and MP3D behind their separate dataset evaluators.
"""
from __future__ import annotations

import math

import numpy as np

from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.pose import AgentPose


def habitat_pose(position, body_rotation, camera_rotation) -> AgentPose:
    """Habitat (+Y up, -Z forward) -> right-handed Z-up ENU.

    Public (x,y,z)=(-habitat_z,-habitat_x,habitat_y). A positive Habitat
    rotation about +Y turns left in both frames. Rotation matrices are
    world-from-local; the camera's optical forward axis is local -Z.
    """
    base_forward = -np.asarray(body_rotation)[:, 2]
    camera_forward = -np.asarray(camera_rotation)[:, 2]
    if abs(float(base_forward[1])) > 1e-4:
        raise EnvContractError("ObjectNav base must be upright (yaw-only)")
    return AgentPose(
        x=-float(position[2]), y=-float(position[0]), z=float(position[1]),
        yaw=math.atan2(-base_forward[0], -base_forward[2]),
        camera_pitch=math.atan2(-camera_forward[1],
                                math.hypot(camera_forward[0], camera_forward[2])))


def habitat_position(pose):
    """The exact inverse of :func:`habitat_pose`'s translation: ENU -> Habitat XYZ.

    Adapters need it to ask the pathfinder about the agent's current place
    without a second source of truth for the frame. Getting one sign wrong
    here moves every distance-to-goal without failing anything, so it is
    written once, beside the forward conversion.
    """
    return (-pose.y, pose.z, -pose.x)


def metric_depth(raw, camera, normalized=False):
    """Decode Lab normalized depth or raw Sim metres; preserve clip semantics."""
    depth = np.asarray(raw, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[..., 0]
    depth = depth.copy()
    if normalized:
        depth = depth * (camera.max_depth_m - camera.min_depth_m) + camera.min_depth_m
    near = (depth <= camera.min_depth_m) | (depth <= 0)
    far = depth >= camera.max_depth_m
    depth[near] = np.nan
    depth[far] = np.inf
    return depth


#: Use the publisher's ``<scene>.navmesh`` exactly as shipped, whatever agent
#: it was built for.
NAVMESH_PUBLISHED = "published"
#: Use the navmesh habitat-sim derives for THIS agent's radius and height --
#: what habitat-lab's task actually runs on. See the class docstring.
NAVMESH_AGENT_RECOMPUTED = "agent"
NAVMESH_SOURCES = (NAVMESH_PUBLISHED, NAVMESH_AGENT_RECOMPUTED)


class HabitatRGBDSimulator:
    """Synchronous kinematic actions; no ROS, flight control or learned depth.

    **Which navmesh scores the run is an adapter decision, and it is not a
    detail.** habitat-sim's ``Simulator._config_pathfinder`` auto-loads
    ``<scene>.navmesh``, compares its stored settings with the agent's radius
    and height, and silently *recomputes* it when they differ. The shipped
    Habitat navmeshes carry the library defaults (radius 0.1 m, height 1.5 m),
    and habitat-lab copies a task's ``AGENT_0.RADIUS``/``HEIGHT`` straight into
    ``habitat_sim.AgentConfiguration`` (``create_sim_config``, no ignore key
    for either), so an ObjectNav agent of 0.18 m / 0.88 m always runs on a
    recomputed navmesh. Both its geodesic distances (``DistanceToGoal``, hence
    SR and SPL) and its collision filtering come from that one.

    Measured here on Gibson's Collierville with habitat-sim 0.2.4: the
    recomputed navmesh has 44.145 m2 of navigable area against the shipped
    one's 58.006 m2; 16 of 25 sampled geodesic distances differ by more than
    5 cm, and 14 of 39 sampled point pairs are reachable only on the shipped
    one. Calling ``pathfinder.load_nav_mesh()`` after construction puts the
    shipped navmesh back and therefore scores a *different, more permissive*
    world than the published evaluator does.

    So the choice is explicit and recorded in the run configuration:
    :data:`NAVMESH_AGENT_RECOMPUTED` reproduces habitat-lab, and
    :data:`NAVMESH_PUBLISHED` reproduces an evaluator that uses the shipped
    file. Neither is a library default, and this class still never *generates*
    a navmesh where the publisher shipped none.

    Args:
        camera: Shared CameraSpec.
        actions: Shared DiscreteActionSpec, mapped by name, never integer.
        radius_m: Embodiment radius, and half of what selects the navmesh
            under :data:`NAVMESH_AGENT_RECOMPUTED`.
        allow_sliding: Must match the chosen benchmark, not a library default.
        gpu_device: Rendering device index; this class never starts other models.
        height_m: Physical body height, independent of the camera mounting height.
        navmesh: One of :data:`NAVMESH_SOURCES`. No default is assumed for an
            adapter that does not say.
    """

    def __init__(self, camera, actions, radius_m, allow_sliding, gpu_device=0, *,
                 height_m, navmesh=NAVMESH_PUBLISHED):
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0
               for v in (radius_m, height_m)):
            raise ValueError("Explicit positive finite body radius and height are required")
        if navmesh not in NAVMESH_SOURCES:
            raise ValueError("navmesh must be one of %r; which one scores the run "
                             "changes SR and SPL" % (NAVMESH_SOURCES,))
        self.camera = camera
        self.actions = actions
        self.radius_m = radius_m
        self.height_m = height_m
        self.allow_sliding = allow_sliding
        self.gpu_device = gpu_device
        self.navmesh = navmesh
        self._sim = None
        self._scene = None

    def _install_navmesh(self, navmesh_path):
        """Put the adapter's chosen navmesh on the pathfinder, or refuse."""
        pathfinder = self._sim.pathfinder
        if self.navmesh == NAVMESH_PUBLISHED:
            if not pathfinder.load_nav_mesh(str(navmesh_path)):
                raise EnvContractError("Unable to load published navmesh: %s" % navmesh_path)
            return
        # habitat-sim has already loaded <scene>.navmesh and recomputed it for
        # this agent. Leave it alone -- but never run on "no navmesh at all",
        # which habitat-sim degrades to with only a warning.
        if not pathfinder.is_loaded:
            raise EnvContractError(
                "habitat-sim loaded no navmesh for %s; it would run with no "
                "collision checking at all" % navmesh_path)
        settings = pathfinder.nav_mesh_settings
        if (abs(settings.agent_radius - self.radius_m) > 1e-3
                or abs(settings.agent_height - self.height_m) > 1e-3):
            raise EnvContractError(
                "habitat-sim did not derive the navmesh for this agent: it has "
                "radius %.3f m / height %.3f m, the agent is %.3f m / %.3f m. "
                "Distances and collisions would come from another embodiment."
                % (settings.agent_radius, settings.agent_height,
                   self.radius_m, self.height_m))

    def reset(self, scene_path, navmesh_path, position, rotation_wxyz, seed):
        """Load the scene if needed and teleport ONLY to the published start."""
        import habitat_sim
        import quaternion

        if self._scene != str(scene_path):
            self.close()
            self._sim = habitat_sim.Simulator(self._configuration(scene_path))
            try:
                self._install_navmesh(navmesh_path)
            except BaseException:
                self.close()
                raise
            self._scene = str(scene_path)
        self._sim.seed(seed)
        state = habitat_sim.AgentState()
        state.position = np.asarray(position, dtype=np.float32)
        state.rotation = quaternion.from_float_array(rotation_wxyz)
        self._sim.get_agent(0).set_state(state, reset_sensors=True)
        return self._observe()

    def _configuration(self, scene_path):
        import habitat_sim

        config = habitat_sim.SimulatorConfiguration()
        config.scene_id = str(scene_path)
        config.gpu_device_id = self.gpu_device
        config.enable_physics = False
        config.allow_sliding = self.allow_sliding
        agent = habitat_sim.agent.AgentConfiguration()
        agent.height = self.height_m
        agent.radius = self.radius_m
        sensors = []
        k = self.camera.intrinsics
        hfov = math.degrees(2 * math.atan(k.width / (2 * k.fx)))
        for name, kind in (("rgb", habitat_sim.SensorType.COLOR),
                           ("depth", habitat_sim.SensorType.DEPTH)):
            sensor = habitat_sim.CameraSensorSpec()
            sensor.uuid = name
            sensor.sensor_type = kind
            sensor.resolution = [k.height, k.width]
            sensor.position = [0.0, self.camera.height_m, 0.0]
            sensor.hfov = hfov
            sensors.append(sensor)
        agent.sensor_specifications = sensors
        amounts = {"move_forward": self.actions.forward_step_m,
                   "turn_left": self.actions.turn_angle_deg,
                   "turn_right": self.actions.turn_angle_deg,
                   "look_up": self.actions.tilt_angle_deg,
                   "look_down": self.actions.tilt_angle_deg}
        agent.action_space = {
            a.name.lower(): habitat_sim.agent.ActionSpec(
                a.name.lower(), habitat_sim.agent.ActuationSpec(amount=amounts[a.name.lower()]))
            for a in self.actions.actions if a != DiscreteAction.STOP}
        return habitat_sim.Configuration(config, [agent])

    def navmesh_provenance(self):
        """What the loaded navmesh actually is, for the run record.

        The whole point of the ``navmesh`` choice is that it is invisible in a
        result unless someone writes it down.
        """
        settings = self.pathfinder.nav_mesh_settings
        return {"source": self.navmesh, "scene": self._scene,
                "agent_radius_m": float(settings.agent_radius),
                "agent_height_m": float(settings.agent_height),
                "cell_size_m": float(settings.cell_size),
                "navigable_area_m2": float(self.pathfinder.navigable_area)}

    @property
    def pathfinder(self):
        """The loaded published navmesh, for an adapter's evaluator-only distances.

        Read-only use: geodesic distance and navigability. Nothing here plans
        for the policy, and the policy never sees this object.
        """
        if self._sim is None:
            raise EnvContractError("Simulator not reset; no navmesh is loaded")
        return self._sim.pathfinder

    def step(self, action):
        """STOP preserves the final view; other actions use native collision tests."""
        if self._sim is None or not self.actions.allows(action):
            raise EnvContractError("Simulator not reset, or unsupported action")
        if action != DiscreteAction.STOP:
            raw = self._sim.step(action.name.lower())
            return self._observe(raw)
        return self._observe()

    def _observe(self, raw=None):
        import quaternion

        if raw is None:
            raw = self._sim.get_sensor_observations()
        state = self._sim.get_agent(0).get_state()
        pose = habitat_pose(state.position, quaternion.as_rotation_matrix(state.rotation),
                            quaternion.as_rotation_matrix(state.sensor_states["depth"].rotation))
        rgb = np.asarray(raw["rgb"])[..., :3].copy()
        return rgb, metric_depth(raw["depth"], self.camera), pose

    def close(self):
        """Release rendering memory before another scene/process takes it."""
        if self._sim is not None:
            self._sim.close()
        self._sim = None
        self._scene = None

