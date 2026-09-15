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


class HabitatRGBDSimulator:
    """Synchronous kinematic actions; no ROS, flight control or learned depth.

    Args:
        camera: Shared CameraSpec.
        actions: Shared DiscreteActionSpec, mapped by name, never integer.
        radius_m: Embodiment radius. The published navmesh remains authoritative.
        allow_sliding: Must match the chosen benchmark, not a library default.
        gpu_device: Rendering device index; this class never starts other models.
    """

    def __init__(self, camera, actions, radius_m, allow_sliding, gpu_device=0):
        self.camera = camera
        self.actions = actions
        self.radius_m = radius_m
        self.allow_sliding = allow_sliding
        self.gpu_device = gpu_device
        self._sim = None
        self._scene = None
        self.last_collision = None  # evaluator-only; not part of RGB-D/pose tuples

    def reset(self, scene_path, navmesh_path, position, rotation_wxyz, seed):
        """Load the scene if needed and teleport ONLY to the published start."""
        import habitat_sim
        import quaternion

        if self._scene != str(scene_path):
            self.close()
            self._sim = habitat_sim.Simulator(self._configuration(scene_path))
            if not self._sim.pathfinder.load_nav_mesh(str(navmesh_path)):
                self.close()
                raise EnvContractError("Unable to load published navmesh: %s" % navmesh_path)
            self._scene = str(scene_path)
        self._sim.seed(seed)
        state = habitat_sim.AgentState()
        state.position = np.asarray(position, dtype=np.float32)
        state.rotation = quaternion.from_float_array(rotation_wxyz)
        self._sim.get_agent(0).set_state(state, reset_sensors=True)
        self.last_collision = None
        return self._observe()

    def _configuration(self, scene_path):
        import habitat_sim

        config = habitat_sim.SimulatorConfiguration()
        config.scene_id = str(scene_path)
        config.gpu_device_id = self.gpu_device
        config.enable_physics = False
        config.allow_sliding = self.allow_sliding
        agent = habitat_sim.agent.AgentConfiguration()
        agent.height = self.camera.height_m
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

    def step(self, action):
        """STOP preserves the final view; other actions use native collision tests."""
        if self._sim is None or not self.actions.allows(action):
            raise EnvContractError("Simulator not reset, or unsupported action")
        if action != DiscreteAction.STOP:
            raw = self._sim.step(action.name.lower())
            collided = raw.get("collided", getattr(self._sim, "previous_step_collided", None))
            self.last_collision = None if collided is None else bool(collided)
            return self._observe(raw)
        self.last_collision = False
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

