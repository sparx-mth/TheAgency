"""Optional reusable habitat-sim RGB-D bridge. No task goals or metric access.

Never snaps a published episode start. Import is GPU-free; a rendering context
is created only by reset(). The same bridge can serve Gibson, HM3D and MP3D
behind their separate dataset evaluators.

**Which navmesh an episode runs on is an explicit adapter choice**, because it
is not a detail: the navmesh decides what is reachable, so it sets the geodesic
``l`` and therefore every SPL. ``navmesh="published"`` loads the file the
publisher shipped and uses exactly that. ``navmesh="agent"`` is habitat-sim's
own behaviour, and the one habitat-lab's ObjectNav actually runs:
``Simulator._config_pathfinder`` loads the sibling ``<scene>.navmesh`` and then
recomputes it whenever the mesh's recorded settings differ from the agent's
radius and height -- which they do for HM3D, whose meshes are built at the
library defaults (0.10 m / 1.50 m) while ObjectNav's agent is 0.18 m / 0.88 m.
Neither is guessed from a benchmark name, and whichever is chosen,
``navmesh_provenance()`` records what was actually in force.

Beyond RGB-D and pose the bridge exposes two evaluator/policy-side readings of
the loaded navmesh: :meth:`scene_structure` (storeys and stair connectors, the
one declared ground-truth input of the multi-floor search) and
:attr:`last_collision` (recorder-only). Neither is part of an observation.

**The semantic sensor is opt-in and privileged** (``semantic=True``). It
renders habitat-sim's per-pixel instance ids from the scene's HM3D-Semantics
annotation mesh, registered to the depth camera, into :attr:`last_semantic`
after every :meth:`reset` and :meth:`step`; :meth:`semantic_objects` names
each instance's raw annotation category. The RGB-D/pose tuple the bridge
returns is unchanged, so nothing reaches a policy through an observation: the
ground-truth perception of the benchmark runtime (``ground_truth/``) reads the
frame beside it and stands in for the detector and the segmenter. Loading the
annotations needs the publisher's *annotated* scene dataset config
(``scene_dataset_config``), without which habitat-sim renders every pixel as
instance 0 and says nothing -- so a scene that loads no semantic objects is
refused rather than flown blind.
"""
from __future__ import annotations

import math

import numpy as np

from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.core.mapping.depth.depth_holes import fill_depth_holes

# Measured reconciliations were 0.021-0.050 m; past this it's a mismatched
# navmesh, not a reconciliation -- refuse it instead of fixing it quietly.
_MAX_RESET_SETTLE_M = 0.5
#: A landing point this close to the navmesh's own answer is confirmed real.
_NAVMESH_CONFIRM_M = 0.05


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


def metric_depth(raw, camera, normalized=False, depth_holes=None, stats=None):
    """Decode Lab normalized depth or raw Sim metres; preserve clip semantics.

    ``depth_holes`` (a :class:`DepthHoleFill`) fills the enclosed no-return
    pixels -- raw ``0``, the scan's missing geometry -- from their valid
    neighbours before the near/far clipping, so a mirror or a black screen
    projects as the wall around it; the remaining holes are ``NaN``, no
    reading, as every baseline treats them. ``stats`` (a dict) receives the
    fill's counts when given.
    """
    depth = np.asarray(raw, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[..., 0]
    depth = depth.copy()
    holes = depth <= 0 if not normalized else np.zeros(depth.shape, dtype=bool)
    if normalized:
        depth = depth * (camera.max_depth_m - camera.min_depth_m) + camera.min_depth_m
    if depth_holes is not None and depth_holes.enabled and holes.any():
        depth, counts = fill_depth_holes(depth, holes, depth_holes)
        if stats is not None:
            stats.update(counts)
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
        height_m: Physical body height, independent of the camera mounting height.
        navmesh: ``"published"`` to navigate the shipped navmesh verbatim, or
            ``"agent"`` for habitat-sim's own load-then-recompute-at-the-agent's
            -dimensions behaviour, which is what habitat-lab's ObjectNav does.
            See the module docstring; this is a protocol decision.
        semantic: Render the per-pixel instance-id frame beside RGB-D (see the
            module docstring). Off for a benchmark run; on only for the
            ground-truth perception upper bound.
        scene_dataset_config: The publisher's annotated scene dataset config
            (``hm3d_annotated_basis.scene_dataset_config.json``), which tells
            habitat-sim where each scene's ``.semantic.glb`` / ``.semantic.txt``
            are. Required with ``semantic=True``.
    """

    def __init__(self, camera, actions, radius_m, allow_sliding, gpu_device=0, *,
                 height_m, navmesh="published", depth_holes=None, semantic=False,
                 scene_dataset_config=None):
        if any(isinstance(v, bool) or not math.isfinite(v) or v <= 0
               for v in (radius_m, height_m)):
            raise ValueError("Explicit positive finite body radius and height are required")
        if navmesh not in ("published", "agent"):
            raise ValueError("navmesh must be 'published' or 'agent', got %r" % (navmesh,))
        if semantic and scene_dataset_config is None:
            raise ValueError("semantic=True needs the annotated scene_dataset_config; without it "
                             "habitat-sim renders every pixel as instance 0 and says nothing")
        self.navmesh = navmesh
        self.camera = camera
        self.actions = actions
        self.radius_m = radius_m
        self.height_m = height_m
        self.allow_sliding = allow_sliding
        self.gpu_device = gpu_device
        self.semantic = bool(semantic)
        self.scene_dataset_config = None if scene_dataset_config is None else str(scene_dataset_config)
        self.depth_holes = depth_holes          # DepthHoleFill or None (every hole dropped, as the baselines do)
        self.depth_hole_stats = {"frames": 0, "frames_with_holes": 0, "holes": 0, "filled": 0,
                                 "pixels_filled": 0, "pixels_left": 0}
        self._sim = None
        self._scene = None
        self._structure = None  # per-scene navmesh storeys and stair connectors
        self._semantic_objects = None  # per-scene {instance id: raw annotation category}
        self.last_collision = None  # evaluator-only; not part of RGB-D/pose tuples
        self.last_sensor_alignment = None
        #: Privileged, refreshed every _observe() under ``semantic=True``: the
        #: HxW int32 instance-id frame registered to the depth frame. Never
        #: part of the returned tuple; ``None`` without the sensor.
        self.last_semantic = None
        # Diagnostic only, never read by the policy: the navmesh-corrected
        # height's gap from the raw published one, set fresh each reset().
        self.last_reset_settle_gap_m = 0.0
        #: Checker-only, refreshed every _observe(): is the current pose
        #: within _NAVMESH_CONFIRM_M of the navmesh's own answer there.
        self.last_pose_on_navmesh = False
        #: Checker-only, refreshed every _observe(): the pose the step filter
        #: will start the next body action from -- the current pose with its
        #: position projected onto the navmesh. habitat-sim's ``tryStep``
        #: begins every turn and MOVE_FORWARD at ``projectToPoly(start)``,
        #: the same projection ``pathfinder.snap_point`` returns; an agent it
        #: finds off its mesh (a stride on a stair edge can leave it so) is
        #: carried onto it by that action, or left where it is when it
        #: refuses the step. None until the first observation, or when the
        #: mesh has no answer near the agent. Never seen by the policy.
        self.last_placed_pose = None

    def reset(self, scene_path, navmesh_path, position, rotation_wxyz, seed):
        """Load the scene if needed and place the agent at the published start.

        Under ``navmesh="published"`` the start is used verbatim, as the
        publisher wrote it. Under ``navmesh="agent"`` the recomputed mesh's
        floor sits a few centimetres from the published one, so the height is
        fixed once here from this run's own navmesh -- the episode's one
        working height under its 2.5-D contract -- instead of jumping on the
        first forward step, which habitat-sim would otherwise snap. The (x, z)
        start is never moved in either mode.
        """
        import habitat_sim
        import quaternion

        self.depth_hole_stats = {key: 0 for key in self.depth_hole_stats}     # per episode
        if self._scene != str(scene_path):
            self.close()
            self._sim = habitat_sim.Simulator(self._configuration(scene_path))
            if self.navmesh == "published":
                if not self._sim.pathfinder.load_nav_mesh(str(navmesh_path)):
                    self.close()
                    raise EnvContractError("Unable to load published navmesh: %s"
                                           % navmesh_path)
            else:
                sibling = str(scene_path)
                sibling = sibling[:sibling.rfind(".")] + ".navmesh"
                if str(navmesh_path) != sibling:
                    self.close()
                    raise EnvContractError(
                        "navmesh='agent' uses the navmesh habitat-sim finds beside "
                        "the scene (%s), not %s" % (sibling, navmesh_path))
                if not self._sim.pathfinder.is_loaded:
                    self.close()
                    raise EnvContractError(
                        "habitat-sim loaded no navmesh for %s; nothing can be "
                        "measured on it" % scene_path)
            self._scene = str(scene_path)
            self._structure = None
            self._semantic_objects = self._read_semantic_objects(scene_path) if self.semantic else None
        self._sim.seed(seed)
        raw_position = np.asarray(position, dtype=np.float32)
        self.last_reset_settle_gap_m = 0.0
        if self.navmesh == "agent":
            snapped = np.asarray(self._sim.pathfinder.snap_point(raw_position), dtype=np.float32)
            if snapped.shape != (3,) or not np.isfinite(snapped).all():
                raise EnvContractError(
                    "Published start %s is not near this run's navmesh at all; "
                    "the recomputed mesh does not match the episodes" % (raw_position.tolist(),))
            self.last_reset_settle_gap_m = float(snapped[1] - raw_position[1])
            if abs(self.last_reset_settle_gap_m) > _MAX_RESET_SETTLE_M:
                raise EnvContractError(
                    "Published start height is %.3f m from this navmesh's own height "
                    "at (x, z); too far to be the usual agent-radius navmesh-recompute "
                    "reconciliation (checked against %.2f m) -- the navmesh likely "
                    "doesn't match this episode at all" % (self.last_reset_settle_gap_m,
                                                            _MAX_RESET_SETTLE_M))
            raw_position = np.array(
                [raw_position[0], snapped[1], raw_position[2]], dtype=np.float32)
        state = habitat_sim.AgentState()
        state.position = raw_position
        state.rotation = quaternion.from_float_array(rotation_wxyz)
        self._sim.get_agent(0).set_state(state, reset_sensors=True)
        self.last_collision = None
        return self._observe()

    def scene_structure(self):
        """Storeys and stair connectors of the loaded scene's navmesh, ENU.

        Ground truth for the building's vertical structure -- never the goal.
        Computed once per scene from whichever navmesh is in force (see
        ``habitat/stair_connectors.py``).
        """
        if self._sim is None:
            raise EnvContractError("Simulator not reset; no navmesh to read the building from")
        if self._structure is None:
            from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.stair_connectors import (
                scene_structure_from_pathfinder)
            self._structure = scene_structure_from_pathfinder(self._sim.pathfinder)
        return self._structure

    def semantic_objects(self):
        """``{instance id: raw annotation category}`` of the loaded scene, under ``semantic=True``.

        The ids are the ones :attr:`last_semantic` renders and the ones the
        published episodes' goal rows name as ``object_id``; the category is
        the annotator's own word (``"armchair"``, ``"tv stand"``), not a
        detector label -- the mapping into the method's vocabulary is the
        ground-truth perception's, not the bridge's.

        Raises:
            EnvContractError: No scene is loaded, or the sensor is off.
        """
        if self._sim is None or self._semantic_objects is None:
            raise EnvContractError("Simulator not reset with semantic=True; there are no annotations to read")
        return self._semantic_objects

    def _read_semantic_objects(self, scene_path):
        """Read the loaded scene's annotation table, refusing a scene that has none.

        habitat-sim does not fail when a scene's ``.semantic.glb`` or
        ``.semantic.txt`` is missing: it renders instance 0 everywhere and
        ``semantic_scene.objects`` is empty. An oracle that sees nothing would
        score like a blind agent and look like a method result, so this is a
        contract error here, with the file the publisher ships named.
        """
        objects = {}
        for obj in getattr(self._sim.semantic_scene, "objects", ()) or ():
            if obj is None:
                continue
            try:
                instance = int(obj.semantic_id)
            except (AttributeError, TypeError, ValueError):
                # habitat-sim 0.2.x names instances "<region>_<id>"; the id is the trailing field
                instance = int(str(obj.id).rsplit("_", 1)[-1])
            category = obj.category.name() if obj.category is not None else ""
            if instance <= 0:
                continue                  # habitat-sim's id 0 is "unknown": not an instance
            objects[instance] = str(category).strip().lower()
        if not objects:
            self.close()
            raise EnvContractError(
                "habitat-sim loaded no semantic annotations for %s: the scene dataset config (%s) must be the "
                "publisher's annotated one and the scene's .semantic.glb / .semantic.txt must sit beside the mesh "
                "(HM3D-Semantics annotations; hm3d.assets --instructions names the download)"
                % (scene_path, self.scene_dataset_config))
        return objects

    def _configuration(self, scene_path):
        import habitat_sim

        config = habitat_sim.SimulatorConfiguration()
        config.scene_id = str(scene_path)
        config.gpu_device_id = self.gpu_device
        config.enable_physics = False
        config.allow_sliding = self.allow_sliding
        if self.semantic:
            # The annotated dataset config names each scene's .semantic.glb/.semantic.txt;
            # without load_semantic_mesh the sensor renders instance 0 everywhere.
            config.scene_dataset_config_file = self.scene_dataset_config
            config.load_semantic_mesh = True
        agent = habitat_sim.agent.AgentConfiguration()
        agent.height = self.height_m
        agent.radius = self.radius_m
        sensors = []
        k = self.camera.intrinsics
        hfov = math.degrees(2 * math.atan(k.width / (2 * k.fx)))
        kinds = [("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)]
        if self.semantic:
            kinds.append(("semantic", habitat_sim.SensorType.SEMANTIC))
        for name, kind in kinds:
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

    @property
    def pathfinder(self):
        """The loaded navmesh, for an adapter's own privileged measurements.

        Geodesic distances belong to the evaluator, never to the policy: this
        is the same navmesh the episode is executed on, so a distance measured
        through it is the one the benchmark means. It is available only between
        reset() and close(), because a navmesh is a property of a loaded scene.
        """
        if self._sim is None:
            raise EnvContractError("Simulator not reset; there is no navmesh yet")
        return self._sim.pathfinder

    def navmesh_provenance(self):
        """What navmesh is actually in force, as recorded run metadata.

        Returns:
            The requested mode, the pathfinder's own settings, its navigable
            area and island count -- enough to tell afterwards which mesh a
            number was measured on.
        """
        pathfinder = self.pathfinder
        settings = {}
        recorded = getattr(pathfinder, "nav_mesh_settings", None)
        for name in ("agent_radius", "agent_height", "agent_max_climb",
                     "agent_max_slope", "cell_size", "cell_height"):
            value = getattr(recorded, name, None)
            if value is not None:
                settings[name] = float(value)
        return {"mode": self.navmesh, "settings": settings,
                "navigable_area_m2": float(pathfinder.navigable_area),
                "islands": int(pathfinder.num_islands)}

    def project_target(self, point):
        """Project a requested ENU footpoint onto the local NavMesh.

        This explicitly privileged geometry service reads neither semantic
        annotations nor evaluator targets. Reject other floors, remote snaps
        and invalid geometry; observed-map A* decides reachability.
        """
        if self._sim is None:
            raise EnvContractError("Simulator not reset; cannot project a target approach")
        point = np.asarray(point, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("Target approach projection needs a finite ENU xyz point")
        native = np.array([-point[1], point[2], -point[0]], dtype=np.float32)
        snapped = np.asarray(self._sim.pathfinder.snap_point(native), dtype=float)
        if snapped.shape != (3,) or not np.isfinite(snapped).all():
            return None
        enu = np.array([-snapped[2], -snapped[0], snapped[1]])
        if abs(enu[2] - point[2]) > 0.2 or np.linalg.norm(enu[:2] - point[:2]) > 0.5:
            return None
        return tuple(float(v) for v in enu)

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
        rgb_sensor, depth_sensor = state.sensor_states["rgb"], state.sensor_states["depth"]
        rgb_rotation = quaternion.as_rotation_matrix(rgb_sensor.rotation)
        depth_rotation = quaternion.as_rotation_matrix(depth_sensor.rotation)
        expected_position = np.asarray(state.position) + np.array([0.0, self.camera.height_m, 0.0])
        if (not np.allclose(rgb_sensor.position, depth_sensor.position, atol=1e-5, rtol=0)
                or not np.allclose(rgb_rotation, depth_rotation, atol=1e-5, rtol=0)
                or not np.allclose(depth_sensor.position, expected_position, atol=1e-4, rtol=0)):
            raise EnvContractError("RGB/depth/pose extrinsics are not registered to the declared camera mount")
        if self.semantic:
            semantic_sensor = state.sensor_states["semantic"]
            if (not np.allclose(semantic_sensor.position, depth_sensor.position, atol=1e-5, rtol=0)
                    or not np.allclose(quaternion.as_rotation_matrix(semantic_sensor.rotation), depth_rotation,
                                       atol=1e-5, rtol=0)):
                raise EnvContractError("The semantic sensor is not registered to the depth camera")
        body_rotation = quaternion.as_rotation_matrix(state.rotation)
        pose = habitat_pose(state.position, body_rotation, depth_rotation)
        rgb_pose = habitat_pose(state.position, body_rotation, rgb_rotation)
        self.last_sensor_alignment = {"rgb_pitch_rad": rgb_pose.camera_pitch, "depth_pitch_rad": pose.camera_pitch,
                                      "mount_error_m": float(np.linalg.norm(depth_sensor.position - expected_position)),
                                      "registered": True}
        snapped = np.asarray(self._sim.pathfinder.snap_point(state.position), dtype=float)
        self.last_pose_on_navmesh = bool(abs(float(snapped[1]) - float(state.position[1])) <= _NAVMESH_CONFIRM_M)
        self.last_placed_pose = (habitat_pose(snapped, body_rotation, depth_rotation)
                                 if snapped.shape == (3,) and np.isfinite(snapped).all() else None)
        rgb = np.asarray(raw["rgb"])[..., :3].copy()
        if self.semantic:
            ids = np.asarray(raw["semantic"])
            if ids.ndim == 3 and ids.shape[2] == 1:
                ids = ids[..., 0]
            if ids.shape != rgb.shape[:2]:
                raise EnvContractError("The semantic frame %s is not the RGB frame's size %s" % (ids.shape, rgb.shape[:2]))
            self.last_semantic = ids.astype(np.int32, copy=True)
        counts = {}
        depth = metric_depth(raw["depth"], self.camera, depth_holes=self.depth_holes, stats=counts)
        totals = self.depth_hole_stats
        totals["frames"] += 1
        if counts:
            totals["frames_with_holes"] += 1
            for key in ("holes", "filled", "pixels_filled", "pixels_left"):
                totals[key] += int(counts.get(key, 0))
        return rgb, depth, pose

    def close(self):
        """Release rendering memory before another scene/process takes it."""
        if self._sim is not None:
            self._sim.close()
        self._sim = None
        self._scene = None
        self._structure = None
        self._semantic_objects = None
        self.last_semantic = None

