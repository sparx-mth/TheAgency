"""RoboTHOR episodes and challenge scoring behind the shared environment contract.

The whole benchmark rule set lives here, and nowhere above here:

* **Success is the challenge's**, not a distance test of ours. The runner in
  ``robothor_challenge/challenge.py`` computes ``stopped and
  target_obj["visible"]``; AI2-THOR's ``visible`` folds proximity (bounded by
  ``visibilityDistance: 1.0``), the camera frustum and an occlusion ray into
  one flag, which is how the published "within 1 metre and visible" rule is
  actually enforced. Nothing here recomputes it.

  One ambiguity has to be resolved rather than inherited. Upstream's
  ``get_object_by_type`` returns the **first** object of the goal type in the
  metadata list and asks only whether *that* instance is visible, so an
  episode in a scene with two Mugs is failed when the agent finds the second.
  AllenAct's task -- and the plain reading of the published rule -- accepts
  **any** visible instance. This adapter takes the any-instance rule and
  records the first-instance verdict beside it in ``info``, so the size of the
  difference is measurable instead of assumed. :attr:`success_rule` selects it
  explicitly; it is written into every run manifest.

* **``l`` is the shipped polyline**, ``episode["shortest_path_length"]``, the
  same number ``ai2thor.util.metrics.compute_spl`` uses. It is never
  recomputed at run time: a locally regenerated geodesic would be a different
  benchmark. :mod:`.distance` can *check* it against the live build, which is
  a preflight, not a scoring path.

* **``p`` is accounted twice, on purpose.** Once from the poses this
  environment hands the agent, in world ENU -- which the harness re-derives
  and cross-checks -- and once in AI2-THOR's own frame, exactly as upstream's
  ``path_distance`` does, which becomes ``native_metrics["spl"]``. The
  conversion between the two frames is an isometry, so the two lengths must
  agree; if they ever do not, the frame conversion has stopped being one, and
  the native cross-check says so. That is worth more here than on a Habitat
  benchmark, because AI2-THOR computes no metrics of its own to check against.

* **A failed action still spends a step.** Upstream increments its counter
  before issuing the action, unconditionally, so a collision or a LOOK
  refused at the LoCoBot's horizon clamp costs budget and contributes nothing
  to the path. That is reproduced here.

Python 3.8 syntax.
"""
from __future__ import annotations

import math

from sparx_agency.core.planning.objnav.errors import EnvContractError
from sparx_agency.core.planning.objnav.interfaces.env import ObjNavEnv
from sparx_agency.core.planning.objnav.types.actions import DiscreteAction
from sparx_agency.core.planning.objnav.types.episode import ObjNavEpisode
from sparx_agency.core.planning.objnav.types.measurement import (
    TERMINATION_STEP_LIMIT, TERMINATION_STOP, EpisodeMeasurement,
)
from sparx_agency.core.planning.objnav.types.observation import ObjNavObservation
from sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.dataset import (
    path_distance, vector_distance,
)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.distance import (
    RobothorGeodesic,
)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.robothor.protocol import (
    PROTOCOL,
)
from sparx_agency.tasks.planning.objnav_benchmark_runtime.thor.simulator import (
    AI2ThorRGBDSimulator,
)

#: Accept any visible instance of the goal type (AllenAct, and the plain
#: reading of the published rule).
SUCCESS_ANY_INSTANCE = "any_visible_instance"
#: Reproduce ``robothor-challenge``'s ``get_object_by_type``, which inspects
#: only the first instance listed in the scene metadata.
SUCCESS_FIRST_INSTANCE = "first_listed_instance"
SUCCESS_RULES = (SUCCESS_ANY_INSTANCE, SUCCESS_FIRST_INSTANCE)

#: How far the simulator may land from the published spawn point, metres.
_START_TOLERANCE_M = 1e-4


def instances_of(objects, object_type):
    """Every object of ``object_type``, in the metadata's own order.

    Matched on ``objectId.split("|")[0]``, which is what upstream matches on:
    ``objectType`` is not present on every build, and the id prefix is.
    """
    return [obj for obj in objects
            if str(obj.get("objectId", "")).split("|")[0] == object_type]


def official_spl(success, shortest_path_m, path_length_m) -> float:
    """``ai2thor.util.metrics.compute_single_spl``, transcribed.

    Including its edge case: both lengths zero gives a ratio of 1, but a
    zero ``l`` with any movement at all gives 0. Six per cent of the published
    validation episodes ship ``shortest_path_length == 0`` -- the agent spawns
    already inside the goal cushion -- so on those a success that took a
    single step still scores SPL 0. That is upstream's arithmetic, not a bug
    introduced here, and the adapter's README says so.
    """
    indicator = 1.0 if success else 0.0
    if max(path_length_m, shortest_path_m) > 0:
        ratio = shortest_path_m / max(path_length_m, shortest_path_m)
    else:
        ratio = 1.0
    return indicator * ratio


class RobothorEnv(ObjNavEnv):
    """The RoboTHOR ObjectNav challenge behind :class:`ObjNavEnv`.

    Args:
        dataset: A :class:`~...robothor.dataset.RobothorDataset`.
        simulator: An :class:`~...thor.simulator.AI2ThorRGBDSimulator`, or
            None to build one from the protocol.
        success_rule: One of :data:`SUCCESS_RULES`.
        geodesic_telemetry: Whether :meth:`evaluation_diagnostics` should ask
            the navmesh for a geodesic distance on every step. It is a
            simulator round trip per step, so it is off unless a recording
            wants the distance-to-goal curve; the cheap Euclidean distance is
            always reported, under its own name.
        platform: Passed to the simulator when this builds one.

    Raises:
        ValueError: On an unknown success rule.
    """

    name = "ai2thor/robothor-objectnav-challenge-2021"

    def __init__(self, dataset, simulator=None, *,
                 success_rule=SUCCESS_ANY_INSTANCE, geodesic_telemetry=False,
                 platform=None):
        if success_rule not in SUCCESS_RULES:
            raise ValueError("success_rule must be one of %r" % (SUCCESS_RULES,))
        self._dataset = dataset
        self._camera = PROTOCOL.camera()
        self._actions = PROTOCOL.actions()
        self.success_rule = success_rule
        self.geodesic_telemetry = geodesic_telemetry
        self._simulator = simulator if simulator is not None else (
            AI2ThorRGBDSimulator(
                self._camera, self._actions, PROTOCOL.agent_radius_m,
                height_m=PROTOCOL.body_height_m,
                origin_height_m=PROTOCOL.origin_height_m,
                initialize=PROTOCOL.initialize(),
                commit_id=PROTOCOL.thor_build_id,
                width=PROTOCOL.width, height=PROTOCOL.height,
                platform=platform))
        self._geodesic = None
        self._episode = None
        self._row = None
        self._observation = None
        self._steps = 0
        self._stopped = False
        self._path_m = 0.0
        self._thor_path = []
        self._action_failures = 0
        #: How the two published readings of the visibility rule compare over
        #: this run. The harness's EpisodeRecord keeps only the canonical
        #: native metrics, so a measurement's ``info`` never reaches disk;
        #: the adapter tallies it here and the CLI writes it beside the
        #: results, so the choice of rule is reported rather than assumed.
        self.visibility_tally = {"episodes": 0, "any_instance_successes": 0,
                                 "first_instance_successes": 0,
                                 "episodes_that_differ": 0}

    # ------------------------------------------------------------ contract

    def episode_ids(self):
        return tuple(self._dataset.episodes)

    def reset(self, episode_id):
        row = self._dataset.episodes[episode_id]
        frame = self._simulator.reset(row.scene, row.start_position,
                                      row.start_rotation_deg,
                                      row.start_horizon_deg)
        self._row = row
        self._steps = 0
        self._stopped = False
        self._action_failures = 0
        self._path_m = PROTOCOL.path_length_epsilon_m
        # No privileged value reaches the episode: not the goal position, not
        # the shipped path, not its length. Only the public task description.
        self._episode = ObjNavEpisode(
            episode_id, row.scene, PROTOCOL.benchmark, PROTOCOL.split,
            row.category, self._camera, self._actions, PROTOCOL.max_steps)
        self._observation = self._observation_from(frame)
        self._check_start(row)
        self._check_goal_present(row)
        self._thor_path = [dict(self._agent_position())]
        return self._episode, self._observation

    def step(self, action):
        if (self._episode is None or self.episode_over
                or not self._actions.allows(action)):
            raise EnvContractError("Episode not running, or action not allowed")
        before = self._observation.pose
        frame = self._simulator.step(action)
        self._steps += 1
        self._stopped = action == DiscreteAction.STOP
        if not self._simulator.last_action_succeeded:
            self._action_failures += 1
        self._observation = self._observation_from(frame)
        after = self._observation.pose
        self._path_m += math.sqrt((after.x - before.x) ** 2
                                  + (after.y - before.y) ** 2
                                  + (after.z - before.z) ** 2)
        self._thor_path.append(dict(self._agent_position()))
        return self._observation

    @property
    def episode_over(self):
        return self._episode is not None and (
            self._stopped or self._steps >= PROTOCOL.max_steps)

    def measure(self):
        if not self.episode_over:
            raise EnvContractError(
                "measure() is only available after episode end")
        objects = self._simulator.metadata["objects"]
        found = instances_of(objects, self._row.category)
        any_visible = any(bool(obj.get("visible")) for obj in found)
        first_visible = bool(found[0].get("visible")) if found else False
        visible = (any_visible if self.success_rule == SUCCESS_ANY_INSTANCE
                   else first_visible)
        success = bool(self._stopped and visible)
        self._tally(any_visible, first_visible)
        shortest = self._row.shortest_path_length_m
        distance = self._final_distance()
        return EpisodeMeasurement(
            success=success, stop_called=self._stopped,
            termination=TERMINATION_STOP if self._stopped else TERMINATION_STEP_LIMIT,
            steps=self._steps,
            shortest_path_m=shortest,
            start_distance_to_goal_m=shortest,
            final_distance_to_goal_m=distance,
            path_length_m=self._path_m,
            # Accounted in AI2-THOR's own frame, by upstream's own arithmetic:
            # an independent path through a different coordinate system, so a
            # conversion that stopped being an isometry shows up here.
            native_metrics={"spl": official_spl(success, shortest,
                                                path_distance(self._thor_path))},
            info={
                "success_rule": self.success_rule,
                "any_instance_visible": any_visible,
                "first_listed_instance_visible": first_visible,
                "instances_of_goal_in_scene": len(found),
                "failed_actions": self._action_failures,
                "thor_path_length_m": path_distance(self._thor_path),
                "shortest_path_is_zero": shortest == 0.0,
                "euclidean_distance_to_goal_m": self._euclidean_distance(),
                "geodesic_distance_measured": self._geodesic is not None,
            })

    def _tally(self, any_visible, first_visible):
        """Record what each reading of the rule would have scored."""
        any_hit = bool(self._stopped and any_visible)
        first_hit = bool(self._stopped and first_visible)
        self.visibility_tally["episodes"] += 1
        self.visibility_tally["any_instance_successes"] += int(any_hit)
        self.visibility_tally["first_instance_successes"] += int(first_hit)
        self.visibility_tally["episodes_that_differ"] += int(any_hit != first_hit)

    def evaluation_diagnostics(self):
        """Recorder/evaluator-only values, never policy observations."""
        if self._episode is None:
            raise EnvContractError("No episode has been reset")
        telemetry = {"path_length_m": self._path_m,
                     "shortest_path_m": self._row.shortest_path_length_m,
                     "euclidean_distance_to_goal_m": self._euclidean_distance()}
        if self.geodesic_telemetry:
            telemetry["distance_to_goal_m"] = self._geodesic_distance()
        return telemetry

    def close(self):
        self._simulator.close()

    # -------------------------------------------------------------- inside

    def _observation_from(self, frame):
        rgb, depth, pose = frame
        return ObjNavObservation(rgb, depth, pose, self._camera,
                                 self._episode.target_category, self._steps)

    def _agent_position(self):
        """The agent position in AI2-THOR's own frame, for upstream accounting."""
        return self._simulator.metadata["agent"]["position"]

    def _check_start(self, row):
        """The published start is the episode; a moved one is another episode."""
        actual = self._agent_position()
        drift = vector_distance(actual, row.start_position)
        if drift > _START_TOLERANCE_M:
            raise EnvContractError(
                "Simulator placed %s %.5f m from its published start"
                % (row.episode_id, drift))

    def _check_goal_present(self, row):
        """Upstream asserts the goal exists; say why instead of asserting."""
        if not instances_of(self._simulator.metadata["objects"], row.category):
            raise EnvContractError(
                "Scene %s holds no %s, but episode %s asks for one; the "
                "episode files and the loaded build disagree"
                % (row.scene, row.category, row.episode_id))

    def _goal_objects(self):
        return instances_of(self._simulator.metadata["objects"], self._row.category)

    def _euclidean_distance(self) -> float:
        """Straight-line metres to the nearest goal instance. Cheap, and not DTG."""
        position = self._agent_position()
        distances = [vector_distance(position, obj["position"])
                     for obj in self._goal_objects() if "position" in obj]
        return min(distances) if distances else float("inf")

    def _geodesic_distance(self) -> float:
        """Navmesh metres to the nearest goal instance; the reported DTG."""
        if self._geodesic is None:
            self._geodesic = RobothorGeodesic(self._simulator.controller)
        return self._geodesic.distance(self._agent_position(), self._row.category)

    def _final_distance(self) -> float:
        """``dT``: geodesic where the navmesh can be asked, Euclidean otherwise.

        The geodesic is the quantity the literature reports and the one
        ``start_distance_to_goal_m`` already is, so SoftSPL stays coherent.
        A build without a reachable navmesh query falls back to the Euclidean
        distance, which is never larger, and the fallback is recorded in
        ``info`` rather than hidden.
        """
        try:
            distance = self._geodesic_distance()
        except (AttributeError, KeyError, ValueError):
            return self._euclidean_distance()
        if math.isfinite(distance):
            return distance
        return self._euclidean_distance()

    def geodesic_query_counts(self):
        """How often the navmesh needed a relaxed tolerance, for the manifest."""
        return dict(self._geodesic.queries) if self._geodesic else {}
