"""MP3D episodes and habitat-lab's own ObjectNav metrics behind ObjNavEnv.

The **MP3D adapter's environment**: the shared Habitat environment
(:class:`~sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.env.HabitatObjNavEnv`,
which owns the episode, the measurement the way habitat-lab's ``objectnav_mp3d``
task measures it, start validation, navmesh provenance and the oracle
hand-off) specialised by what is Matterport3D's alone -- its name, its
dataset, and how an annotated instance is named. What differs from HM3D is
entirely in ``mp3d/dataset.py`` (the published release's layout, the
``.house`` / ``_semantic.ply`` annotation files and the publisher's
``mp3d.scene_dataset_config.json`` that registers them to the mesh) and
``ground_truth/mp3d_categories.py`` (the two names each instance carries); the
algorithm sees only ``ObjNavObservation`` and never learns which benchmark it
is on.

**The ground-truth perception upper bound** (``ground_truth=`` a
:class:`~...ground_truth.source.GroundTruthSource` built with
``mp3d_categories.MP3D_TABLE``) labels the scene from both of each instance's
names -- the raw ``.house`` category and its mpcat40 class
(:meth:`MP3DEnv._annotations`) -- and the publisher's goal ids
(``MP3DDataset.goal_instances``). Loading the annotations needs
``MP3DDataset.semantic_scene_dataset_config`` installed.
"""
from __future__ import annotations

from sparx_agency.tasks.planning.objnav_benchmark_runtime.ground_truth.mp3d_categories import MP3DAnnotation
from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.env import HabitatObjNavEnv


class MP3DEnv(HabitatObjNavEnv):
    """One MP3D ObjectNav split, executed and measured under one protocol.

    Args:
        dataset: A loaded :class:`~...mp3d.dataset.MP3DDataset`.
        protocol: The :class:`~...mp3d.protocol.MP3DProtocol` it was loaded for.
        seed, gpu_device, simulator, excluded_episode_ids, distance_factory, ground_truth:
            As :class:`HabitatObjNavEnv`.
    """

    BENCHMARK = "mp3d"
    name = "habitat/mp3d-objectnav"

    def _annotations(self):
        """``{instance: MP3DAnnotation(raw, mpcat40)}`` of the loaded scene: both names each MP3D instance carries, for the oracle's table."""
        raw = self._simulator.semantic_objects("raw")
        mpcat40 = self._simulator.semantic_objects("mpcat40")
        return {instance: MP3DAnnotation(raw[instance], mpcat40[instance]) for instance in sorted(raw)}
