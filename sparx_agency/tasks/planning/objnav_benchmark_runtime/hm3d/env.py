"""HM3D episodes and habitat-lab's own ObjectNav metrics behind ObjNavEnv.

The **HM3D adapter's environment**: the shared Habitat environment
(:class:`~sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.env.HabitatObjNavEnv`,
which owns the episode, the measurement, start validation, navmesh provenance
and the oracle hand-off) specialised by what is HM3D's alone -- its name, and
its dataset. What differs from MP3D is entirely in ``hm3d/dataset.py`` (the
published release's layout, the HM3D-Semantics annotation files and their
``hm3d_annotated_basis.scene_dataset_config.json``) and
``ground_truth/hm3d_categories.py`` (the raw annotation names); the algorithm
sees only ``ObjNavObservation`` and never learns which benchmark it is on.

**The ground-truth perception upper bound** (``ground_truth=`` a
:class:`~...ground_truth.source.GroundTruthSource`) reads habitat-sim's
default category name per instance -- the HM3D-Semantics raw name -- which the
source's ``HM3D_TABLE`` labels into the method's vocabulary beside the
publisher's own goal instance ids (``HM3DDataset.goal_instances``). Loading
the annotations needs the publisher's annotated config installed
(``HM3DDataset.annotated_scene_dataset_config``).
"""
from __future__ import annotations

from sparx_agency.tasks.planning.objnav_benchmark_runtime.habitat.env import HabitatObjNavEnv


class HM3DEnv(HabitatObjNavEnv):
    """One HM3D ObjectNav split, executed and measured under one protocol.

    Args:
        dataset: A loaded :class:`~...hm3d.dataset.HM3DDataset`.
        protocol: The :class:`~...hm3d.protocol.HM3DProtocol` it was loaded for.
        seed, gpu_device, simulator, excluded_episode_ids, distance_factory, ground_truth:
            As :class:`HabitatObjNavEnv`.
    """

    BENCHMARK = "hm3d"
    #: Identifier of the environment. The version is part of it: rows from the
    #: two HM3D datasets are never mixed, and a results directory should say
    #: which one produced it without opening ``run.json``.
    name = "habitat/hm3d-objectnav"
