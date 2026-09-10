"""A fake ObjectNav environment for testing the pipeline: a grid building, an ObjNavEnv on it, and a privileged oracle.

A **test rig, not a benchmark.** It lets the whole headless loop -- the
environment contract, the agent, the action converter, the runner, the
scoring and the logger -- run in seconds with no simulator, and its oracle
shows that loop can score SR 1.0. Nothing it produces is a result.

* :class:`GridWorld` -- the building: walls, floor and objects on a grid
  (``world.py``), drawn as text (``ascii_map.py``); where an agent can stand,
  its goal regions and the voxels a camera sees (``raster.py``); grid
  distances (``geodesics.py``).
* :class:`FakeObjNavEnv` and :class:`FakeEpisodeSpec` -- the
  :class:`~sparx_agency.core.planning.objnav.interfaces.ObjNavEnv` on it
  (``env.py``), with its episodes and their ground truth (``episodes.py``),
  ray-cast depth (``rendering.py``) and Habitat-shaped success.
* :class:`OracleSearchPolicy` -- privileged: walks the geodesic into the goal
  region and stops (``oracle_policy.py``, its aim in ``oracle_aim.py``). An
  upper bound, never a baseline.
* :func:`fake_label_mapper` -- the building's categories as a label table.

Its errors are the core ObjectNav family (``ObjNavError``,
``EnvContractError``), not the harness's: it plays the part of a simulator
adapter, and raises what the ``ObjNavEnv`` contract names.

Python 3.8 syntax, numpy.
"""
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.env import (
    DEFAULT_AGENT_RADIUS_M,
    DEFAULT_MAX_STEPS,
    DEFAULT_SUCCESS_DISTANCE_M,
    FakeObjNavEnv,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.episodes import (
    FakeEpisodeSpec,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.labels import (
    fake_label_mapper,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.oracle_policy import (
    OracleSearchPolicy,
)
from sparx_agency.tasks.planning.objnav_benchmark.fake_env.world import GridWorld

__all__ = [
    "GridWorld",
    "FakeEpisodeSpec",
    "FakeObjNavEnv",
    "OracleSearchPolicy",
    "fake_label_mapper",
    "DEFAULT_AGENT_RADIUS_M",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_SUCCESS_DISTANCE_M",
]
