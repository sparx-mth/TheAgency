"""What every Habitat ObjectNav entrypoint (``hm3d/run.py``, ``mp3d/run.py``) shares below its dataset.

The entrypoints differ in the dataset they load, the label table they hand the
method and the protocol registry they read; everything that follows from the
*protocol* rather than the dataset -- the method's defaults, the preflight's
geodesic check, the oracle liveness rule, the frozen-configuration keys a
resume may differ in -- is one definition here, so the two cannot drift apart
in what they tell the one search algorithm.
"""
from __future__ import annotations

import os

#: How many validated starts must reproduce the publisher's own geodesic
#: distance to within a millimetre before the run is allowed to proceed. Not
#: 100%: a handful of starts sit exactly on a view point, where float32 and
#: navmesh snapping put the two numbers a hair apart.
GEODESIC_AGREEMENT_FRACTION = 0.99
#: A scene too small for one per cent to be a whole start -- HM3D v1 val runs 99
#: per scene, where the fraction admits none -- may still miss this many, each
#: off by no more than ``GEODESIC_MISS_MAX_M`` (since 2026-10-26). The
#: 2026-10-22 v1 sweep refused five of its twenty scenes at the gate for one to
#: three starts 0.016-0.090 m off with the median at 3 micrometres: snapping at
#: a view point, not another mesh, which moves the median by decimetres. One
#: stride (0.25 m) is the bound: a start a stride from where the publisher
#: measured it is not the same start.
GEODESIC_AGREEMENT_MIN_MISSES = 3
GEODESIC_MISS_MAX_M = 0.25

#: The frozen-config keys a ``--resume`` may differ in when the operator names them (``--resume-accepting-changes``):
#: provenance (``source_sha256``, ``gpu_plan``) and the kinematic self-check (``kinematics``, and ``evaluation``,
#: which nests it) -- none an input to a decision or a score, so the rows before and after are one experiment.
#: The 2026-10-22 HM3D v1 sweep lost two scenes mid-way to the self-check (00873 at its 69th episode, 00878 at its
#: second); finishing them under the corrected tolerance means exactly these differences, as Gibson's did on
#: 2026-10-11 (``gibson/run.py``). Anything else still refuses the resume.
RESUMABLE_CONFIG_CHANGES = ("kinematics", "evaluation", "source_sha256", "gpu_plan")

#: Packages the runtime needs, and the distribution each version is read from.
RUNTIME_PACKAGES = (
    ("numpy", "numpy"), ("scipy", "scipy"), ("skimage", "scikit-image"),
    ("cv2", "opencv-python"), ("networkx", "networkx"), ("requests", "requests"),
    ("quaternion", "numpy-quaternion"), ("habitat_sim", "habitat-sim"),
)

#: The scoring rule under which the STOP is taken on the deciding frame (``method_defaults``).
DISTANCE_TO_VIEW_POINTS = "view_points"


def method_defaults(protocol):
    """The ``RPTSettings`` fields a Habitat entrypoint sets below ``--policy-config``: the benchmark's own facts.

    The agent -- body height and radius -- and what its evaluator measures to.
    The closing's STOP overshoot (``TargetClosingSettings.stop_overshoot_*``)
    walks the margin between a near-edge reading and Gibson's FOOTPRINT
    evaluator; habitat-lab's ObjectNav (HM3D and MP3D alike) scores the
    geodesic distance to the goal's VIEW POINTS -- navigable spots within a
    metre of the object from which it is seen -- with a 0.1 m success radius
    (``protocol.distance_to``), so a decided STOP already stands on them and
    a step nearer the object can leave them. On the 70 recorded HM3D STOPs of
    2026-10-28 to 2026-11-05 the step never brought a frame from outside
    0.1 m to inside it, moved the agent farther from the view points in 18
    (by 2-14 cm) and nearer in 2, and walked 00891-cvZr5TUy5C5/000006 from
    0.02 m to 0.16 m against the toilet: a success into a failure. Under a
    view-point evaluator the planar overshoot is off and the STOP is taken on
    the deciding frame. A RAISED target keeps its own steps
    (``stop_overshoot_elevated_max_steps``, the MP3D finding of 2026-11-10:
    the view points of a wall cabinet or a picture stand closer, 0.84 m
    against 1.02 m), which ``TargetClosing._terminal`` reads apart. Every
    other method knob (clearance, stop distance, closing, peeks, ...) is
    ``RPTSettings``' default here exactly as it is on Gibson; nothing here
    came from a validation score.
    """
    defaults = {"body_height_m": protocol.agent_height_m,
                "body_radius_m": protocol.agent_radius_m}
    if protocol.distance_to == DISTANCE_TO_VIEW_POINTS:
        defaults["target_closing"] = {"stop_overshoot_steps": 0, "stop_overshoot_max_steps": 0}
    return defaults


def geodesic_agreement_issue(agreement, *, allow=False):
    """Why this run's shortest paths should not be trusted, or ``None``.

    ``info.geodesic_distance`` is the publisher's geodesic to the nearest goal
    view point -- the same quantity as ``l``. So the two disagreeing is not a
    difference of definition: it is our navmesh, our frames or our view points
    being wrong, and it would move every SPL without failing anything
    downstream. That makes it a preflight issue rather than a logged number.

    Args:
        agreement: What :meth:`HabitatObjNavEnv.geodesic_agreement` returned.
        allow: The operator has said the deviation is deliberate.

    Returns:
        The complaint, or ``None`` when the agreement is good enough -- at
        least ``GEODESIC_AGREEMENT_FRACTION`` of the starts within a
        millimetre, or no more than ``GEODESIC_AGREEMENT_MIN_MISSES`` of them
        outside it and none by more than ``GEODESIC_MISS_MAX_M`` -- or was
        explicitly accepted.
    """
    if agreement is None or allow:
        return None
    if agreement["within_1mm"] >= GEODESIC_AGREEMENT_FRACTION * agreement["episodes"]:
        return None
    misses = int(agreement["episodes"]) - int(agreement["within_1mm"])
    if misses <= GEODESIC_AGREEMENT_MIN_MISSES and agreement["max_abs_m"] <= GEODESIC_MISS_MAX_M:
        return None                       # a small scene's handful at a view point, each within a stride
    return ("Only %d of %d checked starts reproduce the publisher's own geodesic "
            "distance within 1 mm (median off by %.3f m, worst %.3f m): more than "
            "the %d starts a scene may have a hair off a view point, or one off by "
            "more than %.2f m. That is the same quantity as l, so this is most "
            "likely the wrong navmesh, and every SPL would be wrong with it. "
            "Investigate, or pass --allow-geodesic-mismatch to record the "
            "deviation and continue."
            % (agreement["within_1mm"], agreement["episodes"],
               agreement["median_abs_m"], agreement["max_abs_m"],
               GEODESIC_AGREEMENT_MIN_MISSES, GEODESIC_MISS_MAX_M))


def oracle_alive(live, row, limit=None):
    """Stop the run when the node oracle has answered nothing for ``limit`` episodes in a row.

    The search refuses a uniform distribution and explores the nearest
    frontier while the room LLM is away, so a dead oracle costs no episode
    -- and nothing announces that the run stopped measuring the method. The
    HM3D 2026-10-28 GT sweep ran 2,000 episodes that way. An episode counts
    as dead when its room LLM failed at least once and no oracle answer was
    used (``llm_queries`` + ``oracle_reuses`` both zero); one answer resets
    the count. ``limit`` defaults to ``ORACLE_DEAD_EPISODES`` (3); 0 disables.

    Raises:
        RuntimeError: ``limit`` dead episodes in a row; the rows written so far stand and
            ``--resume`` continues once the LLM service is fixed.
    """
    if limit is None:
        limit = int(os.environ.get("ORACLE_DEAD_EPISODES", "3"))
    info = (getattr(row, "agent_info", None) or {}).get("policy") or {}
    failures = int((((info.get("exploration_fallback") or {}).get("stats") or {}).get("room_llm_failures")) or 0)
    answered = int(info.get("llm_queries") or 0) + int(info.get("oracle_reuses") or 0)
    if answered > 0:
        live["oracle_dead"] = 0
    elif failures > 0:
        live["oracle_dead"] = live.get("oracle_dead", 0) + 1
    if limit > 0 and live.get("oracle_dead", 0) >= limit:
        raise RuntimeError("The node oracle answered nothing in %d episodes in a row (the last: %s); the run is no "
                           "longer measuring the method. Fix the LLM service and --resume."
                           % (live["oracle_dead"], getattr(row, "episode_id", "?")))
