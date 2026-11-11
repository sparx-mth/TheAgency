"""What every dataset's scene campaign shares below its dataset: the child-argument rules and the episode plan.

A campaign (``hm3d/campaign.py``, ``mp3d/campaign.py``, ``mp3d/sharded_campaign.py``)
runs one ``run.py`` child per scene and hands each the operator's extra
arguments; these helpers decide what each child may carry and which episodes
each scene's child runs. Nothing here knows a dataset.
"""
from __future__ import annotations


#: The child flag that only a resuming child may carry (``run.py``: it requires ``--resume``).
RESUME_ACCEPTANCE_FLAG = "--resume-accepting-changes"


def split_resume_acceptance(extra):
    """``(rest, acceptance)``: the child arguments without ``--resume-accepting-changes KEY...``, and that group.

    The campaign hands ``extra`` to every child unchanged, but the acceptance
    is a resume's: ``run.py`` refuses it on a fresh scene. So a resumed root
    gives it to the scenes that continue and withholds it from the ones that
    start, and the operator writes the launcher's command line once.
    """
    rest, acceptance = [], []
    tokens = list(extra)
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == RESUME_ACCEPTANCE_FLAG:
            acceptance.append(token)
            index += 1
            while index < len(tokens) and not str(tokens[index]).startswith("--"):
                acceptance.append(tokens[index])
                index += 1
            continue
        rest.append(token)
        index += 1
    return rest, acceptance


def episode_plan(episode_ids, known_scenes):
    """``{scene: [episode id, ...]}`` from a flat list of ``<scene>/<index>`` ids, scenes in order of first appearance.

    Raises:
        ValueError: An id is malformed, names a scene the split does not hold, or repeats.
    """
    known = set(known_scenes)
    plan, seen = {}, set()
    for episode_id in episode_ids:
        text = str(episode_id)
        scene, sep, index = text.partition("/")
        if not sep or not index or scene not in known:
            raise ValueError("episode id %r is not <scene>/<index> of a scene of this split" % text)
        if text in seen:
            raise ValueError("episode id %r is listed twice" % text)
        seen.add(text)
        plan.setdefault(scene, []).append(text)
    return plan
