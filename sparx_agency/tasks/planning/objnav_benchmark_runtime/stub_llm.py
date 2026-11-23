"""A fixed stand-in for the language model, in process: paired checks of changes that are not the model's.

A change to the closing, the planner or the map is measured by running the
same episodes before and after it. With the real model in the loop two such
runs differ for a second reason -- the oracle answers the same prompt
differently run to run (``LESSONS.md`` 2026-10-31) -- and need a GPU. This
module takes the model out: every room is labelled :data:`STUB_LABEL` with
confidence :data:`STUB_CONFIDENCE`, and the nodes of an oracle prompt share
:data:`STUB_TOTAL_PERCENT` equally (three nodes 30 % each, nine 10 %) --
no opinion about any of them. Nothing else of the method changes: the room
partition, the code-side rescaling of the numbers, RPT*, the visits, the
closing all run as they are, so the pipeline is deterministic and every
difference between two runs is the change under test.

These are the replies of the paired check of 2026-11-15 (the closing without
the navigation mesh), kept exactly, so that a later check is the SAME check
and its rows can be read against the earlier ones: on 2026-11-16 this module
reproduced that check's stored trajectories from the same tree.

What such a run is NOT: a result of the method. The search order it produces
is travel cost and exploration state alone, and the absolute success rate
means nothing; only paired differences on the same episodes do. The frozen
configuration names the stand-in (``method.llm_identity["stub"]``).

The stand-in is installed by replacing the runtime's verified LLM client
(``services/llm.py``) before the benchmark's own entrypoint builds the
policy, so the entrypoint, the preflight and the scoring are the real ones::

    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.stub_llm mp3d --version v1 ...

It can also be run BY PATH against another copy of the source tree (a frozen
snapshot of an earlier commit that does not hold this file), with that tree
on ``PYTHONPATH``::

    cd <snapshot> && PYTHONPATH=<snapshot> python <this file> mp3d --version v1 ...

Python 3.9 syntax (the Habitat environment), standard library only.
"""
from __future__ import annotations

import importlib
import os
import re
import sys

#: What the room labeller is answered with, for every room.
STUB_LABEL = "living_room"
STUB_CONFIDENCE = 0.8
#: What the nodes of one oracle prompt are answered with between them, percent: each gets an equal whole share.
STUB_TOTAL_PERCENT = 90
#: The node ids of an oracle prompt: every line that opens with ``id=<int>``.
NODE_ID = re.compile(r"^id=(\d+)", re.MULTILINE)
#: The benchmarks whose ``run`` module this can drive.
BENCHMARKS = ("mp3d", "hm3d", "gibson")
RUNTIME = "sparx_agency.tasks.planning.objnav_benchmark_runtime"
DESCRIPTION = ("in-process stand-in, no model: a boxed frontier gets p=0.15; historical room prompts: "
               "every room %s at confidence %.1f, the nodes of an oracle prompt "
               "an equal share of %d%%" % (STUB_LABEL, STUB_CONFIDENCE, STUB_TOTAL_PERCENT))


def stub_reply(system, user):
    """The stand-in's answer to one request, by the route its user prompt belongs to.

    A labeller prompt opens ``Room observed objects``. Anything else is an
    oracle prompt, which lists its nodes one per line, each opening
    ``id=<int>``: the ids share :data:`STUB_TOTAL_PERCENT` equally, in whole
    percent and never under 1, and what is left of 100 is ``elsewhere``. No
    reading and no ``pass`` verdict are written: the stand-in has no opinion.
    """
    if "Room observed objects" in user:
        return {"label": STUB_LABEL, "confidence": STUB_CONFIDENCE, "reasoning": "stub"}
    ids = [int(found) for found in NODE_ID.findall(user)]
    share = max(1, int(round(STUB_TOTAL_PERCENT / max(1, len(ids)))))
    return {"nodes": [{"id": nid, "why": "no opinion", "p": share} for nid in ids],
            "elsewhere": max(0, 100 - share * len(ids))}


class StubLLMClient:
    """Stands where ``services.llm.VerifiedLLMClient`` does: the same calls, no server behind them.

    Args:
        client: The real client the factory built; kept only for its
            config, which nothing here sends anywhere.
    """

    def __init__(self, client=None):
        self.client = client
        self.calls = 0

    def health(self):
        """The identity the frozen configuration records: a stand-in, by name."""
        return {"model": "stub", "reasoning_model": "stub", "digest": None, "immutable_revision_exposed": False,
                "models": {}, "stub": DESCRIPTION}

    def probe(self):
        """Nothing to prove: there is no server."""

    def chat_json(self, system, user, **kwargs):
        self.calls += 1
        if kwargs.get("image_jpeg") is not None:
            return {"probability": 0.15}
        return stub_reply(system, user)

    def set_answer_scope(self, scope):
        """No answer store: every answer is a function of its prompt."""

    def answer_stats(self):
        return None


def install():
    """Replace the runtime's verified LLM client with the stand-in, for every policy built after this call."""
    service = importlib.import_module(RUNTIME + ".services.llm")
    service.VerifiedLLMClient = StubLLMClient
    service.VerifiedRoomClient = StubLLMClient
    return service


def main(argv=None):
    """``<benchmark> [the benchmark's own run arguments]``: its entrypoint, with the stand-in installed."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in BENCHMARKS:
        sys.stderr.write("usage: stub_llm.py {%s} [run arguments]\n" % "|".join(BENCHMARKS))
        return 2
    install()
    run = importlib.import_module("%s.%s.run" % (RUNTIME, argv[0]))
    return run.main(argv[1:])


if __name__ == "__main__":
    # Run by path, this file's directory heads sys.path and its siblings (replay, recording, progress ...)
    # would shadow nothing useful and could shadow something else: the tree to run is PYTHONPATH's or the cwd's.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [entry for entry in sys.path if os.path.abspath(entry or os.getcwd()) != here]
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    raise SystemExit(main())
