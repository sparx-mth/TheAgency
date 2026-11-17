"""Probe the node oracle against a REAL model: latency, schema, and whether its own judgement is a sensible one.

Not a test -- it needs a live Ollama and takes minutes on a CPU. Run from the
repo root::

    LLM_BASE_URL=http://127.0.0.1:11434 LLM_REASONING_MODEL=<model> PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \\
        .venv/bin/python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.probe_node_oracle \\
        [--prompt VERSION] [--set P|O|PO] [--only P4,O2] [--repair] [out.json]

``LLM_REASONING_MODEL`` names the model asked (the oracle's route);
``LLM_NUM_GPU=0`` keeps it on the CPU -- the way to probe on a machine whose
card is in use (``/api/ps`` then shows ``size_vram`` 0; check the service's
memory first, ``LESSONS.md`` 2026-11-16). ``--prompt`` names the prompt
version asked under (``core/mapping/topology/search_node_prompts.py``; the
default version otherwise): **this is the tool a wording is tuned with** --
copy a version, probe it on the model it is for, reword, probe again.

**What it checks.** Since 2026-11-16 the prompt states the task, the meaning
of the fields and the reply format -- and, since 2026-11-17, what the search
so far means for a room, in words -- and no knowledge of ours: where the
target belongs is the model's own to work out. So the probe does not check
that rules were followed. Each
scenario states ONE coarse, ordinal claim that a reasoner who knows homes
would not get wrong -- which of two nodes ranks above the other -- and the
probe reports whether the model's numbers honour it. The claims are ours and
are never shown to the model: they measure the model, they do not brief it.

**Two sets** (``probe_node_oracle_scenarios.py``). ``P`` (twelve) is what a
run sends by default: rooms only, the mapped floor on every line, finished
rooms not offered. ``O`` (seven) is the first probe's
set, kept so results stay comparable: finished rooms offered, two openings as
nodes of their own (the deprecated mode), and five scenarios on lines without
size detail.

**Read the numbers for what they are**: a few hand-written scenarios, one
seed, temperature 0. A small model is brittle to wording -- one sentence
moves three scenarios -- so a difference of one or two claims between two
wordings is noise; the replies (``--json`` keeps them) say more than the
count. The figures recorded so far are in
``core/mapping/topology/README.md``.

``--repair`` asks through ``RepairingNodeOracle``, as a run does: a reply
that breaks the schema gets one repair call before it counts as unusable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from sparx_agency.core.mapping.topology.llm_client import LLMClient, LLMConfig
from sparx_agency.core.mapping.topology.search_node_oracle import SearchNodeOracle
from sparx_agency.core.mapping.topology.search_node_prompts import DEFAULT_PROMPT, PROMPTS, prompt_version
from sparx_agency.tasks.planning.objnav_benchmark_runtime.tests.probe_node_oracle_scenarios import SETS


def run(oracle, scenarios, verbose=True):
    """Every scenario through ``oracle``: ``[{name, target, seconds, source, pass_verdict, reading, nodes, claim, holds}]``."""
    out = []
    for name, target, ctx, nodes, claim, check in scenarios:
        user = oracle.prompt(target, nodes, ctx)
        if verbose:
            print("\n=== %s ===" % name)
            print(user)
        oracle.remember(None, None)                               # no reuse across scenarios
        started = time.monotonic()
        result = oracle.probabilities(target, nodes, ctx)
        seconds = time.monotonic() - started
        probs = {int(k): float(v) for k, v in result.probs.items()}
        holds = result.source == "llm" and bool(check(probs))
        out.append({"name": name, "target": target, "system_chars": len(oracle.system(nodes, ctx)), "user_prompt": user,
                    "seconds": round(seconds, 1), "source": result.source, "pass_verdict": result.pass_verdict,
                    "omitted": list(result.omitted), "reading": dict(result.reading), "claim": claim, "holds": holds,
                    "raw_reply": result.raw_reply,
                    "nodes": [{"id": n.id, "label": n.label, "p": round(probs.get(n.id, 0.0), 3),
                               "why": result.reasons.get(n.id, "")} for n in nodes]})
        if verbose:
            print("--- %.1f s | source=%s | pass=%s | omitted=%s" % (seconds, result.source, result.pass_verdict,
                                                                   list(result.omitted)))
            for key, value in result.reading.items():
                print("  %s: %s" % (key, value))
            for item in nodes:
                print("  id=%-7d p=%.2f  %s" % (item.id, probs.get(item.id, 0.0), result.reasons.get(item.id, "")))
            print("  -> %s: %s" % ("HOLDS" if holds else "DOES NOT HOLD", claim))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", nargs="?", help="write every scenario, prompt and reply here as JSON")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, choices=sorted(PROMPTS),
                        help="the prompt version to ask under (default: %(default)s)")
    parser.add_argument("--set", default="P", choices=sorted(SETS), dest="scenario_set",
                        help="P: what a run sends by default; O: the first probe's seven; PO: both")
    parser.add_argument("--only", help="comma-separated scenario names' first words (P4,O2)")
    parser.add_argument("--repair", action="store_true", help="ask as a run does: one schema-repair call on a broken reply")
    parser.add_argument("--quiet", action="store_true", help="one line per scenario")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    cfg = LLMConfig.from_env()
    if os.environ.get("LLM_BASE_URL") is None:
        cfg.base_url = "http://127.0.0.1:11434"
    client = LLMClient(cfg)
    version = prompt_version(args.prompt)
    print("service %s | model %s (timeout %.0fs, num_ctx %d, num_gpu %s) | prompt %s, tuned on %s, %d characters, sha256 %s"
          % (cfg.base_url, cfg.reasoning_model, cfg.reasoning_timeout_s, cfg.reasoning_num_ctx, cfg.num_gpu, version.name,
             version.tuned_on, len(version.system), version.sha256[:12]))
    if not client.ping():
        print("no LLM service answers; start Ollama first")
        return 2
    scenarios = SETS[args.scenario_set]
    if args.only:
        wanted = tuple(args.only.split(","))
        scenarios = [s for s in scenarios if s[0].split()[0] in wanted]
    if args.repair:
        from sparx_agency.core.planning.object_search.rooms.oracle_retry import RepairingNodeOracle
        oracle = RepairingNodeOracle(client, prompt=version)
    else:
        oracle = SearchNodeOracle(client, prompt=version)
    results = run(oracle, scenarios, verbose=not args.quiet)
    held = sum(1 for s in results if s["holds"])
    usable = sum(1 for s in results if s["source"] == "llm")
    if args.quiet:
        for s in results:
            print("%-74s %5.1fs  %-13s %s" % (s["name"][:74], s["seconds"], "holds" if s["holds"] else "DOES NOT HOLD",
                                               " ".join("%d:%.2f" % (n["id"], n["p"]) for n in s["nodes"])))
    print("\n%d of %d claims hold; %d of %d replies usable%s" % (held, len(results), usable, len(results),
                                                                " (after repair)" if args.repair else ""))
    if args.out:
        with open(args.out, "w") as stream:
            json.dump({"model": cfg.reasoning_model, "num_gpu": cfg.num_gpu, "seed": cfg.seed, "prompt": version.describe(),
                       "system_prompt": version.system, "set": args.scenario_set, "repair": bool(args.repair),
                       "scenarios": results}, stream, indent=1)
    return 0 if held == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
