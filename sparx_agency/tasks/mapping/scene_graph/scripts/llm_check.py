#!/usr/bin/env python3
"""End-to-end sanity check for the scene-graph LLM backend, no ROS.

Run this BEFORE launching the pipeline to verify that:
  1. The configured LLM backend (Ollama or OpenAI-compat) is reachable.
  2. The classifier prompt produces a sensible in-set room label.
  3. The search oracle produces a probability vector over every room
     that normalizes to ~1.0 (source='llm', not the uniform fallback).

Usage::

    .venv/bin/python -m sparx_agency.tasks.mapping.scene_graph.scripts.llm_check
    .venv/bin/python -m ...scripts.llm_check --target "car keys"
    LLM_MODEL=llama3.2:3b .venv/bin/python -m ...scripts.llm_check

Exits with code 0 on full success, 1 on any check failure. Intended
for use in CI / smoke-test scripts too.
"""

from __future__ import annotations

import argparse
import sys
import time

from sparx_agency.core.mapping.topology.llm_client import LLMClient
from sparx_agency.core.mapping.topology.room_classifier import (
    DEFAULT_LABEL_SET,
    SYSTEM_PROMPT_TEMPLATE as CLS_SYS,
    USER_PROMPT_TEMPLATE as CLS_USER,
    format_object_list,
)
from sparx_agency.core.mapping.topology.search_oracle import (
    OracleRoom,
    SearchOracle,
)


def _ok(s: str) -> None:
    print(f"  \033[32m✓\033[0m {s}")


def _fail(s: str) -> None:
    print(f"  \033[31m✗\033[0m {s}")


def _hdr(s: str) -> None:
    print(f"\n── {s} ──")


def check_ping(llm: LLMClient) -> bool:
    """Check 1: the configured backend answers at all."""
    _hdr("1. Ping LLM server")
    print(f"  backend={llm.cfg.backend}  url={llm.cfg.base_url}  "
          f"model={llm.cfg.model}")
    if llm.ping():
        _ok("server answered")
        return True
    _fail(f"server at {llm.cfg.base_url} did NOT answer")
    print("    fixes to try:")
    print("      - is Ollama running?         ollama serve &")
    print("      - is the model pulled?       ollama pull qwen2.5:3b-instruct")
    print("      - is the port reachable?     curl http://localhost:11434/api/tags")
    return False


def check_classifier(llm: LLMClient) -> bool:
    """Check 2: the classifier prompt yields an in-set label."""
    _hdr("2. Room classifier prompt")
    objs = ["refrigerator", "sink", "refrigerator", "dining_table", "oven"]
    sys_p = CLS_SYS.format(label_set=", ".join(DEFAULT_LABEL_SET))
    user_p = CLS_USER.format(obj_list=format_object_list(objs))
    print(f"  objects: {objs}")

    t0 = time.time()
    try:
        reply = llm.chat_json(sys_p, user_p)
    except Exception as e:
        _fail(f"call failed: {e}")
        return False
    print(f"  round-trip: {time.time() - t0:.2f}s")
    print(f"  reply: {reply}")

    label = str(reply.get("label", "")).lower()
    if label in DEFAULT_LABEL_SET:
        _ok(f"label '{label}' is in the configured label set")
    else:
        _fail(f"label '{label}' is NOT in the label set "
              "(RoomTypeClassifier would coerce it to 'unknown')")
        return False
    # Soft check: a kitchen-like prompt ideally returns 'kitchen'.
    # Not strictly required; small models sometimes say dining_room.
    if label == "kitchen":
        _ok("label is 'kitchen' as expected for these objects")
    else:
        print(f"  note: expected 'kitchen', got '{label}'. Usually OK, "
              "but if this is consistently wrong consider a larger model.")
    return True


def check_oracle(llm: LLMClient, target: str) -> bool:
    """Check 3: the oracle yields a normalized in-order distribution."""
    _hdr("3. LLM search oracle")
    # A small synthetic scene: one kitchen-like room lightly searched,
    # one bedroom-like room heavily searched, one empty hallway. We
    # mainly verify the RESULT SHAPE here, not the specific numbers.
    rooms = [
        OracleRoom(id=0, label="kitchen", searched_s=5.0,
                   frontier_clusters=3,
                   observed_classes=("refrigerator", "sink")),
        OracleRoom(id=1, label="bedroom", searched_s=60.0,
                   frontier_clusters=1,
                   observed_classes=("bed", "nightstand")),
        OracleRoom(id=2, label="hallway", searched_s=0.0,
                   frontier_clusters=2),
    ]
    print(f"  target: {target!r}")

    t0 = time.time()
    try:
        result = SearchOracle(llm).probabilities(target, rooms)
    except Exception as e:
        _fail(f"call failed: {e}")
        return False
    print(f"  round-trip: {time.time() - t0:.2f}s")
    print(f"  raw reply: {result.raw_reply}")

    if result.source != "llm":
        _fail(f"oracle degraded to source={result.source!r} — the reply "
              "was unusable (missing rooms list or all-zero probs)")
        return False
    _ok("reply parsed as a usable LLM distribution")

    missing = [r.id for r in rooms if r.id not in result.probs]
    if missing:
        _fail(f"rooms {missing} missing from the probability vector")
        return False
    _ok("every input room got a probability")

    total = sum(result.probs.values())
    if abs(total - 1.0) > 1e-6:
        _fail(f"probabilities sum to {total:.6f}, expected 1.0")
        return False
    top = sorted(result.probs.items(), key=lambda kv: -kv[1])
    top_str = ", ".join(f"R{k}={v:.2f}" for k, v in top)
    print(f"  normalized: {top_str}   (sum = 1.00)")
    _ok("probabilities normalize cleanly")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", default="car keys",
                    help="target object for the oracle test")
    args = ap.parse_args()

    print("\n=== LLM backend sanity check ===")
    llm = LLMClient.from_env()
    passed = [check_ping(llm)]
    if not passed[-1]:
        print("\nStopping — server unreachable means the next checks will "
              "all fail the same way.")
        sys.exit(1)
    passed.append(check_classifier(llm))
    passed.append(check_oracle(llm, args.target))

    print()
    if all(passed):
        print("\033[32mAll checks passed.\033[0m "
              "Launching the pipeline should work.")
        sys.exit(0)
    print("\033[31mOne or more checks failed.\033[0m See messages above.")
    sys.exit(1)


if __name__ == "__main__":
    main()
