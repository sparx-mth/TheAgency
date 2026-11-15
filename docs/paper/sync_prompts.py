#!/usr/bin/env python3
"""Copy the method's two LLM prompts from the code into ``zson_paper.tex``, verbatim.

Appendix C of the paper must show the prompts the method actually sends, so
they are never typed into the paper by hand. This script imports them from the
code and rewrites the regions of the ``.tex`` file between
``%%PROMPT-BEGIN:<name>%%`` and ``%%PROMPT-END:<name>%%``.

Run it whenever a prompt changes (from the repository root or anywhere else)::

    python docs/paper/sync_prompts.py            # rewrite the listings
    python docs/paper/sync_prompts.py --check    # exit 1 if the paper is stale

Sources:
    sparx_agency/core/mapping/topology/room_classifier.py     (room labels)
    sparx_agency/core/mapping/topology/search_node_oracle.py  (per-room probabilities)

The supplements of the oracle's system prompt (``STAIRS_SUPPLEMENT``,
``OPENINGS_SUPPLEMENT``) belong to configurations the paper does not report
and are deliberately not copied.
"""
from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().with_name("zson_paper.tex")
REPO = Path(__file__).resolve().parents[2]


def prompts() -> dict:
    """``{marker name: prompt text}``, read from the code."""
    sys.path.insert(0, str(REPO))
    from sparx_agency.core.mapping.topology import room_classifier, search_node_oracle

    # The label set the classifier is built with when no caller overrides it (the method does not).
    label_set = inspect.signature(room_classifier.RoomTypeClassifier.__init__).parameters["label_set"].default
    return {
        "labeller-system": room_classifier.SYSTEM_PROMPT_TEMPLATE.format(label_set=", ".join(label_set)),
        "labeller-user": room_classifier.USER_PROMPT_TEMPLATE,
        "oracle-system": search_node_oracle.SYSTEM_PROMPT,
        "oracle-user": search_node_oracle.USER_PROMPT_TEMPLATE,
    }


def listing(text: str) -> str:
    """``text`` as a LaTeX listing in the paper's ``prompt`` style."""
    if not text.isascii():
        raise ValueError("a prompt contains non-ASCII characters; the listings package cannot typeset them as is")
    if "\\end{lstlisting}" in text:
        raise ValueError("a prompt contains the listing terminator")
    return "\\begin{lstlisting}[style=prompt]\n%s\n\\end{lstlisting}\n" % text.strip("\n")


def render(tex: str) -> str:
    """``tex`` with every prompt region replaced by the current prompt."""
    for name, text in prompts().items():
        pattern = re.compile(r"(%%PROMPT-BEGIN:" + re.escape(name) + r"%%\n).*?(%%PROMPT-END:" + re.escape(name) + r"%%)",
                             re.DOTALL)
        if len(pattern.findall(tex)) != 1:
            raise ValueError("expected exactly one %%PROMPT-BEGIN:%s%% ... %%PROMPT-END:%s%% region in %s"
                             % (name, name, PAPER.name))
        block = listing(text)
        tex = pattern.sub(lambda match: match.group(1) + block + match.group(2), tex)
    return tex


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="do not write; exit 1 if the paper's listings are stale")
    args = parser.parse_args()
    current = PAPER.read_text(encoding="utf-8")
    updated = render(current)
    if args.check:
        if updated != current:
            print("%s: prompt listings are STALE; run docs/paper/sync_prompts.py" % PAPER.name)
            return 1
        print("%s: prompt listings match the code" % PAPER.name)
        return 0
    if updated != current:
        PAPER.write_text(updated, encoding="utf-8")
        print("%s: prompt listings rewritten from the code" % PAPER.name)
    else:
        print("%s: prompt listings already match the code" % PAPER.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())

