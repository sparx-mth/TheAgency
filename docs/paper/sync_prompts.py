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
    sparx_agency/core/mapping/topology/room_classifier.py      (room labels)
    sparx_agency/core/mapping/topology/search_node_prompts.py  (per-room probabilities: the DEFAULT prompt version)

The oracle's prompt exists in named versions, one per language model
(``search_node_prompts.py``). The paper lists the default one, and the region
``oracle-version`` says which that is -- its name, the model it was written
for and the digest of its text -- so the paper cannot show one version under
another's name. The one supplement of the oracle's system prompt (for exits
offered as nodes of their own) belongs to a configuration the paper does not
report and is deliberately not copied.
"""
from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().with_name("zson_paper.tex")
REPO = Path(__file__).resolve().parents[2]


def oracle_version():
    """The default version of the oracle's prompt (``search_node_prompts.NodeOraclePrompt``)."""
    sys.path.insert(0, str(REPO))
    from sparx_agency.core.mapping.topology import search_node_prompts
    return search_node_prompts.PROMPTS[search_node_prompts.DEFAULT_PROMPT]


def prompts() -> dict:
    """``{marker name: prompt text}``, read from the code."""
    sys.path.insert(0, str(REPO))
    from sparx_agency.core.mapping.topology import room_classifier

    # The label set the classifier is built with when no caller overrides it (the method does not).
    label_set = inspect.signature(room_classifier.RoomTypeClassifier.__init__).parameters["label_set"].default
    version = oracle_version()
    return {
        "labeller-system": room_classifier.SYSTEM_PROMPT_TEMPLATE.format(label_set=", ".join(label_set)),
        "labeller-user": room_classifier.USER_PROMPT_TEMPLATE,
        "oracle-system": version.system,
        "oracle-user": version.user_template,
    }


def statements() -> dict:
    """``{marker name: LaTeX text}`` -- regions that are a sentence of the paper, not a listing."""
    version = oracle_version()

    def code(value: str) -> str:
        return "\\code{%s}" % value.replace("_", "\\_")
    return {"oracle-version": "Prompt version %s, written for %s; SHA-256 of its text %s\\ldots{} System prompt.\n"
                              % (code(version.name), code(version.tuned_on), code(version.sha256[:16]))}


def listing(text: str) -> str:
    """``text`` as a LaTeX listing in the paper's ``prompt`` style."""
    if not text.isascii():
        raise ValueError("a prompt contains non-ASCII characters; the listings package cannot typeset them as is")
    if "\\end{lstlisting}" in text:
        raise ValueError("a prompt contains the listing terminator")
    return "\\begin{lstlisting}[style=prompt]\n%s\n\\end{lstlisting}\n" % text.strip("\n")


def render(tex: str) -> str:
    """``tex`` with every prompt region replaced by the current prompt, and every statement region by its sentence."""
    blocks = {name: listing(text) for name, text in prompts().items()}
    blocks.update(statements())
    for name, block in blocks.items():
        pattern = re.compile(r"(%%PROMPT-BEGIN:" + re.escape(name) + r"%%\n).*?(%%PROMPT-END:" + re.escape(name) + r"%%)",
                             re.DOTALL)
        if len(pattern.findall(tex)) != 1:
            raise ValueError("expected exactly one %%PROMPT-BEGIN:%s%% ... %%PROMPT-END:%s%% region in %s"
                             % (name, name, PAPER.name))
        tex = pattern.sub(lambda match, block=block: match.group(1) + block + match.group(2), tex)
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

