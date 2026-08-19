"""Read what an acquired repository contains, without running any of it.

Separated from :mod:`..trt_optimizer.acquire` because *getting* the code and
*understanding* it are different jobs with different risks. Acquisition talks to
the network; this module only reads files.

The rule that shapes everything here: **never import the acquired code.**
Importing a stranger's repository executes its module-level statements, which is
how an inventory step ends up downloading weights, mutating a config, or worse.
So model definitions are found by parsing the source with :mod:`ast` and looking
at base-class names, checkpoints by suffix and size, and requirements by reading
the declaration files as text. Every answer here is therefore best-effort and
occasionally wrong -- which is the correct trade for a step whose whole purpose
is to tell you what you are about to deal with.
"""
from __future__ import annotations

import ast
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_LOG = logging.getLogger(__name__)

#: File suffixes that mean "trained weights" for the inventory. Deliberately
#: not ``.bin``: that matches far more non-weight files than it finds.
CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".engine")

#: File suffixes counted as configuration. A released checkpoint usually hides
#: several architectures behind one of these, so they are worth listing.
CONFIG_SUFFIXES = (".yaml", ".yml", ".json")

#: Files that declare an environment, in the order a human would read them.
REQUIREMENT_NAMES = ("pyproject.toml", "setup.py", "setup.cfg", "Pipfile",
                     "environment.yml", "environment.yaml")

#: Directories never worth walking: build output, caches, vendored installs.
SKIP_DIRS = frozenset({
    "__pycache__", "node_modules", "site-packages", "build", "dist",
    "venv", "env", "wandb", "outputs", "logs",
})

#: A source file larger than this is not a model definition anyone wrote by
#: hand; parsing it costs more than the answer is worth.
MAX_SOURCE_BYTES = 2 * 1024 * 1024

#: How much of a licence file is read. The identifying sentence is always in
#: the first screenful, and the restriction clause in the first few.
MAX_LICENSE_BYTES = 256 * 1024

#: How many list entries :func:`summarize` prints before saying "+N more".
SUMMARY_LIMIT = 8


@dataclass
class LicenseNote:
    """One line about the licence, and the flag that decides deployment.

    Args:
        name: the identified licence (``"MIT"``, ``"CC-BY-NC-4.0"``), or None.
        restricted: True when the text restricts use to non-commercial,
            research or evaluation purposes. This is a real deployment blocker
            and the reason the whole function exists.
        line: the one-line identification, written for a human.
        path: the licence file, relative to the code directory, or None.
    """

    name: Optional[str] = None
    restricted: bool = False
    line: str = ""
    path: Optional[str] = None


# --------------------------------------------------------------------------
# naming and the workspace
# --------------------------------------------------------------------------

_URL_RE = re.compile(r"^(?:[a-z][a-z0-9+.\-]*://|git@[^:/]+:)", re.IGNORECASE)
_BARE_HOST_RE = re.compile(r"^(?:www\.)?[a-z0-9.\-]+\.[a-z]{2,}/", re.IGNORECASE)

#: Path segments a browser adds to a repository URL. ``.../tree/main`` is what
#: a copy-paste from GitHub's address bar looks like, and taking its last
#: component would name every workspace "main".
WEB_VIEW_SEGMENTS = ("tree", "blob", "commit", "commits", "releases", "-")

HUB_HOSTS = ("huggingface.co", "hf.co")


def _walk(code_dir):
    """Yield every file under ``code_dir``, skipping caches and build output."""
    for dirpath, dirnames, filenames in os.walk(str(code_dir)):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            yield Path(dirpath) / name


def _relative(path, code_dir):
    """Path relative to the code directory, in POSIX form, for a report."""
    try:
        return Path(path).relative_to(Path(code_dir)).as_posix()
    except ValueError:
        return str(path)


def _read_text(path, limit=MAX_SOURCE_BYTES):
    """File text, or None when it is unreadable or too large to be worth it."""
    try:
        data = Path(path).read_bytes()
    except OSError as exc:  # noqa: BLE001  (an unreadable file is not fatal)
        _LOG.debug("skipping %s: %s", path, exc)
        return None
    if len(data) > limit:
        _LOG.debug("skipping %s: %d bytes exceeds the %d byte scan limit",
                   path, len(data), limit)
        return None
    return data.decode("utf-8", "replace")


def _top_level_file(code_dir, stems):
    """The first top-level file whose stem matches, case-insensitively."""
    directory = Path(code_dir)
    if not directory.is_dir():
        return None
    for entry in sorted(directory.iterdir()):
        if entry.is_file() and entry.stem.lower() in stems:
            return entry
    return None


def _base_name(node):
    """The dotted source text of a class base expression, best effort."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _base_name(node.value)
        return "%s.%s" % (prefix, node.attr) if prefix else node.attr
    if isinstance(node, ast.Call):
        return _base_name(node.func)
    if isinstance(node, ast.Subscript):
        return _base_name(node.value)
    return ""


def _module_classes(source):
    """Names of classes in ``source`` that derive from something Module-shaped.

    The test is textual and intentionally generous: any base whose final dotted
    component ends in ``Module`` counts, so ``nn.Module``, ``torch.nn.Module``,
    a bare ``Module`` and framework wrappers like ``LightningModule`` are all
    found -- as is a repo's own ``BaseModule``, which is usually where the real
    architecture lives.

    Returns:
        Sorted class names, empty when the file does not parse. A stranger's
        repository legitimately contains Python 2, Jinja templates named
        ``.py`` and deliberately broken test fixtures; refusing to inventory
        the whole tree because one file does not parse would be worse than the
        gap, so the skip is logged at debug level instead.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as exc:  # noqa: BLE001
        _LOG.debug("unparsed source in the model-def scan: %s", exc)
        return []
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            tail = _base_name(base).rsplit(".", 1)[-1]
            if tail.endswith("Module") and tail != "":
                found.add(node.name)
                break
    return sorted(found)


_MAIN_GUARD_RE = re.compile(r"__name__\s*==\s*['\"]__main__['\"]")


def _is_entrypoint(source):
    """True when a file has a ``__main__`` guard or builds an argument parser."""
    return bool(_MAIN_GUARD_RE.search(source)) or "ArgumentParser(" in source


def find_entrypoints(code_dir):
    """Best-effort inventory of a repository, without importing any of it.

    Args:
        code_dir: the directory holding the acquired code.

    Returns:
        A dict with these keys, every path relative to ``code_dir``:

        * ``readme`` / ``license``: the top-level file, or None.
        * ``requirements``: files declaring an environment.
        * ``configs``: YAML/JSON configuration -- a released checkpoint often
          hides several architectures behind one of these.
        * ``model_defs``: ``[{"path", "classes"}]`` for every ``.py`` whose
          source defines a Module subclass, found by AST scan.
        * ``checkpoints``: ``[{"path", "bytes"}]``, largest first.
        * ``entrypoints``: files with a ``__main__`` guard or an argparse
          parser -- where the deployed forward path is actually called from.

    Raises:
        ValueError: if ``code_dir`` is not a directory. An inventory of nothing
            that reports zero of everything reads exactly like a repository
            with no model in it.
    """
    directory = Path(code_dir)
    if not directory.is_dir():
        raise ValueError(
            "%s is not a directory, so there is nothing to inventory. Acquire "
            "the source first, or pass the checkout's path." % directory)

    readme = _top_level_file(directory, {"readme"})
    license_file = _license_file(directory)
    requirements, configs, checkpoints = [], [], []
    model_defs, entrypoints = [], []

    for path in _walk(directory):
        name, suffix = path.name, path.suffix.lower()
        relative = _relative(path, directory)
        if suffix in CHECKPOINT_SUFFIXES:
            try:
                checkpoints.append({"path": relative,
                                    "bytes": int(path.stat().st_size)})
            except OSError as exc:  # noqa: BLE001  (a broken symlink)
                _LOG.debug("skipping checkpoint %s: %s", path, exc)
            continue
        if (name.startswith("requirements") and suffix == ".txt") \
                or name in REQUIREMENT_NAMES:
            requirements.append(relative)
            continue
        if suffix in CONFIG_SUFFIXES:
            configs.append(relative)
            continue
        if suffix != ".py":
            continue
        source = _read_text(path)
        if source is None:
            continue
        classes = _module_classes(source)
        if classes:
            model_defs.append({"path": relative, "classes": classes})
        if _is_entrypoint(source):
            entrypoints.append(relative)

    checkpoints.sort(key=lambda item: (-item["bytes"], item["path"]))
    return {
        "readme": _relative(readme, directory) if readme else None,
        "license": _relative(license_file, directory) if license_file else None,
        "requirements": sorted(requirements),
        "configs": sorted(configs),
        "model_defs": sorted(model_defs, key=lambda item: item["path"]),
        "checkpoints": checkpoints,
        "entrypoints": sorted(entrypoints),
    }


# --------------------------------------------------------------------------
# requirements
# --------------------------------------------------------------------------

_REQ_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._\-]*)"
    r"(?:\[(?P<extras>[^\]]*)\])?"
    r"\s*(?P<spec>.*)$")

def _normalize(name):
    """PEP-503 name normalization, so ``Pillow`` and ``pillow`` are one thing."""
    return re.sub(r"[-_.]+", "-", str(name)).lower().strip()


def _installed_version(name):
    """Version of ``name`` in the *current* interpreter, or None."""
    from importlib import metadata as importlib_metadata

    for candidate in (name, _normalize(name), str(name).replace("-", "_")):
        try:
            return importlib_metadata.version(candidate)
        except Exception:  # noqa: BLE001  (absent, or a broken dist-info)
            continue
    return None


def _satisfies(installed, spec):
    """Does ``installed`` meet ``spec``? None when it cannot be decided.

    PEP 440 comparison is delegated to ``packaging`` rather than reimplemented:
    epochs, pre-releases, ``~=`` and wildcard pins all have exact rules that a
    hand-rolled tuple comparison gets subtly wrong, and a requirements report
    that is subtly wrong is worse than one that says it does not know.
    """
    if installed is None:
        return False
    if not spec:
        return True
    try:
        from packaging.specifiers import SpecifierSet
    except ImportError:
        _LOG.debug("packaging is absent; %r left undecided", spec)
        return None
    try:
        return bool(SpecifierSet(spec).contains(installed, prereleases=True))
    except Exception as exc:  # noqa: BLE001  (a poetry caret, a URL pin)
        _LOG.debug("undecidable specifier %r: %s", spec, exc)
        return None


def _parse_requirements_txt(text):
    """``(name, specifier)`` pairs from one requirements file.

    Comments, blank lines, pip options (``-r``, ``-e``, ``--index-url``) and
    environment markers are dropped; extras and ``name @ url`` forms are kept
    with the URL as the requirement, because "this pins a git fork" is exactly
    what the operator needs to see.
    """
    out = []
    for raw in str(text).splitlines():
        line = re.split(r"(?:^|\s)#", raw, maxsplit=1)[0].strip()
        if not line or line.startswith("-"):
            continue
        line = line.split(";", 1)[0].strip()
        if "@" in line and not line.startswith("@"):
            name, _, url = line.partition("@")
            if name.strip():
                out.append((name.strip().split("[")[0], "@ " + url.strip()))
                continue
        match = _REQ_RE.match(line)
        if match is None:
            continue
        out.append((match.group("name"), match.group("spec").strip()))
    return out


_PROJECT_DEPS_RE = re.compile(
    r"^\s*dependencies\s*=\s*\[(?P<body>.*?)\]", re.DOTALL | re.MULTILINE)
_POETRY_SECTION_RE = re.compile(
    r"^\s*\[tool\.poetry\.dependencies\]\s*$(?P<body>.*?)(?=^\s*\[|\Z)",
    re.DOTALL | re.MULTILINE)
_POETRY_LINE_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._\-]*)\s*=\s*(?P<spec>.+?)\s*$")


def _parse_pyproject(text):
    """``(name, specifier)`` pairs from a pyproject, read textually.

    Textually rather than with a TOML parser on purpose: ``tomllib`` does not
    exist before Python 3.11 and ``tomli`` is not a dependency of this repo, so
    a parser-based reader would work in one of this machine's four interpreters
    and raise in the others. Both the PEP-621 ``dependencies`` array and
    poetry's table are read; a caret or tilde constraint is reported verbatim
    and left undecided rather than silently reinterpreted.
    """
    out = []
    match = _PROJECT_DEPS_RE.search(str(text))
    if match:
        for item in re.findall(r"['\"]([^'\"]+)['\"]", match.group("body")):
            out.extend(_parse_requirements_txt(item))
    section = _POETRY_SECTION_RE.search(str(text))
    if section:
        for raw in section.group("body").splitlines():
            line = re.split(r"(?:^|\s)#", raw, maxsplit=1)[0].strip()
            if not line or line.startswith("["):
                continue
            entry = _POETRY_LINE_RE.match(line)
            if entry is None:
                continue
            spec = entry.group("spec").strip().strip("'\"")
            if spec.startswith("{"):
                spec = ""
            out.append((entry.group("name"), spec))
    return out


def requirements_report(code_dir):
    """What this repo asks for, against what the current interpreter has.

    Reports; installs nothing. Closing the gap is a stop-and-ask, because on
    this machine the answer is usually "run it in a different interpreter"
    rather than "pip install into this one".

    Args:
        code_dir: the acquired code directory.

    Returns:
        Sorted list of ``(package, required, installed_or_None, satisfied)``.
        ``required`` is the specifier verbatim (``">=2.1"``, ``"@ git+https://
        ..."``, ``"^1.4"``) or ``""``. ``satisfied`` is False when the package
        is absent from this interpreter, True when it is present and meets the
        specifier, and **None** when it is present but the specifier cannot be
        decided -- a poetry caret, a URL pin, or anything ``packaging``
        rejects. None is not False and must not be rendered as one.

    Raises:
        ValueError: if ``code_dir`` is not a directory.
    """
    directory = Path(code_dir)
    if not directory.is_dir():
        raise ValueError(
            "%s is not a directory; nothing to read requirements from."
            % directory)

    pairs = []
    for path in _walk(directory):
        text = None
        if path.name.startswith("requirements") and path.suffix.lower() == ".txt":
            text = _read_text(path)
            pairs.extend(_parse_requirements_txt(text or ""))
        elif path.name == "pyproject.toml":
            text = _read_text(path)
            pairs.extend(_parse_pyproject(text or ""))

    rows, seen = [], set()
    for name, spec in pairs:
        key = (_normalize(name), spec)
        if key in seen:
            continue
        seen.add(key)
        installed = _installed_version(name)
        rows.append((name, spec, installed, _satisfies(installed, spec)))
    return sorted(rows, key=lambda row: (_normalize(row[0]), row[1]))


# --------------------------------------------------------------------------
# licence
# --------------------------------------------------------------------------

#: Substrings that identify a licence, checked in order against the lowercased
#: text. Ordered so the more specific text wins: an NVIDIA or Llama licence
#: quotes Apache, and AGPL/LGPL both contain "general public license".
_LICENSE_MARKERS = (
    ("nvidia source code license", "NVIDIA Source Code License"),
    ("llama 2 community license", "Llama-2-Community"),
    ("llama 3 community license", "Llama-3-Community"),
    ("attribution-noncommercial", "CC-BY-NC-4.0"),
    ("creative commons", "CC"),
    ("gnu affero general public", "AGPL-3.0"),
    ("gnu lesser general public", "LGPL"),
    ("gnu general public license", "GPL"),
    ("mozilla public license", "MPL-2.0"),
    ("apache license", "Apache-2.0"),
    ("mit license", "MIT"),
    ("permission is hereby granted, free of charge", "MIT"),
    ("this is free and unencumbered software", "Unlicense"),
    ("permission to use, copy, modify, and/or distribute", "ISC"),
    ("redistribution and use in source and binary forms", "BSD"),
)

#: Phrases that make a licence a deployment blocker. Every one of them has
#: appeared verbatim on a released model checkpoint.
_RESTRICTED_MARKERS = (
    "noncommercial", "non-commercial", "non commercial",
    "not for commercial", "no commercial use", "commercial use is prohibited",
    "research purposes only", "research purpose only", "research use only",
    "research-only", "academic research only", "academic purposes only",
    "evaluation purposes only", "internal evaluation only",
    "non-production", "not for production",
)

#: Filenames that hold a licence, matched on the stem, case-insensitively.
_LICENSE_STEMS = frozenset({"license", "licence", "copying", "license-model",
                            "model_license", "license_model"})


def _license_file(code_dir):
    """The top-level licence file, or None."""
    return _top_level_file(code_dir, _LICENSE_STEMS)


def license_note(code_dir):
    """Identify the licence and flag anything that blocks deployment.

    One line to check, and the one that can stop the whole exercise after the
    engines are built. A non-commercial or research-only licence is not a
    footnote: it means the accelerated model cannot ship, however good the
    numbers are.

    Args:
        code_dir: the acquired code directory.

    Returns:
        A :class:`LicenseNote`. A missing licence file is reported loudly in
        ``line`` but leaves ``restricted`` False -- the flag means "the text
        forbids commercial use", and absence of a text is a different problem
        (unlicensed code is all rights reserved by default) that a human has to
        take to the upstream author.
    """
    path = _license_file(code_dir)
    if path is None:
        return LicenseNote(
            name=None, restricted=False, path=None,
            line=("no LICENSE/COPYING file found -- code published without a "
                  "licence is all rights reserved by default. Confirm the "
                  "terms with the upstream author before shipping anything "
                  "built from it."))
    text = _read_text(path, limit=MAX_LICENSE_BYTES) or ""
    haystack = " ".join(text.lower().split())
    name = None
    for marker, identified in _LICENSE_MARKERS:
        if marker in haystack:
            name = identified
            break
    if name == "GPL" and "version 3" in haystack:
        name = "GPL-3.0"
    if name == "BSD":
        name = "BSD-3-Clause" if "neither the name" in haystack \
            else "BSD-2-Clause"
    hits = [marker for marker in _RESTRICTED_MARKERS if marker in haystack]
    relative = _relative(path, code_dir)
    if hits:
        label = name or "custom licence"
        return LicenseNote(
            name=name or "custom", restricted=True, path=relative,
            line=("%s: %s, RESTRICTED -- the text says %s. This is a "
                  "deployment blocker, not a footnote: settle it before "
                  "spending the optimization effort."
                  % (relative, label,
                     ", ".join(repr(hit) for hit in hits))))
    if name is None:
        return LicenseNote(
            name=None, restricted=False, path=relative,
            line=("%s: unidentified licence text -- read it before deploying; "
                  "it matches none of the standard licences this recognises."
                  % relative))
    return LicenseNote(
        name=name, restricted=False, path=relative,
        line="%s: %s, no non-commercial restriction found." % (relative, name))


# --------------------------------------------------------------------------
# the operator's block
# --------------------------------------------------------------------------

def _human_bytes(count):
    """Bytes as a human string, three significant figures, never scientific."""
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return "%.4g %s" % (value, unit)
        value /= 1024.0
    return "%.4g TiB" % value


def _listing(items, limit=SUMMARY_LIMIT):
    """A comma-separated head of ``items`` with an honest "+N more" tail."""
    items = list(items)
    if not items:
        return "none"
    head = ", ".join(str(i) for i in items[:limit])
    extra = len(items) - limit
    return head if extra <= 0 else "%s (+%d more)" % (head, extra)


def summarize(source, inventory):
    """A plain-text block for the operator, and for ``NOTES.md``.

    Written to be pasted into the workspace's notes unedited: it names what was
    acquired, what is in it, what the licence permits, and -- explicitly -- the
    two things this phase deliberately did not do, so the next reader does not
    assume an environment exists.

    Args:
        source: the :class:`Source` from :func:`acquire`.
        inventory: the dict from :func:`find_entrypoints`.

    Returns:
        A multi-line string, no trailing newline.
    """
    code = source.code_dir
    lines = [
        "Source",
        "  kind:       %s" % source.kind,
        "  origin:     %s" % source.url_or_path,
        "  slug:       %s" % source.slug,
        "  workspace:  %s" % source.workspace,
        "  code:       %s%s" % (code, "" if code.exists() else "  (not present)"),
        "  commit:     %s" % (source.commit or "unknown"),
    ]

    note = license_note(code) if code.is_dir() else None
    lines.append("  license:    %s" % (note.line if note is not None
                                       else source.license_name or "unknown"))
    if note is not None and note.restricted:
        lines.append("  ** RESTRICTED LICENCE -- settle this before optimizing **")

    checkpoints = inventory.get("checkpoints") or []
    weights = ["%s (%s)" % (c["path"], _human_bytes(c["bytes"]))
               for c in checkpoints]
    defs = ["%s [%s]" % (d["path"], ", ".join(d["classes"]))
            for d in (inventory.get("model_defs") or [])]
    lines.extend([
        "",
        "Inventory",
        "  readme:       %s" % (inventory.get("readme") or "none"),
        "  requirements: %s" % _listing(inventory.get("requirements") or []),
        "  configs:      %s" % _listing(inventory.get("configs") or []),
        "  model defs:   %s" % _listing(defs),
        "  checkpoints:  %s" % _listing(weights),
        "  entrypoints:  %s" % _listing(inventory.get("entrypoints") or []),
        "",
        "Not done here, deliberately",
        "  - nothing was installed: run requirements_report() and decide which",
        "    interpreter this belongs in before any pip install.",
        "  - no weights were downloaded: the checkpoint is the largest and most",
        "    license-encumbered artifact here, and it is your call.",
        "  - nothing in the repository was imported or executed; the inventory",
        "    above is an AST scan, so a class it missed is a scan gap, not proof",
        "    the model is not there.",
    ])
    if not checkpoints:
        lines.append("  - no weights are present, so nothing can be profiled yet.")
    return "\n".join(lines)
