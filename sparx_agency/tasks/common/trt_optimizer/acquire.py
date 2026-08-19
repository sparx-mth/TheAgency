"""Phase 0 of the toolkit: get the model onto this machine and say what arrived.

Every other stage in this package starts from a model object that somebody has
already built. The operator's workflow starts one step earlier and one level
lower: they hand over a GitHub URL or a local path and expect the pipeline to
take it from there. This module is that step, and it is deliberately the least
clever module in the package -- its whole value is in the four things it
refuses to do.

**It never imports or executes the acquired code.** The inventory is built by
reading bytes and parsing an AST. Importing a stranger's repository to find out
what is inside it runs whatever that repository's module scope decides to run,
which on a research checkout routinely means a CUDA probe, a ``sys.path`` edit,
or a multi-gigabyte weight download triggered by a default argument. An AST scan
finds ``class Policy(nn.Module)`` perfectly well and cannot execute anything, so
the scan is the only mechanism offered here.

**It never installs anything.** :func:`requirements_report` compares what the
repo asks for against what the current interpreter actually has and prints the
difference. Choosing to close that gap -- a pip install, a new conda env -- is
the user's decision and frequently the wrong one, because this repo runs four
interpreters and the right answer is often "use a different one".

**It never downloads weights.** A checkpoint is the largest and most
license-encumbered artifact in the exercise, and it is the user's call.

**It never writes inside this repository.** The workspace is ``~/trt/<slug>/``
by default (override with ``$TRT_WORKSPACE``) because a cloned repo or a
multi-gigabyte checkpoint under the source tree eventually ends up in somebody's
commit, and ``.gitignore`` only covers the patterns it already knows about.

Nothing here knows what kind of network it is looking at. A classifier, a
detector, a depth model, an ASR encoder and a segmentation head all arrive the
same way: a repository of Python, some weights, a licence, and a list of things
the author wanted installed. Pure standard library, Python-3.8-compatible
syntax, no torch and no network beyond one ``git clone``.
"""
from __future__ import annotations

import ast
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_LOG = logging.getLogger(__name__)

#: Environment variable that overrides the default workspace root.
ENV_WORKSPACE = "TRT_WORKSPACE"

#: Where workspaces go when nothing overrides it. Outside the repo, on purpose.
DEFAULT_WORKSPACE_ROOT = "~/trt"

#: Subdirectories every workspace gets. ``code`` holds the clone (or stays
#: empty when the source is a local path); ``notes`` holds the working notes
#: the operator writes as they read the model; ``artifacts`` holds the plan,
#: the baseline and anything else the later stages produce.
CODE_DIR = "code"
NOTES_DIR = "notes"
ARTIFACTS_DIR = "artifacts"
WORKSPACE_DIRS = (CODE_DIR, NOTES_DIR, ARTIFACTS_DIR)

#: Longest slug this will produce. Short enough to type, long enough to stay
#: recognisable next to a dozen sibling workspaces.
MAX_SLUG_CHARS = 48

#: Timeouts for the two git calls this module makes. A shallow clone of a repo
#: with a large history or a slow mirror is legitimately slow; a ``rev-parse``
#: never is.
GIT_CLONE_TIMEOUT_S = 900
GIT_QUERY_TIMEOUT_S = 30

#: Prefixes that mean "a model hub id, not a git repository".
#: Hosts whose URLs name a model hub rather than a git remote. Their weights are
#: deliberately not fetched here -- downloading is the user's decision.
HUB_HOSTS = ("huggingface.co", "hf.co")

#: Path segments a web UI adds that are not part of the repository path, so a
#: pasted browser URL still slugs to the repository's own name.
WEB_VIEW_SEGMENTS = ("tree", "blob", "releases", "commit", "commits", "raw")

#: A URL with an explicit scheme.
_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")

#: A scp-style git remote (``git@host:org/repo``) or a bare ``host/org/repo``.
_BARE_HOST_RE = re.compile(r"^(?:[\w.-]+@)?[\w.-]+\.[a-zA-Z]{2,}[:/]")

#: URL schemes that name a model hub directly.
HUB_PREFIXES = ("hf://", "hub:", "huggingface:")


# --------------------------------------------------------------------------
# the source
# --------------------------------------------------------------------------

@dataclass
class Source:
    """Where the model came from and where its workspace is.

    Args:
        kind: ``"git"`` (cloned from a URL), ``"local"`` (a path on this
            machine, referenced in place) or ``"hub"`` (a model-hub id, whose
            weights this module deliberately does not fetch).
        url_or_path: the clone URL, the absolute local path, or the hub id --
            verbatim enough that re-running :func:`acquire` on it reproduces
            this Source.
        slug: the filesystem-safe short name; the workspace directory name and
            the stem of everything the later stages write.
        workspace: the workspace root, always outside this repository.
        commit: resolved commit SHA, or None when the source is not a git
            checkout or nothing was cloned (a dry run).
        license_name: identified licence, or None when there is no licence file
            or it could not be identified. :func:`license_note` carries the part
            that decides deployment.
    """

    kind: str
    url_or_path: str
    slug: str
    workspace: Path
    commit: Optional[str] = None
    license_name: Optional[str] = None

    @property
    def code_dir(self):
        """Where the code actually is: the clone, or the local path itself.

        A local source is referenced in place and never copied, so for it this
        is the path the operator gave, not something under ``workspace``.
        """
        if self.kind == "local":
            return Path(self.url_or_path)
        return self.workspace / CODE_DIR


def _looks_like_url(text):
    """True when ``text`` is a URL rather than a filesystem path."""
    return bool(_URL_RE.match(text) or _BARE_HOST_RE.match(text))


def slugify(name):
    """A filesystem-safe short slug from a repo URL or path.

    Takes the last meaningful component, drops a ``.git`` suffix and any
    trailing slash, and reduces everything else to lowercase ``a-z0-9-``. For a
    URL it also drops the browser's view segments, so a link copied out of the
    address bar does not produce a workspace called ``main``.

    Args:
        name: a clone URL, a hub id, or a filesystem path.

    Returns:
        The slug, at most :data:`MAX_SLUG_CHARS` characters.

    Raises:
        ValueError: when nothing usable survives -- an empty string, a bare
            host, or a name made entirely of punctuation. A silent fallback
            here would put two unrelated models in one workspace.
    """
    text = str(name).strip()
    if not text:
        raise ValueError("cannot derive a slug from an empty source")
    for prefix in HUB_PREFIXES:
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
            break
    is_url = _looks_like_url(text)
    text = re.sub(r"[?#].*$", "", text)
    text = re.split(r"://", text, maxsplit=1)[-1]
    text = re.sub(r"^[^@/]+@[^:/]+:", "", text)
    parts = [p for p in re.split(r"[/\\]", text) if p not in ("", ".", "..")]
    if is_url:
        parts = _drop_web_view(parts)
    if not parts:
        raise ValueError(
            "cannot derive a slug from %r: no name component in it. Pass the "
            "repository URL or the directory holding the model." % name)
    tail = parts[-1]
    if tail.lower().endswith(".git"):
        tail = tail[:-4]
    slug = re.sub(r"[^a-z0-9]+", "-", tail.lower()).strip("-")
    if not slug:
        raise ValueError(
            "cannot derive a slug from %r: %r reduces to nothing a directory "
            "can be named after." % (name, tail))
    return slug[:MAX_SLUG_CHARS].strip("-")


def _drop_web_view(parts):
    """Trim a URL's browser view segments (``owner/repo/tree/main`` -> two)."""
    for index, part in enumerate(parts):
        if index >= 1 and part.lower() in WEB_VIEW_SEGMENTS:
            return parts[:index]
    return parts


def repo_root():
    """The root of this checkout -- the one directory a workspace must avoid."""
    return Path(__file__).resolve().parents[4]


def _within(path, parent):
    """True when ``path`` is ``parent`` or lives under it."""
    try:
        Path(path).relative_to(Path(parent))
        return True
    except ValueError:
        return False


def _workspace_path(slug, root=None):
    """Resolve the workspace path and refuse one inside this repository."""
    chosen = root if root is not None else os.environ.get(ENV_WORKSPACE)
    chosen = chosen or DEFAULT_WORKSPACE_ROOT
    expanded = os.path.expanduser(os.path.expandvars(str(chosen)))
    workspace = (Path(expanded) / slug).resolve()
    checkout = repo_root()
    if _within(workspace, checkout):
        raise ValueError(
            "refusing the workspace %s: it is inside this repository (%s). A "
            "cloned repo or a multi-GB checkpoint in the source tree ends up "
            "in somebody's commit. Point %s somewhere else, or leave it unset "
            "to use %s."
            % (workspace, checkout, ENV_WORKSPACE, DEFAULT_WORKSPACE_ROOT))
    return workspace


def workspace_for(slug, root=None):
    """The workspace directory for ``slug``, created with its subdirectories.

    Args:
        slug: the slug from :func:`slugify`.
        root: workspace root; ``$TRT_WORKSPACE`` when unset, and
            :data:`DEFAULT_WORKSPACE_ROOT` when that is unset too.

    Returns:
        The created workspace :class:`~pathlib.Path`, holding
        :data:`WORKSPACE_DIRS`.

    Raises:
        ValueError: if the resulting path is inside this repository.
        OSError: if the directories cannot be created -- reported rather than
            swallowed, because every later stage writes here.
    """
    workspace = _workspace_path(slug, root=root)
    for name in WORKSPACE_DIRS:
        (workspace / name).mkdir(parents=True, exist_ok=True)
    return workspace


# --------------------------------------------------------------------------
# git
# --------------------------------------------------------------------------

def _git(args, cwd=None, timeout=GIT_QUERY_TIMEOUT_S):
    """Run one git command and return its stdout.

    Raises:
        RuntimeError: if git is missing, timed out, or exited non-zero. All
            three are reported with what to do about them; a clone that half
            worked is worse than one that refused.
    """
    command = ["git"] + [str(a) for a in args]
    try:
        proc = subprocess.run(command, cwd=(str(cwd) if cwd else None),
                              capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise RuntimeError(
            "git is not on PATH, so %r cannot run. Install git, or pass a "
            "local path that is already on this machine."
            % " ".join(command))
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "%r timed out after %d s. A shallow clone of a very large "
            "repository can exceed this; clone it by hand and pass the local "
            "path instead." % (" ".join(command), timeout))
    if proc.returncode != 0:
        raise RuntimeError(
            "%r failed (exit %d): %s"
            % (" ".join(command), proc.returncode,
               (proc.stderr or proc.stdout or "").strip() or "no output"))
    return proc.stdout.strip()


def _clone_url(text):
    """The URL git should be handed, from what the operator typed."""
    if _URL_RE.match(text):
        return text
    return "https://" + text.lstrip("/")


def _repo_identity(url):
    """``host/owner/name``, so two spellings of one repo compare equal."""
    text = re.sub(r"[?#].*$", "", str(url).strip()).rstrip("/")
    text = re.split(r"://", text, maxsplit=1)[-1]
    text = re.sub(r"^[^@/]+@", "", text).replace(":", "/", 1)
    if text.lower().endswith(".git"):
        text = text[:-4]
    return re.sub(r"/+", "/", text).lower()


def _git_commit(code_dir):
    """The resolved HEAD of a git checkout, or None when it is not one."""
    if not (Path(code_dir) / ".git").exists():
        return None
    return _git(["rev-parse", "HEAD"], cwd=code_dir) or None


def _clone(url, code_dir, depth):
    """Shallow-clone ``url`` into ``code_dir``, or reuse a matching clone.

    Raises:
        RuntimeError: when ``code_dir`` already holds something else. Two
            different repositories can slugify to the same name, and silently
            cloning over the first one -- or silently reusing it -- is how an
            analysis ends up describing the wrong model.
    """
    code_dir = Path(code_dir)
    if code_dir.exists() and any(code_dir.iterdir()):
        if not (code_dir / ".git").exists():
            raise RuntimeError(
                "%s already exists and is not a git checkout; refusing to "
                "clone into it. Remove it, or pass a different workspace root."
                % code_dir)
        existing = _git(["remote", "get-url", "origin"], cwd=code_dir)
        if _repo_identity(existing) != _repo_identity(url):
            raise RuntimeError(
                "%s already holds a clone of %s, which is not %s. Two repos "
                "slugified to the same workspace name; pass a different "
                "workspace root for one of them." % (code_dir, existing, url))
        _LOG.info("reusing the existing clone at %s", code_dir)
        return code_dir
    code_dir.parent.mkdir(parents=True, exist_ok=True)
    args = ["clone", "--depth", str(int(depth))] if depth else ["clone"]
    _git(args + [url, str(code_dir)], timeout=GIT_CLONE_TIMEOUT_S)
    return code_dir


# --------------------------------------------------------------------------
# acquisition
# --------------------------------------------------------------------------

def classify(source):
    """Decide whether ``source`` is a git URL, a local path or a hub id.

    Raises:
        ValueError: on a string that is neither a URL nor an existing path.
            Guessing here is what turns a typo into an empty workspace and a
            confusing failure three stages later.
    """
    text = str(source).strip()
    if not text:
        raise ValueError("no model source given")
    lowered = text.lower()
    if lowered.startswith(HUB_PREFIXES) or any(
            host in _repo_identity(text).split("/")[0] for host in HUB_HOSTS):
        return "hub"
    if _looks_like_url(text) or lowered.endswith(".git"):
        return "git"
    candidate = Path(os.path.expanduser(text))
    if candidate.exists():
        return "local"
    raise ValueError(
        "%r is neither an existing path nor a repository URL. Accepted: a git "
        "URL (https://host/owner/repo[.git], git@host:owner/repo.git), a "
        "directory on this machine, or a hub id prefixed with 'hf://'." % text)


def acquire(source, workspace_root=None, depth=1, dry_run=False):
    """Resolve a GitHub URL or a local path into a :class:`Source`.

    For a URL this shallow-clones into ``<workspace>/code`` with a plain
    ``git`` subprocess -- no gitpython, because a git binary is already a hard
    requirement of this repo and a second implementation of git is not. For a
    local path it records the path and copies nothing: the operator's checkout
    stays the one on disk, so their edits are the ones analysed.

    It never runs ``pip install`` and never downloads weights. Both are the
    user's decision and both are frequently the wrong move on a machine with
    four interpreters -- see :func:`requirements_report` for the difference
    between what the repo asks for and what is actually here.

    Args:
        source: a git URL, a local path, or a hub id.
        workspace_root: workspace root; ``$TRT_WORKSPACE`` or ``~/trt`` when
            omitted.
        depth: ``git clone --depth``. 1 is right unless the analysis needs
            history; pass 0 for a full clone.
        dry_run: report what would happen and touch nothing. No network, no
            subprocess, and no directory is created.

    Returns:
        A :class:`Source`. When ``dry_run`` is set, one describing the intent:
        the workspace path exists only as a path, and ``commit`` is None.

    Raises:
        ValueError: on a source that is neither a path nor a URL, or on a
            workspace inside this repository.
        RuntimeError: if git is missing, the clone fails or times out, or the
            code directory already holds a different repository.
    """
    text = str(source).strip()
    kind = classify(text)
    slug = slugify(text)
    if dry_run:
        workspace = _workspace_path(slug, root=workspace_root)
    else:
        workspace = workspace_for(slug, root=workspace_root)

    if kind == "local":
        recorded = str(Path(os.path.expanduser(text)).resolve())
    elif kind == "git":
        recorded = _clone_url(text)
    else:
        recorded = text

    resolved = Source(kind=kind, url_or_path=recorded, slug=slug,
                      workspace=workspace)
    if dry_run:
        return resolved

    if kind == "git":
        _clone(recorded, resolved.code_dir, depth)
    if resolved.code_dir.exists():
        resolved.commit = _git_commit(resolved.code_dir)
        resolved.license_name = license_note(resolved.code_dir).name
    return resolved


# --------------------------------------------------------------------------
# inventory
# --------------------------------------------------------------------------


# The inventory half lives in its own module; re-exported so a caller that only
# wants "acquire then look at it" still imports one name.
from sparx_agency.tasks.common.trt_optimizer.inventory import (  # noqa: E402,F401
    CHECKPOINT_SUFFIXES, CONFIG_SUFFIXES, MAX_LICENSE_BYTES, MAX_SOURCE_BYTES,
    REQUIREMENT_NAMES, SKIP_DIRS, SUMMARY_LIMIT, LicenseNote, find_entrypoints,
    license_note, requirements_report, summarize,
)
