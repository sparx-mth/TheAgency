"""Locate the InternNav source tree and import System-1 modules out of it.

InternVLA-N1's System-1 is defined in the upstream InternNav repository, not
here. This module does two things so the rest of the package can simply import
what it needs:

1. **Finds the checkout.** ``$INTERNNAV_HOME``, else ``~/trt/internnav/code``
   (where ``trt_optimizer acquire`` puts it). Missing is an actionable error,
   not an ``ImportError`` three frames deep.
2. **Registers namespace stubs.** ``internnav/model/encoder/__init__.py``
   eagerly imports ``transformers`` for the BERT backbone of unrelated VLN
   baselines. System 1 needs neither transformers nor an installed InternNav, so
   the stubs give the relative imports a ``__path__`` to resolve against while
   the poisoning ``__init__`` files never execute.

Same shape as the NavDP and FlowNav builders, which also require an upstream
fork present at build time and never at run time: the built engine and the numpy
runtime under ``core/`` depend on none of this.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

#: Sub-packages stubbed so relative imports resolve without running __init__.
_STUB_PACKAGES = (
    "internnav",
    "internnav.model",
    "internnav.model.encoder",
    "internnav.model.encoder.depth_anything",
    "internnav.model.encoder.depth_anything.depth_anything_v2",
    "internnav.model.basemodel",
    "internnav.model.basemodel.internvla_n1",
)

#: Where ``trt_optimizer acquire`` clones the repository.
DEFAULT_CHECKOUT = Path.home() / "trt" / "internnav" / "code"


def find_checkout(path=None):
    """Return the InternNav source root.

    Args:
        path: explicit path; overrides everything when given.

    Returns:
        :class:`pathlib.Path` of the repository root.

    Raises:
        FileNotFoundError: naming the three places that were tried, when no
            checkout is present. The build cannot proceed without the upstream
            model definition and saying so early beats an ImportError later.
    """
    candidates = []
    if path:
        candidates.append(Path(path))
    env = os.environ.get("INTERNNAV_HOME")
    if env:
        candidates.append(Path(env))
    candidates.append(DEFAULT_CHECKOUT)
    for candidate in candidates:
        if (candidate / "internnav" / "model").is_dir():
            return candidate
    raise FileNotFoundError(
        "InternNav checkout not found. Tried: %s. Clone it with "
        "`python -m sparx_agency.tasks.common.trt_optimizer acquire "
        "https://github.com/InternRobotics/InternNav`, or set $INTERNNAV_HOME."
        % ", ".join(str(c) for c in candidates))


def install(path=None):
    """Make InternNav's System-1 modules importable. Idempotent.

    Args:
        path: explicit checkout path, forwarded to :func:`find_checkout`.

    Returns:
        The checkout root that was installed.
    """
    root = find_checkout(path)
    for name in _STUB_PACKAGES:
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__path__ = [str(root / Path(*name.split(".")))]
        sys.modules[name] = module
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
