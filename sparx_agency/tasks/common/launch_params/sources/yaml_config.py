"""Read a hand-commented YAML config as a documented parameter set.

``tasks/planning/falcon/config/mission.yaml`` is the object mission's single
source of defaults, and almost all of it is commentary: banner sections, a
paragraph of reasoning above each group, and a concise note after each value.
A YAML parser returns the values and discards every word of that, which is the
wrong half for an editor -- so this reads the file as text.

It also reads the file at the right *precedence*. A value set here is what a
plain run already uses, so it is the default the editor must show and the one
"reset" must return to; the built-in default further down the stack is only
reached for keys this file leaves out.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..spec import ENV, ROSLAUNCH, ParamSpec

#: ``# =====`` / ``# -----`` -- a rule line that opens or closes a banner.
_RULE_RE = re.compile(r"^[-=_~─═]{4,}$")
#: ``# -- The AIM: how it turns to look ------`` -- a one-line subsection.
_SUBSECTION_RE = re.compile(r"^-{2,}\s*(?P<title>.*?)\s*-{3,}$")
#: ``  key: value  # note`` -- one setting.
_KEY_RE = re.compile(r"^(?P<indent>\s*)(?P<key>[\w.]+):(?P<rest>\s.*|)$")


def _split_note(rest: str) -> tuple[str, str]:
    """Separate a value from its trailing ``#`` note, respecting quotes."""
    quote = None
    for index, char in enumerate(rest):
        if quote:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "#":
            return rest[:index].strip(), rest[index + 1:].strip()
    return rest.strip(), ""


def discover(path: str | Path,
             env_schema: dict[str, dict[str, str]] | None = None,
             only_groups: tuple[str, ...] = ()) -> list[ParamSpec]:
    """Read the settings out of a commented YAML config.

    Args:
        path: The YAML file.
        env_schema: ``{top-level group: {key: ENV_VAR}}``, for a config whose
            groups are not all launch arguments. A key found here becomes an
            environment-variable parameter under its mapped name; every other
            key becomes a roslaunch ``key:=value`` argument. Pass the reader's
            own schema so the two cannot drift.
        only_groups: Restrict the result to these top-level groups. A config
            can serve several commands that each use part of it -- the object
            mission's detector sidecar reads the ``detector`` group and none of
            the three hundred flight parameters -- and offering a command knobs
            it ignores is worse than offering none.

    Returns:
        One :class:`~..spec.ParamSpec` per assigned key, in file order, carrying
        the trailing note as its ``doc`` and the paragraph above it as its
        ``detail``. Commented-out keys are skipped -- they are suggestions, and
        their real defaults live in the launch file this config feeds.

    Raises:
        OSError: If the file cannot be read.
    """
    env_schema = env_schema or {}
    label = Path(path).name

    params: list[ParamSpec] = []
    section, group = "", ""
    prose: list[str] = []
    banner: list[str] | None = None

    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()

        if line.startswith("#"):
            body = line[1:].strip()
            if _RULE_RE.match(body):
                # A rule opens a banner; the next one closes it and the lines
                # collected between become the section title.
                if banner is None:
                    banner = []
                else:
                    section, banner, prose = " ".join(banner), None, []
                continue
            if banner is not None:
                banner.append(body)
                continue
            if subsection := _SUBSECTION_RE.match(body):
                section, prose = subsection.group("title"), []
                continue
            prose.append(body)
            continue

        if not line:
            prose = []
            continue

        match = _KEY_RE.match(raw)
        if match is None:
            prose = []
            continue

        value, note = _split_note(match.group("rest"))
        if not value:
            # A mapping key: it opens a group rather than setting anything.
            if not match.group("indent"):
                group = match.group("key")
            prose = []
            continue

        if only_groups and group not in only_groups:
            prose = []
            continue

        env_name = env_schema.get(group, {}).get(match.group("key"))
        explanation = " ".join(prose)
        params.append(ParamSpec(
            name=env_name or match.group("key"),
            default=value.strip("\"'"),
            # A key with no note of its own is explained by the paragraph above
            # it -- which is where the FALCON config puts the harder decisions.
            doc=note or explanation,
            detail=explanation,
            section=section,
            syntax=ENV if env_name else ROSLAUNCH,
            source=label,
        ))
        prose = []

    return params
