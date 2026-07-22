"""Remember the parameter values an operator settled on, between sessions.

Only the values that were *moved off their default* are written. Storing all of
them would freeze today's defaults into the file, so tomorrow's improvement to a
node default would be silently overridden by a copy of yesterday's -- the exact
failure this whole package exists to make visible.
"""
from __future__ import annotations

import json
from pathlib import Path

#: Where the overrides live, unless the caller names somewhere else.
DEFAULT_PATH = Path.home() / ".config" / "sparx_agency" / "launcher_params.json"


class ParamStore:
    """A JSON file of ``{command key: {parameter: value}}``."""

    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        self.path = Path(path)
        self._data: dict[str, dict[str, str]] = {}
        self.load()

    def load(self) -> None:
        """Read the file, treating an absent one as "nothing saved yet".

        Raises:
            ValueError: If the file exists but is not the expected JSON object.
                A corrupt store is worth stopping for: continuing would quietly
                discard every saved override on the next save.
        """
        if not self.path.exists():
            self._data = {}
            return
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("cannot read saved parameters from %s: %s"
                             % (self.path, error))
        if not isinstance(loaded, dict):
            raise ValueError("saved parameters in %s are not a JSON object" % self.path)
        self._data = {str(key): dict(value) for key, value in loaded.items()
                      if isinstance(value, dict)}

    def save(self) -> None:
        """Write the file, creating its directory if needed."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")

    def get(self, key: str) -> dict[str, str]:
        """The saved overrides for one command, or an empty mapping."""
        return dict(self._data.get(key, {}))

    def put(self, key: str, values: dict[str, str]) -> None:
        """Replace one command's overrides and write the file.

        An empty ``values`` drops the entry entirely, so "reset then save" is
        the way to forget a command's overrides.
        """
        if values:
            self._data[key] = dict(values)
        else:
            self._data.pop(key, None)
        self.save()

    def keys(self) -> list[str]:
        """Commands that currently have saved overrides."""
        return sorted(self._data)
