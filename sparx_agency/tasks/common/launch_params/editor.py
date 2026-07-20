"""The screen an operator changes parameters on.

Every parameter is shown against its own default, and every one of them can be
put back -- individually, or the whole screen at once. That is the point of the
widget: a config with three hundred knobs is only usable if "what have I changed,
and how do I undo it" is answerable at a glance, so a moved value is marked, the
marks can be filtered down to on their own, and reset is never more than a click.

The editor owns no parameters. It renders a :class:`~.spec.ParamSet` it is
given and writes straight back into it, so whoever built the set sees the edits
without asking for them.
"""
from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Callable

from .scroll import ScrollableFrame
from .spec import FLAG, FLAG_CHOICES, ParamSet, ParamSpec

#: Marks a value that differs from its default.
CHANGED_COLOR = "#0b5394"
DEFAULT_COLOR = "#444444"
#: Longest ``doc`` shown inline; the rest goes to the detail pane below.
DOC_WIDTH = 74


@dataclass
class _Row:
    """One parameter's widgets, kept so filtering can hide and show them."""

    param: ParamSpec
    variable: tk.StringVar
    widgets: tuple[tk.Widget, ...]
    haystack: str
    grid_row: int


class ParameterEditor(ttk.Frame):
    """A searchable, sectioned, resettable form over a :class:`ParamSet`.

    Args:
        master: Parent widget.
        on_change: Called with no arguments after any edit or reset, so the
            owner can re-render the command preview.
    """

    def __init__(self, master: tk.Misc, on_change: Callable[[], None] | None = None) -> None:
        super().__init__(master)
        self._on_change = on_change
        self._params: ParamSet = ParamSet()
        self._rows: list[_Row] = []
        self._sections: dict[str, ttk.LabelFrame] = {}
        self._suspend = False

        self._search = tk.StringVar()
        self._changed_only = tk.BooleanVar(value=False)
        self._summary = tk.StringVar(value="No command selected.")
        self._detail = tk.StringVar(value="")
        self._build()

    # ── construction ──────────────────────────────────────────────

    def _build(self) -> None:
        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(0, 4))
        ttk.Label(bar, text="Filter:").pack(side="left")
        entry = ttk.Entry(bar, textvariable=self._search, width=26)
        entry.pack(side="left", padx=(4, 10))
        self._search.trace_add("write", lambda *_: self._apply_filter())
        ttk.Checkbutton(bar, text="Changed only", variable=self._changed_only,
                        command=self._apply_filter).pack(side="left")
        ttk.Button(bar, text="Reset all to defaults",
                   command=self.reset_all).pack(side="right", padx=2)

        ttk.Label(self, textvariable=self._summary,
                  foreground=CHANGED_COLOR).pack(anchor="w", pady=(0, 4))

        self._scroller = ScrollableFrame(self)
        self._scroller.pack(fill="both", expand=True)

        detail = ttk.LabelFrame(self, text="About this parameter")
        detail.pack(fill="x", pady=(6, 0))
        ttk.Label(detail, textvariable=self._detail, wraplength=760,
                  justify="left", foreground=DEFAULT_COLOR).pack(
                      anchor="w", padx=6, pady=4)

    # ── population ────────────────────────────────────────────────

    def show(self, params: ParamSet, problems: tuple[str, ...] = ()) -> None:
        """Render ``params``, replacing whatever was on screen.

        Args:
            params: The set to edit. Held by reference and written to in place.
            problems: Discovery failures to surface above the form, so a source
                that could not be read is visible rather than merely absent.
        """
        self._params = params
        self.clear()

        if problems:
            box = ttk.LabelFrame(self._scroller.body,
                                 text="Could not read some parameter sources")
            box.pack(fill="x", padx=2, pady=(2, 6))
            for line in problems:
                ttk.Label(box, text="• " + line, foreground="#a61b1b",
                          wraplength=740, justify="left").pack(anchor="w", padx=6)

        for section in params.sections():
            members = [p for p in params if p.section == section]
            if not members:
                continue
            frame = ttk.LabelFrame(self._scroller.body, text=section or "General")
            frame.pack(fill="x", padx=2, pady=3)
            frame.columnconfigure(3, weight=1)
            self._sections[section] = frame
            for index, param in enumerate(members):
                self._rows.append(self._build_row(frame, param, index))

        self._scroller.to_top()
        self._refresh_summary()

    def clear(self) -> None:
        """Empty the form."""
        for child in self._scroller.body.winfo_children():
            child.destroy()
        self._rows.clear()
        self._sections.clear()
        self._detail.set("")

    def _build_row(self, parent: ttk.LabelFrame, param: ParamSpec, index: int) -> _Row:
        variable = tk.StringVar(value=param.value)
        name = ttk.Label(parent, text=param.name, width=28, anchor="w")

        choices = param.choices or (FLAG_CHOICES if param.syntax == FLAG else ())
        if choices:
            field: tk.Widget = ttk.Combobox(parent, textvariable=variable, width=22,
                                            values=list(choices), state="readonly")
        else:
            field = ttk.Entry(parent, textvariable=variable, width=24)

        revert = ttk.Button(parent, text="↺", width=3,
                            command=lambda p=param: self.reset_one(p))
        doc = param.doc if len(param.doc) <= DOC_WIDTH else param.doc[:DOC_WIDTH - 1] + "…"
        note = ttk.Label(parent, text=doc, foreground=DEFAULT_COLOR,
                         anchor="w", justify="left")

        name.grid(row=index, column=0, sticky="w", padx=(6, 4), pady=1)
        field.grid(row=index, column=1, sticky="w", pady=1)
        revert.grid(row=index, column=2, padx=4)
        note.grid(row=index, column=3, sticky="we", padx=(6, 6))

        row = _Row(param=param, variable=variable,
                   widgets=(name, field, revert, note),
                   haystack=" ".join((param.name, param.doc, param.detail,
                                      param.section)).lower(),
                   grid_row=index)
        variable.trace_add("write", lambda *_, r=row: self._on_edit(r))
        for widget in (field, name, note):
            widget.bind("<Enter>", lambda _e, p=param: self._show_detail(p))
        field.bind("<FocusIn>", lambda _e, p=param: self._show_detail(p))
        self._paint(row)
        return row

    # ── editing ───────────────────────────────────────────────────

    def _on_edit(self, row: _Row) -> None:
        if self._suspend:
            return
        row.param.value = row.variable.get()
        self._paint(row)
        self._refresh_summary()
        if self._on_change:
            self._on_change()

    def _paint(self, row: _Row) -> None:
        """Mark the row according to whether it still sits at its default."""
        changed = row.param.changed
        name, _field, _revert, note = row.widgets
        name.configure(foreground=CHANGED_COLOR if changed else "",
                       font=("TkDefaultFont", 9, "bold" if changed else "normal"),
                       text=("● " if changed else "") + row.param.name)
        note.configure(foreground=CHANGED_COLOR if changed else DEFAULT_COLOR)

    def _show_detail(self, param: ParamSpec) -> None:
        parts = ["%s  (default: %s" % (param.name, param.default or "''")]
        parts[0] += ", from %s)" % param.source if param.source else ")"
        if param.doc:
            parts.append(param.doc)
        if param.detail and param.detail != param.doc:
            parts.append(param.detail)
        self._detail.set("\n".join(parts))

    def reset_one(self, param: ParamSpec) -> None:
        """Put one parameter back to its default."""
        param.reset()
        self._sync()

    def reset_all(self) -> None:
        """Put every parameter on screen back to its default."""
        self._params.reset()
        self._sync()

    def _sync(self) -> None:
        """Push model values into the widgets after a programmatic change."""
        self._suspend = True
        try:
            for row in self._rows:
                if row.variable.get() != row.param.value:
                    row.variable.set(row.param.value)
                self._paint(row)
        finally:
            self._suspend = False
        self._refresh_summary()
        self._apply_filter()
        if self._on_change:
            self._on_change()

    def refresh(self) -> None:
        """Re-read the model into the widgets (after loading saved values)."""
        self._sync()

    # ── filtering ─────────────────────────────────────────────────

    def _apply_filter(self) -> None:
        needle = self._search.get().strip().lower()
        changed_only = self._changed_only.get()
        visible_per_section: dict[ttk.LabelFrame, int] = {}

        for row in self._rows:
            keep = (not needle or needle in row.haystack) and (
                not changed_only or row.param.changed)
            parent = row.widgets[0].master
            visible_per_section[parent] = visible_per_section.get(parent, 0) + int(keep)
            for column, widget in enumerate(row.widgets):
                if keep:
                    widget.grid(row=row.grid_row, column=column)
                else:
                    widget.grid_remove()

        for frame in self._sections.values():
            if visible_per_section.get(frame, 0):
                frame.pack(fill="x", padx=2, pady=3)
            else:
                frame.pack_forget()

    def _refresh_summary(self) -> None:
        changed = self._params.changed()
        if not len(self._params):
            self._summary.set("No editable parameters were found for this command.")
        elif changed:
            self._summary.set("%d of %d parameters changed: %s" % (
                len(changed), len(self._params),
                ", ".join(p.name for p in changed[:8])
                + (" …" if len(changed) > 8 else "")))
        else:
            self._summary.set("%d parameters, all at their defaults."
                              % len(self._params))
