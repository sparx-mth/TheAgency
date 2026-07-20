"""A Tk frame you can put more rows in than the window is tall.

Tk has no scrollable container, only a scrollable Canvas, so the standard trick
is a Canvas holding one window-item that contains the real frame. The two
bindings below are what make it behave: the frame's size drives the scroll
region, and the canvas's width is forced onto the frame so rows fill the width
instead of hugging their content.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class ScrollableFrame(ttk.Frame):
    """A vertically scrolling container. Add children to :attr:`body`."""

    def __init__(self, master: tk.Misc, **kwargs) -> None:
        super().__init__(master, **kwargs)
        self._canvas = tk.Canvas(self, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        #: The frame to put content in.
        self.body = ttk.Frame(self._canvas)
        self._window = self._canvas.create_window((0, 0), window=self.body, anchor="nw")

        self.body.bind("<Configure>", self._on_body_resize)
        self._canvas.bind("<Configure>", self._on_canvas_resize)
        # Wheel events go to the widget under the pointer, which is usually a
        # row rather than the canvas, so bind on the whole subtree.
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.bind_all(sequence, self._on_wheel, add="+")

    def _on_body_resize(self, _event=None) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_resize(self, event) -> None:
        self._canvas.itemconfigure(self._window, width=event.width)

    def _on_wheel(self, event) -> None:
        """Scroll only while the pointer is actually over this canvas.

        The binding is application-wide (wheel events go to whichever widget is
        under the pointer, which is a row rather than the canvas), so the check
        is what stops this stealing the wheel from the rest of the window.
        """
        # Ask through self, not event.widget: Tk delivers a widget NAME rather
        # than an object for some sources, and that has no winfo_containing.
        pointed = self.winfo_containing(event.x_root, event.y_root)
        while pointed is not None:
            if pointed is self:
                break
            pointed = getattr(pointed, "master", None)
        else:
            return
        # X11 sends buttons 4/5; Windows and macOS send a signed delta.
        step = -1 if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0 else 1
        self._canvas.yview_scroll(step, "units")

    def to_top(self) -> None:
        """Scroll back to the first row."""
        self._canvas.yview_moveto(0.0)
