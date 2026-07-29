"""Interactive verifier for the NavDP -> instantaneous-PF/ESDF correction loop.

Click a pixel on the colour or depth image to set a body-frame goal; the tool runs
NavDP, builds the single-frame potential field / signed ESDF from that depth, pushes
the NavDP trajectory lightly off the walls, and shows -- side by side -- the field,
NavDP's trajectory, and the corrected target, so you can confirm the training signal
is right *before* fine-tuning. Sliders retune the correction live (no re-inference).

Run (in the ``navdp`` conda env)::

    PYTHONPATH=<repo> ~/miniconda3/envs/navdp/bin/python -m \
        sparx_agency.tasks.planning.vlas.navdp.finetune.verify.interactive_verify \
        --dataset ~/flight_dataset --rec walk_into --frame 40

Knobs: corrector (esdf | potential_field), target clearance + max shift ("how hard
to push"), camera pitch + height (fix the floor/obstacle split for the occupancy),
and field view (esdf | repulsion). See ``README.md``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider

from . import bev_render, pipeline
from .correction import make_config
from .navdp_infer import NavDPInfer, default_engine_paths


class VerifyApp:
    """Stateful matplotlib app wiring clicks + knobs to the per-pixel pipeline."""

    def __init__(self, dataset: Path, engine_dir: Path, head: Path,
                 rec: str = None, frame: int = None, n_sample: int = 25):
        self.dataset = Path(dataset)
        self.recs = sorted(d.name for d in self.dataset.iterdir() if (d / "rgb").is_dir())
        if not self.recs:
            raise FileNotFoundError(f"no recordings under {self.dataset}")
        self.rec_i = self.recs.index(rec) if rec in self.recs else 0
        self.infer = NavDPInfer(engine_dir, head)
        self.n_sample = int(n_sample)

        # live knob state
        self.corrector = "esdf"
        self.clearance = 0.5
        self.max_shift = 0.8
        self.pitch = 0.0
        self.height = 1.0
        self.smooth = 0.5
        self.field_mode = "esdf"

        # per-frame + per-click state
        self.uv = None
        self.navdp = None
        self.target = None
        self.goal = None
        self._cbar = None

        self._load_rec()
        self.frame_i = self.frames.index(frame) if frame in self.frames else 0
        self._build_ui()
        self._load_frame()

    # ---- data loading -------------------------------------------------
    def _load_rec(self) -> None:
        self.rec_dir = self.dataset / self.recs[self.rec_i]
        self.intr = pipeline.load_intrinsics(self.rec_dir)
        self.frames = pipeline.list_frames(self.rec_dir)

    def _load_frame(self) -> None:
        self.bgr, self.depth = pipeline.load_frame(self.rec_dir, self.frames[self.frame_i])
        self.rgb = self.bgr[:, :, ::-1]
        self.navdp = None
        self.target = None
        if self.uv is not None:
            self._recompute(rerun_navdp=True)
        self._redraw()

    # ---- pipeline -----------------------------------------------------
    def _cfg(self):
        return make_config(self.corrector, self.clearance, self.max_shift,
                           self.pitch, self.height, self.smooth)

    def _recompute(self, rerun_navdp: bool) -> None:
        u, v = self.uv
        from .pixel_goal import pixel_to_goal
        fwd, left, _ = pixel_to_goal(u, v, self.depth, self.intr)
        self.goal = (fwd, left)
        if rerun_navdp or self.navdp is None:
            self.navdp = self.infer.predict(self.bgr, self.depth, self.goal, seed=0)
        from .correction import correct_navdp_trajectory
        self.target = correct_navdp_trajectory(
            self.navdp.trajectory, self.depth, self.intr, self.goal, self._cfg())

    # ---- rendering ----------------------------------------------------
    def _redraw(self) -> None:
        title = "%s  frame %d/%d  (px %s)" % (
            self.recs[self.rec_i], self.frame_i, len(self.frames) - 1,
            self.uv if self.uv else "-")
        if self.goal is not None:
            title += "   goal fwd=%.2f left=%.2f   critic=%.2f" % (
                self.goal[0], self.goal[1], float(self.navdp.critic.max()))
        bev_render.draw_image(self.ax_rgb, self.rgb, self.uv, title)
        bev_render.draw_depth(self.ax_depth, self.depth, self.uv)
        self._redraw_bev_cmp()

    def _redraw_bev_cmp(self) -> None:
        if self.target is None:
            for ax, msg in ((self.ax_bev, "click a pixel"), (self.ax_cmp, "")):
                ax.clear()
                ax.text(0.5, 0.5, msg or "", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            im, label = bev_render.draw_bev(self.ax_bev, self.target, self.goal,
                                            self.field_mode, self.clearance)
            self.ax_bev.set_title("instantaneous field [%s]" % self.field_mode, fontsize=9)
            if self._cbar is None:
                self._cbar = self.fig.colorbar(im, cax=self.cax)
            else:
                self._cbar.update_normal(im)
            self._cbar.set_label(label, fontsize=8)
            bev_render.draw_comparison(self.ax_cmp, self.target, self.goal)
        self.fig.canvas.draw_idle()

    # ---- UI -----------------------------------------------------------
    def _build_ui(self) -> None:
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.canvas.manager.set_window_title("NavDP × PF/ESDF verifier")
        gs = self.fig.add_gridspec(2, 2, left=0.05, right=0.94, top=0.96,
                                   bottom=0.30, hspace=0.28, wspace=0.22)
        self.ax_rgb = self.fig.add_subplot(gs[0, 0])
        self.ax_depth = self.fig.add_subplot(gs[0, 1])
        self.ax_bev = self.fig.add_subplot(gs[1, 0])
        self.ax_cmp = self.fig.add_subplot(gs[1, 1])
        self.cax = self.fig.add_axes([0.455, 0.31, 0.007, 0.15])

        def slider(y, label, lo, hi, val):
            s = Slider(self.fig.add_axes([0.08, y, 0.30, 0.02]), label, lo, hi, valinit=val)
            return s
        self.s_clear = slider(0.22, "clearance m", 0.1, 1.0, self.clearance)
        self.s_shift = slider(0.18, "max shift m", 0.1, 2.0, self.max_shift)
        self.s_smooth = slider(0.14, "smooth", 0.0, 1.0, self.smooth)
        self.s_pitch = slider(0.10, "pitch deg", -30.0, 30.0, self.pitch)
        self.s_height = slider(0.06, "cam height m", 0.3, 2.0, self.height)
        for s in (self.s_clear, self.s_shift, self.s_smooth, self.s_pitch, self.s_height):
            s.on_changed(self._on_slider)

        self.r_corr = RadioButtons(self.fig.add_axes([0.44, 0.06, 0.11, 0.15]),
                                   ("esdf", "potential_field"))
        self.r_corr.on_clicked(self._on_corr)
        self.r_field = RadioButtons(self.fig.add_axes([0.57, 0.06, 0.10, 0.15]),
                                    ("esdf", "repulsion"))
        self.r_field.on_clicked(self._on_field)

        def button(x, y, w, label, cb):
            b = Button(self.fig.add_axes([x, y, w, 0.04]), label)
            b.on_clicked(cb)
            return b
        self.b_pf = button(0.70, 0.16, 0.06, "◀ frame", lambda e: self._step_frame(-1))
        self.b_nf = button(0.77, 0.16, 0.06, "frame ▶", lambda e: self._step_frame(+1))
        self.b_pr = button(0.70, 0.10, 0.06, "◀ rec", lambda e: self._step_rec(-1))
        self.b_nr = button(0.77, 0.10, 0.06, "rec ▶", lambda e: self._step_rec(+1))
        self.b_s = button(0.85, 0.16, 0.11, "sample %d" % self.n_sample, self._on_sample)
        self.b_save = button(0.85, 0.10, 0.11, "save png", self._on_save)

        self.fig.text(0.44, 0.02, "click the colour or depth image to set a goal · "
                      "sliders retune the correction live", fontsize=9, color="dimgray")
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    # ---- event handlers ----------------------------------------------
    def _on_click(self, event) -> None:
        if event.inaxes not in (self.ax_rgb, self.ax_depth) or event.xdata is None:
            return
        h, w = self.depth.shape
        u = int(np.clip(round(event.xdata), 0, w - 1))
        v = int(np.clip(round(event.ydata), 0, h - 1))
        self.uv = (u, v)
        try:
            self._recompute(rerun_navdp=True)
        except ValueError as exc:            # e.g. clicked a no-depth region
            print("skip:", exc)
            return
        self._redraw()

    def _on_slider(self, _val) -> None:
        self.clearance = self.s_clear.val
        self.max_shift = self.s_shift.val
        self.smooth = self.s_smooth.val
        self.pitch = self.s_pitch.val
        self.height = self.s_height.val
        if self.uv is not None:
            self._recompute(rerun_navdp=False)
            self._redraw_bev_cmp()

    def _on_corr(self, label) -> None:
        self.corrector = label
        if self.uv is not None:
            self._recompute(rerun_navdp=False)
            self._redraw_bev_cmp()

    def _on_field(self, label) -> None:
        self.field_mode = label
        self._redraw_bev_cmp()

    def _step_frame(self, d) -> None:
        self.frame_i = int(np.clip(self.frame_i + d, 0, len(self.frames) - 1))
        self._load_frame()

    def _step_rec(self, d) -> None:
        self.rec_i = int(np.clip(self.rec_i + d, 0, len(self.recs) - 1))
        self._load_rec()
        self.frame_i = 0
        self._load_frame()

    def _on_sample(self, _e) -> None:
        """Overlay many corrected trajectories to preview dataset diversity."""
        pix = pipeline.sample_valid_pixels(self.depth, self.n_sample, seed=0)
        first = None
        for u, v in pix:
            try:
                r = pipeline.run_pixel(self.bgr, self.depth, self.intr, u, v,
                                       self.infer, self._cfg(), seed=0)
            except ValueError:
                continue
            if first is None:
                first = r
                bev_render.draw_bev(self.ax_bev, r["target"], r["goal"], self.field_mode, self.clearance)
            s = bev_render.path_xy(r["target"].seed_path)
            c = bev_render.path_xy(r["target"].corrected_path)
            self.ax_bev.plot(s[:, 0], s[:, 1], "-", color="darkorange", lw=0.7, alpha=0.35)
            self.ax_bev.plot(c[:, 0], c[:, 1], "-", color="lime", lw=0.9, alpha=0.6)
        self.ax_bev.set_title("%d sampled goals: NavDP (orange) vs corrected (green)"
                              % len(pix), fontsize=9)
        self.fig.canvas.draw_idle()

    def _on_save(self, _e) -> None:
        out = self.dataset / "verify_shots" / ("%s_f%d.png" % (
            self.recs[self.rec_i], self.frames[self.frame_i]))
        out.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(out, dpi=110)
        print("saved", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path.home() / "data" / "flight" / "xtend_bags")
    ap.add_argument("--rec", default=None)
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--n-sample", type=int, default=25)
    args = ap.parse_args()
    engine_dir, head = default_engine_paths()
    VerifyApp(args.dataset.expanduser(), engine_dir, head,
              rec=args.rec, frame=args.frame, n_sample=args.n_sample)
    plt.show()


if __name__ == "__main__":
    main()
