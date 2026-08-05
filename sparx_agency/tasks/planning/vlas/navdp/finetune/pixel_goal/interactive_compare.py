"""Interactive side-by-side: UNTRAINED vs TRAINED NavDP on a real recording.

Load a flight recording, click any pixel as the goal, and see both models answer
at once: the pretrained baseline and your fine-tuned checkpoint, drawn on the
camera image AND on the bird's-eye map, over the single-frame wall field. Step
through frames and recordings to probe behavior wherever you like.

Both models are held in memory as torch policies (the GPU has room now), so a
click runs both with no weight-swapping. Run in the ``navdp`` conda env:

    PYTHONPATH=<repo> NAVDP_REPO=~/PycharmProjects/NavDP/baselines/navdp \
      ~/miniconda3/envs/navdp/bin/python -m \
      sparx_agency.tasks.planning.vlas.navdp.finetune.pixel_goal.interactive_compare \
      --rec walk_into --trained ~/flight_dataset/run_long/ema_latest.pth
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button

from sparx_agency.tasks.planning.vlas.common.finetune.common.esdf_target import signed_sdf
from sparx_agency.tasks.planning.vlas.common.finetune.common.frames import LocalMapConfig, cloud_to_occupancy_grid, depth_to_body_cloud
from ..verify import bev_render, pipeline
from ..verify.pixel_goal import pixel_to_goal
from .navdp_torch import TorchNavDP

_BASE_CKPT = Path.home() / "Downloads/navdp-cross-modal.ckpt"
_REPO = Path.home() / "PycharmProjects/NavDP/baselines/navdp"
UNTRAINED_C = "#e8873a"   # orange
TRAINED_C = "#2eb6e6"     # blue


def _field(depth, intr, pitch, height):
    """Single-frame occupancy + signed ESDF for the BEV backdrop (walls)."""
    cfg = LocalMapConfig(camera_height_m=height, pitch_deg=pitch)
    occ = cloud_to_occupancy_grid(depth_to_body_cloud(depth, intr, cfg), cfg)
    return occ, signed_sdf(occ, 4.0)


def _project(traj, intr, cam_h):
    """Ground-plane FLU waypoints -> image pixels (u, v); None where behind camera."""
    out = []
    for fwd, left in traj:
        if fwd > 0.05:
            out.append((intr.fx * (-left) / fwd + intr.cx,
                        intr.fy * cam_h / fwd + intr.cy))
        else:
            out.append(None)
    return out


class CompareApp:
    """Click a pixel; the untrained and trained models both answer."""

    def __init__(self, dataset, rec, base_ckpt, repo, trained, device="cuda",
                 pitch=0.0, height=1.0):
        self.dataset = Path(dataset)
        self.recs = sorted(d.name for d in self.dataset.iterdir() if (d / "rgb").is_dir())
        self.rec_i = self.recs.index(rec) if rec in self.recs else 0
        self.pitch, self.height = pitch, height
        self.uv = self.traj_b = self.traj_t = self.goal = None

        print("loading UNTRAINED model ...", flush=True)
        self.m_base = TorchNavDP(str(base_ckpt), str(repo), device)
        print("loading TRAINED model ...", flush=True)
        self.m_trained = TorchNavDP(str(base_ckpt), str(repo), device)
        self.m_trained.load_weights(trained)
        print("ready - click a pixel", flush=True)

        self._load_rec()
        self.frame_i = 0
        self._build_ui()
        self._load_frame()

    # ---- data ----
    def _load_rec(self):
        self.rec_dir = self.dataset / self.recs[self.rec_i]
        self.intr = pipeline.load_intrinsics(self.rec_dir)
        self.frames = pipeline.list_frames(self.rec_dir)

    def _load_frame(self):
        self.bgr, self.depth = pipeline.load_frame(self.rec_dir, self.frames[self.frame_i])
        self.rgb = self.bgr[:, :, ::-1]
        self.occ, self.sdf = _field(self.depth, self.intr, self.pitch, self.height)
        if self.uv is not None:
            self._infer()
        self._redraw()

    # ---- inference ----
    def _infer(self):
        u, v = self.uv
        self.goal = pixel_to_goal(u, v, self.depth, self.intr)[:2]
        self.traj_b = self.m_base.predict(self.bgr, self.depth, self.goal)
        self.traj_t = self.m_trained.predict(self.bgr, self.depth, self.goal)

    # ---- drawing ----
    def _redraw(self):
        title = "%s  frame %d/%d" % (self.recs[self.rec_i], self.frame_i, len(self.frames) - 1)
        if self.goal is not None:
            title += "   goal: fwd=%.2f left=%.2f" % (self.goal[0], self.goal[1])
        bev_render.draw_image(self.ax_rgb, self.rgb, self.uv, title)
        bev_render.draw_depth(self.ax_depth, self.depth, self.uv)
        if self.traj_b is not None:
            self._overlay_image()
            self._draw_bev()
        else:
            self.ax_bev.clear()
            self.ax_bev.text(0.5, 0.5, "click a pixel on the colour or depth image",
                             ha="center", va="center", transform=self.ax_bev.transAxes)
            self.ax_bev.set_xticks([]); self.ax_bev.set_yticks([])
        self.fig.canvas.draw_idle()

    def _overlay_image(self):
        # Draw only the waypoints whose ground projection lands inside the image;
        # near points project far below the frame (drone camera looks ahead, not
        # down), so restore the image extents afterwards or they'd stretch the axis.
        h, w = self.depth.shape
        for traj, c in ((self.traj_b, UNTRAINED_C), (self.traj_t, TRAINED_C)):
            xs, ys = [], []
            for p in _project(traj, self.intr, self.height):
                inb = p is not None and 0 <= p[0] < w and 0 <= p[1] < h
                xs.append(p[0] if inb else np.nan)
                ys.append(p[1] if inb else np.nan)
            self.ax_rgb.plot(xs, ys, "-", color=c, lw=2.5, alpha=0.9)
        self.ax_rgb.set_xlim(0, w)
        self.ax_rgb.set_ylim(h, 0)

    def _draw_bev(self):
        ax = self.ax_bev
        ax.clear()
        h, w = self.occ.grid.shape
        res = self.occ.resolution
        ext = [self.occ.origin_x, self.occ.origin_x + w * res,
               self.occ.origin_y, self.occ.origin_y + h * res]
        ax.imshow(np.clip(self.sdf, -1, 2), extent=ext, origin="lower", cmap="RdYlBu",
                  vmin=-1, vmax=2, aspect="equal", zorder=0)
        ys, xs = np.where(self.occ.grid == self.occ.values.occupied)
        ax.scatter(xs * res + self.occ.origin_x, ys * res + self.occ.origin_y,
                   s=3, c="black", alpha=0.4, zorder=1, linewidths=0)
        ax.plot(self.traj_b[:, 0], self.traj_b[:, 1], "o-", color=UNTRAINED_C, ms=3, lw=2.5,
                label="untrained", zorder=3)
        ax.plot(self.traj_t[:, 0], self.traj_t[:, 1], "o-", color=TRAINED_C, ms=3, lw=2.5,
                label="trained", zorder=4)
        ax.plot(0, 0, "^", color="cyan", ms=13, zorder=5)
        ax.plot(self.goal[0], self.goal[1], "*", color="magenta", ms=20, zorder=6)
        fwd_max = max(2.5, self.traj_b[:, 0].max(), self.traj_t[:, 0].max(), self.goal[0]) + 0.7
        lat = max(1.5, np.abs(self.traj_b[:, 1]).max(), np.abs(self.traj_t[:, 1]).max(),
                  abs(self.goal[1])) + 0.7
        ax.set_xlim(ext[0], min(ext[1], fwd_max)); ax.set_ylim(max(ext[2], -lat), min(ext[3], lat))
        ax.set_xlabel("forward [m]"); ax.set_ylabel("left [m]")
        ax.set_title("bird's-eye: both models over the wall field", fontsize=9)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    # ---- ui ----
    def _build_ui(self):
        self.fig = plt.figure(figsize=(15, 9))
        self.fig.canvas.manager.set_window_title("Untrained vs Trained NavDP")
        gs = self.fig.add_gridspec(2, 2, left=0.05, right=0.97, top=0.95, bottom=0.13,
                                   hspace=0.22, wspace=0.16)
        self.ax_rgb = self.fig.add_subplot(gs[0, 0])
        self.ax_depth = self.fig.add_subplot(gs[0, 1])
        self.ax_bev = self.fig.add_subplot(gs[1, 0])
        self.ax_leg = self.fig.add_subplot(gs[1, 1]); self._legend_panel()

        def button(x, label, cb):
            b = Button(self.fig.add_axes([x, 0.04, 0.11, 0.045]), label)
            b.on_clicked(cb)
            return b
        self.b_pf = button(0.06, "◀ frame", lambda e: self._step_frame(-1))
        self.b_nf = button(0.19, "frame ▶", lambda e: self._step_frame(+1))
        self.b_pr = button(0.34, "◀ rec", lambda e: self._step_rec(-1))
        self.b_nr = button(0.47, "rec ▶", lambda e: self._step_rec(+1))
        self.b_sv = button(0.85, "save png", self._save)
        self.fig.text(0.06, 0.005, "click the colour or depth image to set a goal · "
                      "orange = untrained · blue = trained", fontsize=10, color="dimgray")
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _legend_panel(self):
        ax = self.ax_leg
        ax.axis("off")
        ax.text(0.02, 0.92, "How to read it", transform=ax.transAxes, fontsize=13, weight="bold")
        rows = [(UNTRAINED_C, "Untrained NavDP", "the pretrained baseline"),
                (TRAINED_C, "Trained NavDP", "your fine-tuned checkpoint"),
                ("magenta", "★ goal", "the pixel you clicked, deprojected"),
                ("black", "wall field", "warm = near a wall, cool = free")]
        y = 0.74
        for c, name, desc in rows:
            ax.plot([0.04, 0.12], [y, y], "-", color=c, lw=4, transform=ax.transAxes,
                    clip_on=False) if c not in ("magenta", "black") else \
                ax.text(0.04, y - 0.01, "★" if c == "magenta" else "■",
                        color=c, transform=ax.transAxes, fontsize=15)
            ax.text(0.16, y, name, transform=ax.transAxes, fontsize=12, weight="bold", va="center")
            ax.text(0.16, y - 0.055, desc, transform=ax.transAxes, fontsize=10.5,
                    color="gray", va="center")
            y -= 0.16

    # ---- events ----
    def _on_click(self, event):
        if event.inaxes not in (self.ax_rgb, self.ax_depth) or event.xdata is None:
            return
        h, w = self.depth.shape
        u = int(np.clip(round(event.xdata), 0, w - 1))
        v = int(np.clip(round(event.ydata), 0, h - 1))
        self.uv = (u, v)
        try:
            self._infer()
        except ValueError as exc:
            print("skip:", exc); return
        self._redraw()

    def _step_frame(self, d):
        self.frame_i = int(np.clip(self.frame_i + d, 0, len(self.frames) - 1))
        self._load_frame()

    def _step_rec(self, d):
        self.rec_i = int(np.clip(self.rec_i + d, 0, len(self.recs) - 1))
        self._load_rec(); self.frame_i = 0; self._load_frame()

    def _save(self, _e):
        out = self.dataset / "compare_shots" / ("%s_f%d.png" % (
            self.recs[self.rec_i], self.frames[self.frame_i]))
        out.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(out, dpi=110)
        print("saved", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path.home() / "data" / "flight" / "xtend_bags")
    ap.add_argument("--rec", default="walk_into")
    ap.add_argument("--trained", type=Path, required=True, help="fine-tuned ema_latest.pth")
    ap.add_argument("--ckpt", type=Path, default=_BASE_CKPT)
    ap.add_argument("--navdp-repo", type=Path, default=_REPO)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--height", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    CompareApp(args.dataset.expanduser(), args.rec, args.ckpt.expanduser(),
               args.navdp_repo.expanduser(), args.trained.expanduser(),
               args.device, args.pitch, args.height)
    plt.show()


if __name__ == "__main__":
    main()
