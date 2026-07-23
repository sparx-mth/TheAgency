#!/usr/bin/env python3
"""
OmniVLA Trajectory Visualizer

Adapted from OmniVLA's save_robot_behavior() in run_omnivla.py.
Draws:
  - Current camera image (top-left)
  - Goal image (bottom-left)
  - Bird's-eye trajectory plot (right) with all 8 waypoints
  - Goal pose as red star (if provided)
  - Modality label, velocity info

Returns a numpy BGR image for ROS2 publishing or file saving.
Uses matplotlib with Agg backend (no display needed).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional

# Modality names (same order as OmniVLA paper Table)
_MODALITY_NAMES = [
    "satellite only",        # 0
    "pose + satellite",      # 1
    "satellite + image",     # 2
    "all",                   # 3
    "pose only",             # 4
    "pose + image",          # 5
    "image only",            # 6
    "language only",         # 7
    "language + pose",       # 8
]


def draw_trajectory(
    current_image: Image.Image,
    waypoints: np.ndarray,
    linear_vel: float,
    angular_vel: float,
    modality: str,
    goal_image: Optional[Image.Image] = None,
    goal_pose: Optional[np.ndarray] = None,
    waypoint_index: int = 4,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Draw the OmniVLA trajectory visualization.

    Parameters
    ----------
    current_image  : PIL.Image — current camera view
    waypoints      : np.ndarray shape (8, 4) — predicted waypoints [dx, dy, hx, hy]
    linear_vel     : float — commanded linear velocity
    angular_vel    : float — commanded angular velocity
    modality       : str — active modality name
    goal_image     : PIL.Image | None — goal image (shown bottom-left if provided)
    goal_pose      : np.ndarray shape (4,) | None — if provided, drawn as red star
    waypoint_index : int — which waypoint is being tracked (highlighted)
    save_path      : str | None — if set, also saves to this file path

    Returns
    -------
    np.ndarray — BGR image (for cv2 / ROS2 publishing)
    """
    fig = plt.figure(figsize=(20, 10), dpi=80)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.5])

    ax_cam = fig.add_subplot(gs[0, 0])
    ax_goal = fig.add_subplot(gs[1, 0])
    ax_traj = fig.add_subplot(gs[:, 1])

    # ── Current image (top-left) ──────────────────────────────────────
    ax_cam.imshow(np.array(current_image).astype(np.uint8))
    ax_cam.set_title("Current camera", fontsize=14)
    ax_cam.axis("off")

    # ── Goal image (bottom-left) ──────────────────────────────────────
    if goal_image is not None:
        ax_goal.imshow(np.array(goal_image).astype(np.uint8))
        ax_goal.set_title("Goal image", fontsize=14)
    else:
        ax_goal.text(0.5, 0.5, "No goal image", ha="center", va="center",
                     fontsize=14, color="gray", transform=ax_goal.transAxes)
        ax_goal.set_title("Goal image", fontsize=14, color="gray")
    ax_goal.axis("off")

    # ── Trajectory plot (right) ───────────────────────────────────────
    # OmniVLA coordinate system: X = forward, Y = left
    # Plot: X-axis = left/right, Y-axis = forward
    x_seq = waypoints[:, 0]       # forward
    y_seq_inv = -waypoints[:, 1]  # left→right for plotting

    # Robot at origin + all waypoints
    plot_x = np.insert(y_seq_inv, 0, 0.0)
    plot_y = np.insert(x_seq, 0, 0.0)

    # Full trajectory line
    ax_traj.plot(plot_x, plot_y, linewidth=3, markersize=10, marker="o",
                 color="royalblue", alpha=0.7, label="Predicted trajectory")

    # Highlight robot position
    ax_traj.plot(0, 0, marker="^", color="green", markersize=16,
                 zorder=5, label="Robot")

    # Highlight tracked waypoint
    tracked_x = y_seq_inv[waypoint_index]
    tracked_y = x_seq[waypoint_index]
    ax_traj.plot(tracked_x, tracked_y, marker="D", color="orange",
                 markersize=14, zorder=5, markeredgecolor="black", markeredgewidth=1.5,
                 label=f"Tracked wp[{waypoint_index}]")

    # Goal pose as red star
    if goal_pose is not None:
        ax_traj.plot(-goal_pose[1], goal_pose[0], marker="*", color="red",
                     markersize=20, zorder=5, label="Goal pose")

    # Axis setup
    ax_traj.set_xlim(-3.0, 3.0)
    ax_traj.set_ylim(-0.5, 10.0)
    ax_traj.set_xlabel("← Right     Left →", fontsize=12)
    ax_traj.set_ylabel("Forward →", fontsize=12)
    ax_traj.set_title("OmniVLA Predicted Trajectory (robot frame)", fontsize=14)
    ax_traj.grid(True, alpha=0.3)
    ax_traj.set_aspect("equal")
    ax_traj.legend(loc="upper left", fontsize=10)

    # Info text
    info = f"Modality: {modality}\nv={linear_vel:.3f} m/s   ω={angular_vel:.3f} rad/s"
    ax_traj.text(0.98, 0.02, info, transform=ax_traj.transAxes,
                 fontsize=11, va="bottom", ha="right",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()

    # ── Render to numpy array ─────────────────────────────────────────
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    bgr = buf[:, :, ::-1].copy()  # RGB→BGR for OpenCV/ROS2

    # Save if requested
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.close(fig)
    return bgr