#!/usr/bin/env python3
import argparse
from pathlib import Path
import time

import cv2
import numpy as np

from sparx_agency.tasks.localization.opencv.tag_azimuth_node import TagAzimuthOpenCVTask

"""
Run:
python3 -m sparx_agency.tasks.localization.opencv.debug_tag_azimuth_on_images \
  --images /home/user/test_imgs --show
"""

def iter_images(path: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if path.is_file() and path.suffix.lower() in exts:
        yield path
    elif path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.suffix.lower() in exts:
                yield p


def get_detected_out_dir(images_path: Path) -> Path:
    base_dir = images_path.parent if images_path.is_file() else images_path
    out_dir = base_dir / "DETECTED"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def overlay_text_lines(img_bgr: np.ndarray, lines: list[str]) -> np.ndarray:
    vis = img_bgr.copy()
    y = 30
    for line in lines:
        cv2.putText(
            vis,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28
    return vis


def draw_tag_overlay(vis: np.ndarray, tag_id: int, corners_2d: np.ndarray) -> None:
    """
    corners_2d: (4,2) float
    """
    c = corners_2d.astype(int).reshape(4, 2)

    # polygon
    cv2.polylines(
        vis,
        [c.reshape(-1, 1, 2)],
        isClosed=True,
        color=(0, 255, 0),
        thickness=2,
    )

    # corner dots
    for (x, y) in c:
        cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)

    # label near first corner
    x0, y0 = int(c[0, 0]), int(c[0, 1])
    cv2.putText(
        vis,
        f"id={tag_id}",
        (x0, max(0, y0 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Image file or folder")
    ap.add_argument("--tag-config", default="sparx_agency/tasks/localization/config/tags_azimuth.yaml")
    ap.add_argument("--camera-calib", default="sparx_agency/tasks/localization/config/front_camera_calib.yaml")
    ap.add_argument("--tag-size", type=float, default=0.16)
    ap.add_argument("--show", action="store_true", help="Show debug window")
    args = ap.parse_args()

    task = TagAzimuthOpenCVTask(
        tag_config_path=args.tag_config,
        camera_calib_path=args.camera_calib,
        tag_size_m=args.tag_size,
    )

    img_path = Path(args.images)
    out_dir = get_detected_out_dir(img_path)

    for p in iter_images(img_path):
        frame_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            print(f"[SKIP] Failed to read {p}")
            continue

        timestamp_sec = time.time()

        # We'll draw on this
        vis = frame_bgr.copy()

        yaw_deg = None
        err = None

        # ---- 1) Compute yaw using the Task (your real pipeline) ----
        try:
            yaw_deg = task.compute_azimuth_from_bgr(frame_bgr, timestamp_sec)
            print(f"[OK] {p.name}  yaw_deg={yaw_deg:.2f}")
        except Exception as e:
            err = e
            print(f"[FAIL] {p.name}  error={e}")

        # ---- 2) Independently detect tags again just for drawing ----
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            dets = task.detector.detect(gray)

            for d in dets:
                tag_id = int(d.tag_id)
                corners = np.array(d.corners, dtype=np.float64).reshape(4, 2)

                # if you want: draw only tags that exist in config
                if tag_id not in task.tag_config_deg:
                    continue

                draw_tag_overlay(vis, tag_id=tag_id, corners_2d=corners)

        except Exception as e2:
            # Drawing should not break the run
            cv2.putText(
                vis,
                f"draw_error: {e2}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # ---- 3) Add top text (yaw / error) ----
        lines = [p.name]
        if yaw_deg is not None:
            lines.append(f"yaw_deg={yaw_deg:.2f}")
        if err is not None:
            lines.append(f"ERROR: {type(err).__name__}: {err}")

        vis = overlay_text_lines(vis, lines)

        # ---- 4) Save annotated image ----
        out_path = out_dir / p.name
        ok = cv2.imwrite(str(out_path), vis)
        if not ok:
            print(f"[WARN] Failed to write {out_path}")

        # ---- 5) Show if requested ----
        if args.show:
            cv2.imshow("debug", vis)
            key = cv2.waitKey(0)
            if key == 27:  # ESC
                break

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
