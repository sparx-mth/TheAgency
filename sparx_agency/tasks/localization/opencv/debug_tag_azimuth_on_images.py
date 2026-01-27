#!/usr/bin/env python3
import argparse
from pathlib import Path
import time

import cv2

from sparx_agency.tasks.localization.opencv.tag_azimuth_node import TagAzimuthOpenCVTask

"""
for run:
python3 -m sparx_agency.tasks.localization.opencv.debug_tag_azimuth_on_images --images /home/user/test_imgs --show
"""

def iter_images(path: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if path.is_file() and path.suffix.lower() in exts:
        yield path
    elif path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.suffix.lower() in exts:
                yield p


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

    for p in iter_images(img_path):
        frame_bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            print(f"[SKIP] Failed to read {p}")
            continue

        timestamp_sec = time.time()

        try:
            yaw_deg = task.compute_azimuth_from_bgr(frame_bgr, timestamp_sec)
            print(f"[OK] {p.name}  yaw_deg={yaw_deg:.2f}")
        except Exception as e:
            print(f"[FAIL] {p.name}  error={e}")

        if args.show:
            vis = frame_bgr.copy()
            cv2.putText(
                vis,
                f"{p.name}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("debug", vis)
            key = cv2.waitKey(0)
            if key == 27:  # ESC
                break

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
