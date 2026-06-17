#!/usr/bin/env python3
"""
Interactive frame labeler for offline room mapping.

Cycles through rgb_1/ frames, lets you draw bounding boxes with the mouse,
prompts for a label in the terminal, and saves labels.json.

Usage:
    python label_frames.py \
        --data-dir /home/daphnaa/Documents/xtend_da3_takes/xtend_da3_take_20260616_171539 \
        --output   labels.json \
        --stride   10

Controls (in the image window):
    Draw box   : click-drag with left mouse button
    Skip frame : press 'c' inside selectROI, then type empty label
    Quit       : press 'q' in the review window (saves what you have)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # project root

from sparx_agency.demos.Demo_No4_XTEND_MapRoom.room_mapper.frame_reader import iter_frames


def _draw_existing(img: np.ndarray, entries: list, frame_idx: int) -> np.ndarray:
    """Overlay already-saved labels for this frame onto img."""
    out = img.copy()
    for e in entries:
        if e["frame_idx"] != frame_idx:
            continue
        bx, by, bw, bh = e["bbox"]
        cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.putText(out, e["label"], (bx, by - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def _load_existing(output_path: Path) -> List[dict]:
    if output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
        print(f"[labeler] Resuming — {len(data)} existing labels loaded from {output_path}")
        return data
    return []


def _save(entries: list, output_path: Path) -> None:
    with open(output_path, "w") as f:
        json.dump(entries, f, indent=2)


def main() -> None:
    p = argparse.ArgumentParser(description="Interactive frame labeler.")
    p.add_argument("--data-dir", required=True, help="Recording dir with rgb_1/")
    p.add_argument("--output",   default="labels.json", help="Output labels JSON path")
    p.add_argument("--stride",   type=int, default=10,  help="Sample every Nth frame")
    args = p.parse_args()

    output_path = Path(args.output)
    entries = _load_existing(output_path)
    labeled_frames = {e["frame_idx"] for e in entries}

    frames = list(iter_frames(Path(args.data_dir), stride=args.stride))
    print(f"[labeler] {len(frames)} frames (stride={args.stride}). "
          f"Already labeled: {len(labeled_frames)}. "
          f"Press 'q' in the review window to quit and save.")

    win = "label_frames"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1440, 840)   # 2× the 720×420 frame — easier to draw precise boxes

    for rec in frames:
        if rec.frame_idx in labeled_frames:
            continue

        bgr = rec.load_rgb()
        h, w = bgr.shape[:2]
        display = _draw_existing(bgr, entries, rec.frame_idx)

        info = f"Frame {rec.frame_idx:06d}  ({len(entries)} labels so far)"
        cv2.putText(display, info, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

        # Inner loop: keep drawing boxes on the same frame until empty ROI
        quit_all = False
        while True:
            current = _draw_existing(bgr, entries, rec.frame_idx)
            cv2.putText(current, info, (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
            cv2.putText(current, "draw box + Enter to label | empty ROI (c) = next frame",
                        (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)
            cv2.imshow(win, current)
            cv2.waitKey(1)

            roi = cv2.selectROI(win, current, showCrosshair=True, fromCenter=False)
            rx, ry, rw, rh = (int(v) for v in roi)

            if rw < 5 or rh < 5:
                # empty ROI → done with this frame
                break

            # Draw confirmed box while user types
            preview = current.copy()
            cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), (0, 100, 255), 2)
            cv2.imshow(win, preview)
            cv2.waitKey(1)

            label = input(f"  Label for frame {rec.frame_idx:06d} (Enter to skip): ").strip()
            if not label:
                continue

            entry = {
                "frame_idx":   rec.frame_idx,
                "bbox":        [rx, ry, rw, rh],
                "source_size": [w, h],
                "label":       label,
            }
            entries.append(entry)
            labeled_frames.add(rec.frame_idx)
            _save(entries, output_path)
            print(f"  [saved] '{label}'  bbox=({rx},{ry},{rw},{rh})")

            # Brief flash of saved state, check for quit
            saved_view = _draw_existing(bgr, entries, rec.frame_idx)
            cv2.imshow(win, saved_view)
            key = cv2.waitKey(300) & 0xFF
            if key == ord('q'):
                quit_all = True
                break

        if quit_all:
            break

    cv2.destroyAllWindows()
    _save(entries, output_path)
    print(f"[labeler] Done. {len(entries)} labels saved to {output_path}")


if __name__ == "__main__":
    main()