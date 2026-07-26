#!/usr/bin/env python3
"""webcam_frame_publisher.py -- mock the drone's RGB stream from a laptop webcam.

Reproduces *exactly what the drone publishes* for RGB, so the rest of the
object-approach stack cannot tell the difference between this and the real XTEND:

  * each captured frame is written to disk as a JPEG in a rolling folder
    (default ``/tmp/xtend_frames/frame_XXXXXXXX.jpg``, oldest deleted) -- the same
    "write the frame, don't serialize it over ROS" convention the drone uses;
  * optionally (``--ros``) a ``std_msgs/String`` frame-path message
    ``"<path> <sec> <nsec>"`` is published on ``/xtend/rgb_frame_path`` -- the exact
    format :mod:`sparx_agency.core.common.frame_path_message` parses.

The webcam is centre-cropped to the drone's aspect ratio and resized to the drone's
resolution (default 504x294, the XTEND) so framing/offsets match. Frames are
written atomically (temp file + rename) so a consumer never reads a half-written
JPEG.

There is no depth and no localization here -- the drone provides those, and the
target-lock stack degrades gracefully without them (area-fraction proximity proxy,
never scans without a pose). This publisher is RGB only, on purpose.

Run:
    # just write frames to /tmp/xtend_frames (what the offline tools read):
    python -m sparx_agency.tasks.planning.object_approach_webcam.webcam_frame_publisher

    # also publish the ROS2 frame-path topic (needs rclpy):
    python -m sparx_agency.tasks.planning.object_approach_webcam.webcam_frame_publisher --ros

Press Ctrl+C (or 'q' in the --show preview window) to stop.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np


def center_crop_resize(frame: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Centre-crop ``frame`` to the output aspect ratio, then resize to (out_w, out_h)."""
    h, w = frame.shape[:2]
    target_ar = out_w / float(out_h)
    src_ar = w / float(h)
    if src_ar > target_ar:                     # too wide -> crop width
        new_w = int(round(h * target_ar))
        x0 = (w - new_w) // 2
        frame = frame[:, x0:x0 + new_w]
    else:                                      # too tall -> crop height
        new_h = int(round(w / target_ar))
        y0 = (h - new_h) // 2
        frame = frame[y0:y0 + new_h, :]
    return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)


def write_frame_atomic(out_dir: Path, index: int, bgr: np.ndarray,
                       quality: int) -> Path:
    """Write ``bgr`` to ``frame_XXXXXXXX.jpg`` atomically; return the final path.

    The JPEG is encoded in memory (``cv2.imencode``, which selects the codec from
    the format string, not a filename) and the bytes written to a ``.part`` temp
    that is then renamed. This sidesteps ``cv2.imwrite`` refusing a non-image temp
    extension, and the consumer never sees a half-written or wrongly-named file.
    """
    final = out_dir / ("frame_%08d.jpg" % index)
    tmp = out_dir / ("frame_%08d.jpg.part" % index)   # ignored by the *.jpg consumers
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed for frame %d" % index)
    tmp.write_bytes(buf.tobytes())
    os.replace(tmp, final)                     # atomic: consumers never see a partial file
    return final


def prune_rolling_buffer(out_dir: Path, keep: int) -> None:
    """Keep only the newest ``keep`` ``frame_*.jpg`` files (bound disk use)."""
    files = sorted(out_dir.glob("frame_*.jpg"))
    for old in files[:-keep]:
        try:
            old.unlink()
        except OSError:
            pass


def clear_stale_frames(out_dir: Path) -> int:
    """Delete leftover frames from previous runs; return how many were removed.

    Runs start their index at 0, but an earlier run leaves *higher*-numbered
    ``frame_*.jpg`` files behind. Since both the rolling buffer and the consumer
    order by filename, those stale high numbers would out-rank every fresh frame —
    the prune would delete the new frames and the consumer would sit on an old one.
    Clearing on startup guarantees a clean, monotonic sequence.
    """
    removed = 0
    for p in list(out_dir.glob("frame_*.jpg")) + list(out_dir.glob("frame_*.jpg.part")):
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    return removed


class _RosFramePathPublisher:
    """Optional ROS2 ``/xtend/rgb_frame_path`` publisher (imported only if used)."""

    def __init__(self, topic: str) -> None:
        import rclpy                            # noqa: F401 -- optional dependency
        from rclpy.node import Node
        from std_msgs.msg import String
        rclpy.init()
        self._rclpy = rclpy
        self._String = String
        self._node = Node("webcam_frame_publisher")
        self._pub = self._node.create_publisher(String, topic, 10)

    def publish(self, path: str, stamp_s: float) -> None:
        sec = int(stamp_s)
        nsec = int(round((stamp_s - sec) * 1e9))
        self._pub.publish(self._String(data="%s %d %d" % (path, sec, nsec)))
        self._rclpy.spin_once(self._node, timeout_sec=0.0)

    def shutdown(self) -> None:
        self._node.destroy_node()
        self._rclpy.shutdown()


def open_camera(index: int):
    """Open the webcam, preferring the **V4L2** backend on Linux.

    OpenCV's default backend (often GStreamer) can hand back a single frame and
    then stall on many laptop webcams — the classic "only the first frame shows"
    symptom. V4L2 streams continuously; MJPG is requested to keep bandwidth sane.
    Falls back to the default backend if V4L2 is unavailable or won't open.
    """
    if hasattr(cv2, "CAP_V4L2"):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            except Exception:
                pass
            return cap
        cap.release()
    return cv2.VideoCapture(index)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    removed = clear_stale_frames(out_dir)      # start each session clean (see the helper)
    if removed:
        print("[webcam-pub] cleared %d stale frame(s) from a previous run" % removed)

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            "could not open camera index %d -- try a different --camera, close other "
            "apps using the webcam, or check camera permissions" % args.camera)

    ros = _RosFramePathPublisher(args.topic) if args.ros else None
    period = 1.0 / max(1.0, args.fps)
    print("[webcam-pub] %dx%d @ ~%.0f fps -> %s%s%s"
          % (args.width, args.height, args.fps, out_dir,
             "  (+topic %s)" % args.topic if ros else "",
             "  (preview: press q)" if args.show else ""))

    index = 0
    fails = 0
    stalled = False
    hb_t = time.time()
    hb_n = 0
    try:
        while True:
            t0 = time.time()
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if fails * period > 2.0 and not stalled:
                    print("[webcam-pub] WARNING: no camera frame for ~2s -- the camera "
                          "may have stalled. Try another --camera index, or run this "
                          "publisher alone with --show to watch the raw camera.")
                    stalled = True
                time.sleep(period)
                continue
            if stalled:
                print("[webcam-pub] camera recovered")
                stalled = False
            fails = 0
            bgr = center_crop_resize(frame, args.width, args.height)
            path = write_frame_atomic(out_dir, index, bgr, args.quality)
            prune_rolling_buffer(out_dir, args.buffer)
            if ros is not None:
                ros.publish(str(path), time.time())
            if args.show:
                cv2.imshow("webcam_frame_publisher (drone RGB mock)", bgr)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            index += 1
            hb_n += 1
            now = time.time()
            if now - hb_t >= 3.0:            # heartbeat: prove frames are actually flowing
                print("[webcam-pub] wrote %d frames (~%.1f fps)" % (index, hb_n / (now - hb_t)))
                hb_t, hb_n = now, 0
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        if ros is not None:
            ros.shutdown()
        print("[webcam-pub] stopped after %d frame(s)" % index)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0, help="webcam device index")
    ap.add_argument("--out", default="/tmp/xtend_frames",
                    help="rolling folder to write frames into (what the offline "
                         "target-lock tools read)")
    ap.add_argument("--width", type=int, default=504, help="drone RGB width")
    ap.add_argument("--height", type=int, default=294, help="drone RGB height")
    ap.add_argument("--fps", type=float, default=15.0, help="capture/write rate cap")
    ap.add_argument("--buffer", type=int, default=90,
                    help="rolling buffer size (newest N frames kept on disk)")
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality (1-100)")
    ap.add_argument("--ros", action="store_true",
                    help="also publish the ROS2 frame-path topic (needs rclpy)")
    ap.add_argument("--topic", default="/xtend/rgb_frame_path",
                    help="ROS2 frame-path topic when --ros is set")
    ap.add_argument("--show", action="store_true", help="show a local preview window")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
