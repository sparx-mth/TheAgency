"""Contact sheets of a frontier-view replay: every scored node, its place on the map and the frames it was scored on.

A number says a frame shows 44 % of a node; whether the frame is a picture
of the doorway or of the sofa in front of it is a thing to look at. One
image per scoring (``frontier_view_replay.ScoringEvent``):

* **left, the map of the action** -- free floor light, walls black, unknown
  dark, settled unknown brown; the node's cells red; the unknown behind it
  that the episode mapped LATER green (the hindsight of
  ``frontier_view_metrics``: what there was to show); the agent as it stood
  (white) with its trail; each chosen frame's camera with the edges of its
  field of view, in the frame's colour;
* **right, the chosen frames**, best first, each with its **window** (the
  box the node lands in) and, over every cell of the node the frame sees, the
  height band it was judged on -- green where the frame looks past the cell
  into the unknown, yellow where it sees the cell and no further.

A node that lived long and was **never scored** gets a sheet too
(``unscored_*.jpg``): the map of its last action and the two frames that
came closest -- the most of its cells seen, then the most in the picture --
with the band red over every cell the frame faces and its sight line does
not reach. That is where a selector's blind spots show: a red band over
open floor behind a bench is the two-dimensional map hiding what the camera
looked straight over.

Looking at a dozen of these is the check the numbers cannot do: that the
band lies on the opening in the picture (the projection is right), that
"occluded" means a wall and not a chair, and that what the frame shows is
something a model could say anything about.

Needs the frame log's images; ``cv2`` is imported when a sheet is drawn.
Python 3.9 syntax.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import numpy as np

from sparx_agency.core.planning.object_search.frontier_views.selection import storey_frames
from sparx_agency.core.planning.object_search.frontier_views.sight import OCCLUDED, SEEN, SightScene
from sparx_agency.tasks.planning.objnav_benchmark_runtime.frontier_view_metrics import LONG_LIVED_STEPS
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_extent_replay import VALUES

MAP_PX = 520
FRAME_SCALE = 2.0 / 3.0
BAR_PX = 44
#: BGR. One colour per chosen frame, best first.
VIEW_COLORS = ((235, 190, 40), (200, 60, 220), (40, 170, 250))
UNKNOWN, FREE, OCCUPIED, SETTLED = (58, 58, 58), (228, 228, 228), (0, 0, 0), (70, 95, 125)
NODE, REVEALED, TRAIL, AGENT = (40, 40, 235), (110, 190, 110), (60, 150, 255), (255, 255, 255)
THROUGH, AT, HIDDEN = (70, 220, 70), (40, 220, 240), (50, 50, 240)


def write_sheets(replayed, directory, limit: int = 40, variant: Optional[str] = None) -> List[Path]:
    """Draw up to ``limit`` scorings of ``variant`` (the first by default), spread over the episode; returns the files written.

    Raises:
        SystemExit: If the replay has no frame images to draw.
    """
    import cv2
    if replayed.frames.directory is None:
        raise SystemExit("contact sheets draw the chosen frames: replay with a frame log (--frames)")
    name = variant or replayed.variants[0].name
    events = replayed.events[name]
    if limit and len(events) > limit:
        events = [events[int(round(i))] for i in np.linspace(0, len(events) - 1, limit)]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for event in events:
        path = directory / ("step_%04d_node_%03d.jpg" % (event.step, event.node.node_id))
        cv2.imwrite(str(path), sheet(replayed, event), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        written.append(path)
    ledger = replayed.ledgers[name]
    for record in sorted(ledger.nodes.values(), key=lambda record: record.first_step):
        if record.scored_step is not None or record.last_step - record.first_step + 1 < LONG_LIVED_STEPS:
            continue
        path = directory / ("unscored_node_%03d_step_%04d.jpg" % (record.node_id, record.last_step))
        cv2.imwrite(str(path), miss_sheet(replayed, name, record), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        written.append(path)
    return written


def sheet(replayed, event) -> np.ndarray:
    """One scoring as a BGR image: the map panel, then the chosen frames."""
    import cv2
    settings = next(variant.settings for variant in replayed.variants if variant.name == event.variant)
    world, settled = replayed.world(event.step, event.floor_id)
    records = {record.name: record for record in replayed.frames.records}
    chosen = [records[view.name] for view in event.views.views]
    scene = SightScene(world, event.node.rows, event.node.cols, replayed.frames.camera, settings, settled)
    panels = [_map_panel(replayed, event.step, event.node, event.pose, world, settled, chosen,
                         (event.hindsight or {}).get("cells"))]
    for index, (view, record) in enumerate(zip(event.views.views, chosen)):
        age = event.step - view.step
        captions = ("%s  %s" % (view.name, "the action's own frame" if age == 0 else "%d actions earlier" % age),
                    "%.1f m   sees %.0f%% of the node   past %.0f%%   gain %.2f" % (
                        view.range_m, 100 * view.seen_fraction, 100 * view.through_fraction, view.gain))
        panels.append(_frame_panel(_image(replayed, record), scene.sight(record.pose), view.window, captions,
                                   VIEW_COLORS[index % len(VIEW_COLORS)]))
    hindsight = event.hindsight or {}
    text = "%s  action %d  node %d: %d cells in %d part%s, first seen at action %d   score %.2f  seen %.0f%%  past %.0f%%" % (
        replayed.episode.get("episode_id", replayed.directory.name), event.step, event.node.node_id, event.node.size_cells,
        event.node.parts, "" if event.node.parts == 1 else "s", event.first_step, event.views.score,
        100 * event.views.seen_fraction, 100 * event.views.through_fraction)
    if "shown" in hindsight:
        later = "mapped later behind it: %d cells -- in the chosen frames %.0f%%, in any frame so far %.0f%%" % (
            hindsight["revealed_cells"], 100 * hindsight["shown"], 100 * hindsight["available"])
    else:
        later = "mapped later behind it: %d cells (too few to measure)" % hindsight.get("revealed_cells", 0)
    return _compose(panels, text, later)


def miss_sheet(replayed, variant: str, record) -> np.ndarray:
    """A never-scored node on the map of its last action, with the frames that came closest to showing it."""
    settings = next(v.settings for v in replayed.variants if v.name == variant)
    step, floor_id, node, pose = replayed.last_nodes[record.node_id]
    world, settled = replayed.world(step, floor_id)
    scene = SightScene(world, node.rows, node.cols, replayed.frames.camera, settings, settled)
    offered = storey_frames(replayed.frames.until(step), pose.z, settings.storey_tolerance_m)
    sights = [scene.sight(frame.pose) for frame in offered]
    # The closest misses: the most cells seen, then the most in the picture, then the nearest.
    order = sorted(range(len(offered)), key=lambda i: (-int((sights[i].stage == SEEN).sum()),
                                                        -int((sights[i].stage >= OCCLUDED).sum()),
                                                        float(np.median(sights[i].range_m))))
    close = [i for i in order if (sights[i].stage >= OCCLUDED).any()][:2]
    panels = [_map_panel(replayed, step, node, pose, world, settled, [offered[i] for i in close], None)]
    for index, i in enumerate(close):
        sight = sights[i]
        framed = sight.stage >= OCCLUDED
        window = (float(np.nanmin(sight.u[framed])), float(np.nanmin(sight.v_top[framed])),
                  float(np.nanmax(sight.u[framed])), float(np.nanmax(sight.v_bottom[framed])))
        captions = ("%s  %d actions earlier" % (offered[i].name, step - offered[i].step),
                    "%.1f m   in the picture %.0f%% of the node   seen %.0f%%   (red: faced, not reached)" % (
                        float(np.median(sight.range_m[framed])), 100.0 * framed.mean(), 100.0 * (sight.stage == SEEN).mean()))
        panels.append(_frame_panel(_image(replayed, offered[i]), sight, window, captions,
                                   VIEW_COLORS[index % len(VIEW_COLORS)], hidden=True))
    last = replayed.last_views[variant].get(record.node_id)
    text = "%s  node %d NEVER SCORED: present from action %d to %d, %d cells in %d part%s at the end" % (
        replayed.episode.get("episode_id", replayed.directory.name), record.node_id, record.first_step, record.last_step,
        node.size_cells, node.parts, "" if node.parts == 1 else "s")
    why = "why: %s (best single frame scored %.3f, %d frames asked)" % (
        (last.reason, last.best_score, last.frames) if last is not None else ("never offered a frame", 0.0, 0))
    return _compose(panels, text, why)


def _image(replayed, record) -> np.ndarray:
    import cv2
    image = cv2.imread(str(replayed.frames.directory / record.name))
    if image is None:
        raise SystemExit("%s is in the frame log's index but its image cannot be read" % (replayed.frames.directory / record.name))
    return image


def _compose(panels, first_line: str, second_line: str) -> np.ndarray:
    """The panels side by side under a two-line title bar."""
    import cv2
    height = max(panel.shape[0] for panel in panels)
    body = np.concatenate([np.pad(panel, ((0, height - panel.shape[0]), (0, 4), (0, 0))) for panel in panels], axis=1)
    if body.shape[1] < 1000:                           # room for the title when there is no frame to show
        body = np.pad(body, ((0, 0), (0, 1000 - body.shape[1]), (0, 0)))
    bar = np.zeros((BAR_PX, body.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, first_line, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, second_line, (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.47, REVEALED, 1, cv2.LINE_AA)
    return np.concatenate([bar, body], axis=0)


def _map_panel(replayed, step, node, pose, world, settled, chosen, revealed) -> np.ndarray:
    """The map of action ``step`` round the node, the agent and the chosen cameras, north up."""
    import cv2
    maps, grid = replayed.maps, world.grid
    res, (ox, oy) = maps.resolution, maps.origin
    xs = [(c + 0.5) * res + ox for c in (node.cols.min(), node.cols.max())] + [pose.x] + [r.pose.x for r in chosen]
    ys = [(r + 0.5) * res + oy for r in (node.rows.min(), node.rows.max())] + [pose.y] + [r.pose.y for r in chosen]
    side = max(6.0, max(xs) - min(xs) + 3.0, max(ys) - min(ys) + 3.0)
    cx, cy = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))
    cells = int(math.ceil(side / res))
    x0, y0 = int(math.floor((cx - 0.5 * side - ox) / res)), int(math.floor((cy - 0.5 * side - oy) / res))
    colors = np.full((cells, cells, 3), UNKNOWN, dtype=np.uint8)
    gy0, gy1, gx0, gx1 = max(0, y0), min(grid.shape[0], y0 + cells), max(0, x0), min(grid.shape[1], x0 + cells)
    view = colors[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0]
    part = grid[gy0:gy1, gx0:gx1]
    view[part == VALUES.free] = FREE
    view[part == VALUES.occupied] = OCCUPIED
    view[(part == VALUES.unknown) & settled[gy0:gy1, gx0:gx1]] = SETTLED

    def paint(rows, cols, color):
        keep = (rows >= y0) & (rows < y0 + cells) & (cols >= x0) & (cols < x0 + cells)
        colors[rows[keep] - y0, cols[keep] - x0] = color

    if revealed is not None:
        paint(revealed[0], revealed[1], REVEALED)
    paint(node.rows, node.cols, NODE)
    scale = MAP_PX / float(cells)
    panel = cv2.resize(colors[::-1], (MAP_PX, MAP_PX), interpolation=cv2.INTER_NEAREST)

    def pixel(x, y):
        return (int(round(((x - ox) / res - x0) * scale)), int(round(MAP_PX - ((y - oy) / res - y0) * scale)))

    trail = [pixel(r.pose.x, r.pose.y) for r in replayed.frames.until(step)]
    for a, b in zip(trail, trail[1:]):
        cv2.line(panel, a, b, TRAIL, 1, cv2.LINE_AA)
    half = math.atan(0.5 * replayed.frames.camera.intrinsics.width / replayed.frames.camera.intrinsics.fx)
    for index, record in enumerate(chosen):
        color = VIEW_COLORS[index % len(VIEW_COLORS)]
        here = pixel(record.pose.x, record.pose.y)
        for edge in (-half, half):
            there = pixel(record.pose.x + 2.5 * math.cos(record.pose.yaw + edge),
                          record.pose.y + 2.5 * math.sin(record.pose.yaw + edge))
            cv2.line(panel, here, there, color, 1, cv2.LINE_AA)
        cv2.circle(panel, here, 6, color, -1, cv2.LINE_AA)
        cv2.putText(panel, str(index + 1), (here[0] + 8, here[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    here = pixel(pose.x, pose.y)
    nose = pixel(pose.x + 0.5 * math.cos(pose.yaw), pose.y + 0.5 * math.sin(pose.yaw))
    cv2.circle(panel, here, 4, AGENT, -1, cv2.LINE_AA)
    cv2.line(panel, here, nose, AGENT, 2, cv2.LINE_AA)
    cv2.putText(panel, "map at action %d (%.0f m across)" % (step, side), (6, MAP_PX - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return panel


def _frame_panel(image, sight, window, captions, color, hidden: bool = False) -> np.ndarray:
    """One frame with the node's window and the height band over every cell of it the frame sees (and, with ``hidden``, faces)."""
    import cv2
    overlay = image.copy()
    for index in np.nonzero(sight.stage >= (OCCLUDED if hidden else SEEN))[0]:
        column = int(round(float(sight.u[index])))
        shade = HIDDEN if sight.stage[index] != SEEN else (THROUGH if sight.through[index] else AT)
        cv2.line(overlay, (column, int(round(float(sight.v_bottom[index])))), (column, int(round(float(sight.v_top[index])))),
                 shade, 2)
    image = cv2.addWeighted(overlay, 0.45, image, 0.55, 0.0)
    left, top, right, bottom = (int(round(v)) for v in window)
    cv2.rectangle(image, (left - 3, top - 3), (right + 3, bottom + 3), color, 2)
    panel = cv2.resize(image, None, fx=FRAME_SCALE, fy=FRAME_SCALE, interpolation=cv2.INTER_AREA)
    caption = np.zeros((40, panel.shape[1], 3), dtype=np.uint8)
    cv2.putText(caption, captions[0], (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(caption, captions[1], (6, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate([panel, caption], axis=0)
