"""Replay a recording through the frontier-view selector, offline: which past frames would each frontier node have been shown with?

Before any model is asked what lies behind a frontier, the question that
can be answered from a recording alone is whether the pictures to ask it
with exist. A recording holds the map of every action (``map_journal.npz``)
and the pose of every action (``steps.jsonl``); a frame log
(``OBJNAV_FRAME_LOG_DIR``) holds every frame under its capture pose. This
module walks the episode again, action by action, with no simulator, no
policy and no model:

1. the map of the action is read from the journal, and the frontier
   inventory rebuilt on it by the method's own code -- the planner the
   settings configure (``RPTSettings.planner_params``), the cost grids the
   policy reads, ``accessible_frontiers`` over the settled unknown
   (:func:`inventory_at`);
2. the inventory's clusters become **frontier nodes** with sticky ids
   (``core/planning/object_search/frontier_views/nodes.py``);
3. every node not yet scored is offered the frames logged so far, and the
   selector (``frontier_views/selection.py``) chooses its one to three views
   or says why it has none;
4. the score-once rule (``frontier_views/score_ledger.py``) scores a node
   the first action it has a usable view, never again, and marks one that
   is still there after the agent stood at it as blocked.

Steps 3 and 4 run once per **variant** over the same nodes, so one pass
compares what is being proposed against what it replaces:

* ``selected`` -- the selector as configured;
* ``through_only`` -- the strict reading of Strategy B (``at_weight`` 0):
  only a frame that looks past the boundary counts;
* ``current_frame`` -- no memory at all: only the frame of the action
  itself, which is what a model could be shown without a frame log.

Beside this module: ``frontier_view_frames`` finds the episode's frames,
``frontier_view_hindsight`` measures afterwards what the chosen frames
showed of the space mapped later, ``frontier_view_metrics`` turns the walk
into numbers, ``frontier_view_sheets`` into pictures to look at.

Usage::

    python -m sparx_agency.tasks.planning.objnav_benchmark_runtime.frontier_view_replay <recording dir> [...]
        [--frames <frame log root>] [--camera hm3d|mp3d] [--json out.json] [--sheets <dir>] [--max-sheets N]
        [--at-weight W] [--max-range M] [--max-frames N] [--corner-slack N] [--min-cells N] [--min-age N]
        [--no-hindsight] [-v]

Without ``--frames`` the frames are taken to be one per recorded action at
the recorded pose -- which is what a frame log holds -- so every recording
with a map journal can be replayed; the contact sheets need the images.

**How faithful the rebuilt inventory is, is measured.** The journal holds
the map each decision planned on (the policy stores the world with seen
staircases and drops masked back as the storey's map before the recording
reads it), and the recording lists the frontier goals of the inventory the
policy last built, with the action it built it on. Wherever that action is
the replayed one, the rebuilt goals are compared with the recorded ones
(:func:`same_as_recorded`) and the report says on how many actions they
were the same.

**What the replay does not reproduce.** The nodes are those of the route
the recorded policy walked: a policy that orders its visits differently
sees other frontiers.

Reads only; writes nothing into the recording. Python 3.9 syntax.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from sparx_agency.core.planning.environment import OccupancyGrid2D, OccupancyGrid2DParams
from sparx_agency.core.planning.exploration.frontier_ranking import FrontierInventory, accessible_frontiers
from sparx_agency.core.planning.object_search.frame_log.interfaces import FrameRecord
from sparx_agency.core.planning.object_search.frontier_views.nodes import FrontierNode, NodeSettings, frontier_nodes
from sparx_agency.core.planning.object_search.frontier_views.score_ledger import ScoreLedger
from sparx_agency.core.planning.object_search.frontier_views.selection import NodeViews, select_views
from sparx_agency.core.planning.object_search.frontier_views.settings import ViewSettings
from sparx_agency.core.planning.object_search.settings import RPTSettings
from sparx_agency.core.planning.objnav.types.pose import AgentPose
from sparx_agency.core.planning.planners.astar.cost_grid_2d import assemble_cost_grid
from sparx_agency.core.planning.planners.astar.weighted_planner_2d import WeightedAStarPlanner2D
from sparx_agency.tasks.planning.objnav_benchmark_runtime.frontier_view_frames import EpisodeFrames, load_frames, pose_of
from sparx_agency.tasks.planning.objnav_benchmark_runtime.review_recording import MapAtStep, load_json
from sparx_agency.tasks.planning.objnav_benchmark_runtime.room_extent_replay import VALUES, load_rows

#: Which frames a variant may choose from: every frame logged so far, or the frame of the action alone.
ALL_FRAMES, CURRENT_FRAME = "all", "current"
SELECTED, THROUGH_ONLY, CURRENT_ONLY = "selected", "through_only", "current_frame"
#: The agent is at a node when it stands this close to the node's goal point -- where a planner would send it to look.
VISIT_M = 1.0


@dataclass(frozen=True)
class Variant:
    """One way of choosing frames, replayed beside the others over the same nodes."""

    name: str
    settings: ViewSettings
    frames: str = ALL_FRAMES
    about: str = ""


def variants_for(settings: ViewSettings) -> Tuple[Variant, ...]:
    """The selector as configured, its strict reading, and no memory at all."""
    return (Variant(SELECTED, settings, about="the selector as configured (at_weight %.2f)" % settings.at_weight),
            Variant(THROUGH_ONLY, replace(settings, at_weight=0.0),
                    about="only a frame that looks PAST the boundary counts (at_weight 0)"),
            Variant(CURRENT_ONLY, settings, frames=CURRENT_FRAME,
                    about="no frame log: only the frame of the action itself"))


@dataclass
class ScoringEvent:
    """One node scored: the action, the node as it stood, the frames it was scored on."""

    variant: str
    step: int
    floor_id: int
    node: FrontierNode
    first_step: int
    pose: AgentPose
    views: NodeViews
    hindsight: Optional[Dict] = None
    later: Optional[Dict] = None

    def describe(self) -> str:
        """One line: the action, the variant, the node and what each chosen frame shows of it."""
        return "step %3d  %-13s node %3d (%3d cells, %d part%s)  %s" % (
            self.step, self.variant, self.node.node_id, self.node.size_cells, self.node.parts,
            "" if self.node.parts == 1 else "s",
            "  ".join("%s seen %.0f%% past %.0f%% at %.1f m" % (
                view.name, 100 * view.seen_fraction, 100 * view.through_fraction, view.range_m)
                for view in self.views.views))


@dataclass
class EpisodeReplay:
    """Everything one replayed episode produced; the hindsight, the metrics and the sheets read it."""

    directory: Path
    episode: Dict
    maps: MapAtStep
    frames: EpisodeFrames
    variants: Tuple[Variant, ...]
    node_settings: NodeSettings
    #: One record per action: its nodes, and whether the rebuilt inventory was the recorded one.
    steps: List[Dict] = field(default_factory=list)
    events: Dict[str, List[ScoringEvent]] = field(default_factory=dict)
    ledgers: Dict[str, ScoreLedger] = field(default_factory=dict)
    #: ``{variant: {node_id: NodeViews}}`` -- what the selector last said of every node it was asked about.
    last_views: Dict[str, Dict[int, NodeViews]] = field(default_factory=dict)
    #: ``{node_id: (step, floor_id, node, agent pose)}`` -- every node as it stood on the last action it was present.
    last_nodes: Dict[int, Tuple[int, int, FrontierNode, AgentPose]] = field(default_factory=dict)

    def world(self, step: int, floor_id: int):
        """``(world, settled)`` -- the occupancy grid and the settled unknown as they stood after action ``step``."""
        grid, settled, _ = self.maps.at(step, floor_id)
        return grid_world(grid, self.maps), settled

    def offered(self, variant: Variant, step: int) -> List[FrameRecord]:
        """The frames ``variant`` may choose from at action ``step``."""
        return self.frames.until(step) if variant.frames == ALL_FRAMES else self.frames.at(step)


def grid_world(grid, maps: MapAtStep) -> OccupancyGrid2D:
    """A journal layer as the occupancy grid the method reads."""
    return OccupancyGrid2D(grid, OccupancyGrid2DParams(maps.resolution, maps.origin[0], maps.origin[1], "world"),
                           values=VALUES)


def inventory_at(maps: MapAtStep, planner: WeightedAStarPlanner2D, step: int, floor_id: int, pose: AgentPose):
    """``(world, settled, inventory)`` of one recorded action: the frontier inventory rebuilt on the map of that action.

    As the policy builds it (``SceneGraph.refresh_accessibility``): reachable
    at the planner's floor standoff, distances at the preferred standoff
    where that reaches, the frontier read with the settled unknown closed.
    """
    grid, settled, labels = maps.at(step, floor_id)
    world = grid_world(grid, maps)
    overlay = grid.copy()
    overlay[settled & (grid == VALUES.unknown)] = VALUES.occupied
    params = planner.params
    cost = assemble_cost_grid(planner.fields_for(world), params, params.inflate_floor_m)[0]
    inventory = accessible_frontiers(world, cost, labels, (pose.x, pose.y), pose.yaw, None,
                                     preferred_cost=planner.cost_for(world)[0], frontier_world=grid_world(overlay, maps))
    return world, settled, inventory


def same_as_recorded(row: Dict, step: int, inventory: FrontierInventory) -> Optional[bool]:
    """Whether the rebuilt inventory has the frontier goals the recording lists for this action; None when it cannot be told.

    The recording's snapshot carries the inventory the policy last built and
    the action it built it on (``frontier_inventory_step``). Between rebuilds
    -- a committed transit refreshes its geometry every few actions -- the
    snapshot is older than the map, and nothing is compared. A goal is
    compared by the cluster it stands for: its cell and its size.
    """
    search = (row.get("method") or {}).get("search") or {}
    goals = search.get("frontier_goals")
    if goals is None or search.get("frontier_inventory_step") != step:
        return None
    recorded = sorted((int(goal["cell"][0]), int(goal["cell"][1]), int(goal["size_cells"])) for goal in goals)
    return recorded == sorted((goal.cell[0], goal.cell[1], goal.size_cells) for goal in inventory.goals)


def replay(directory, frames_root=None, camera=None, view_settings: Optional[ViewSettings] = None,
           node_settings: Optional[NodeSettings] = None, method_settings: Optional[RPTSettings] = None,
           variants: Optional[Sequence[Variant]] = None, visit_m: float = VISIT_M, min_age_steps: int = 0,
           verbose: bool = False) -> EpisodeReplay:
    """Walk the recording in ``directory`` and score its frontier nodes under every variant.

    Args:
        directory: A recording directory (``.../recordings/<key>``) with a map journal.
        frames_root: A frame log root (or one episode's folder); None takes
            one frame per recorded action at its recorded pose.
        camera: ``hm3d`` or ``mp3d``; by default the frame log's own camera,
            else the benchmark's by the scene id.
        view_settings: The selector's settings (the ``selected`` variant's).
        node_settings: What a frontier node is.
        method_settings: The method's settings, for the planner the frontier
            inventory is qualified by.
        variants: The variants to replay; :func:`variants_for` by default.
        visit_m: The agent is at a node within this distance of its goal point.
        min_age_steps: Actions a node must have been there before it is scored (``ScoreLedger.min_age_steps``).
        verbose: Print every scoring as it happens.

    Returns:
        The :class:`EpisodeReplay`.

    Raises:
        SystemExit: Without a map journal, or with a frame log that is not this recording's.
    """
    directory = Path(directory)
    maps = MapAtStep(directory)
    if not maps.exact:
        raise SystemExit("%s has no map_journal.npz: the map of each action cannot be replayed" % directory)
    episode = load_json(directory, "episode.json") or {}
    rows = load_rows(directory)
    node_settings = node_settings or NodeSettings()
    variants = tuple(variants) if variants is not None else variants_for(view_settings or ViewSettings())
    replayed = EpisodeReplay(directory, episode, maps, load_frames(directory, rows, episode, frames_root, camera),
                             variants, node_settings,
                             events={variant.name: [] for variant in variants},
                             ledgers={variant.name: ScoreLedger(min_age_steps=min_age_steps) for variant in variants},
                             last_views={variant.name: {} for variant in variants})
    planner = WeightedAStarPlanner2D((method_settings or RPTSettings()).planner_params())
    registry = node_settings.registry()
    for row in rows:
        step, pose = int(row["step"]), pose_of(row)
        floor_id = int((row.get("method") or {}).get("floor_id", 0) or 0)
        world, settled, inventory = inventory_at(maps, planner, step, floor_id, pose)
        nodes = frontier_nodes(world, inventory.clusters, registry, floor_id, node_settings)
        replayed.last_nodes.update({node.node_id: (step, floor_id, node, pose) for node in nodes})
        replayed.steps.append({"step": step, "floor": floor_id, "clusters": len(inventory.clusters),
                               "nodes": [node.node_id for node in nodes], "cells": [node.size_cells for node in nodes],
                               "as_recorded": same_as_recorded(row, step, inventory)})
        for variant in variants:
            for event in _score(replayed, variant, world, settled, nodes, step, floor_id, pose, visit_m):
                if verbose:
                    print(event.describe())
    return replayed


def _score(replayed: EpisodeReplay, variant: Variant, world, settled, nodes: Sequence[FrontierNode], step: int,
           floor_id: int, pose: AgentPose, visit_m: float) -> List[ScoringEvent]:
    """One action of one variant under the score-once rule; returns the scorings it made."""
    ledger = replayed.ledgers[variant.name]
    by_id = {node.node_id: node for node in nodes}
    ledger.observe(step, by_id)
    for node in nodes:
        if math.dist((pose.x, pose.y), node.xy) <= visit_m:
            ledger.mark_visited(node.node_id, step)
    pending = [by_id[node_id] for node_id in ledger.pending()]
    if not pending:
        return []
    views = select_views(world, pending, replayed.offered(variant, step), replayed.frames.camera, variant.settings,
                         settled, reference_z=pose.z)
    replayed.last_views[variant.name].update(views)
    scored = []
    for node_id in ledger.due(views):
        ledger.mark_scored(node_id, step, views[node_id].names)
        scored.append(ScoringEvent(variant.name, step, floor_id, by_id[node_id], ledger.nodes[node_id].first_step, pose,
                                   views[node_id]))
    replayed.events[variant.name].extend(scored)
    return scored


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("recordings", nargs="+", type=Path, help="Recording directories (.../recordings/<key>)")
    parser.add_argument("--frames", type=Path, default=None,
                        help="A frame log root (OBJNAV_FRAME_LOG_DIR of the run); without it one frame per recorded action")
    parser.add_argument("--camera", choices=("hm3d", "mp3d"), default=None, help="By default the frame log's, else by scene id")
    parser.add_argument("--json", type=Path, default=None, help="Write the full result here")
    parser.add_argument("--sheets", type=Path, default=None, help="Write a contact sheet per scored node under this directory")
    parser.add_argument("--max-sheets", type=int, default=40, help="Contact sheets per episode at most")
    parser.add_argument("--at-weight", type=float, default=None, help="ViewSettings.at_weight of the selected variant")
    parser.add_argument("--max-range", type=float, default=None, help="ViewSettings.max_range_m")
    parser.add_argument("--max-frames", type=int, default=None, help="ViewSettings.max_frames")
    parser.add_argument("--corner-slack", type=int, default=None, help="ViewSettings.corner_slack_cells")
    parser.add_argument("--min-cells", type=int, default=None, help="NodeSettings.min_cells")
    parser.add_argument("--min-age", type=int, default=0,
                        help="Actions a node must have been there before it is scored (0: the action it appears)")
    parser.add_argument("--no-hindsight", action="store_true", help="Skip what the chosen frames showed of the space mapped later")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print every scoring")
    return parser


def settings_from(args) -> Tuple[ViewSettings, NodeSettings]:
    """The selector's and the nodes' settings with the command line's overrides."""
    view = {name: value for name, value in (("at_weight", args.at_weight), ("max_range_m", args.max_range),
                                            ("max_frames", args.max_frames),
                                            ("corner_slack_cells", args.corner_slack)) if value is not None}
    node = {} if args.min_cells is None else {"min_cells": args.min_cells}
    return ViewSettings(**view), NodeSettings(**node)


def main(argv=None):
    from sparx_agency.tasks.planning.objnav_benchmark_runtime import frontier_view_hindsight as hindsight
    from sparx_agency.tasks.planning.objnav_benchmark_runtime import frontier_view_metrics as metrics
    args = build_parser().parse_args(argv)
    view_settings, node_settings = settings_from(args)
    if args.sheets is not None and args.frames is None:
        raise SystemExit("--sheets draws the chosen frames: give the run's frame log with --frames")
    results = []
    for directory in args.recordings:
        replayed = replay(directory, args.frames, args.camera, view_settings, node_settings,
                          min_age_steps=args.min_age, verbose=args.verbose)
        if not args.no_hindsight:
            hindsight.add_hindsight(replayed)
            hindsight.add_patience(replayed)
        result = metrics.summarise(replayed)
        results.append(result)
        print(metrics.report(result))
        if args.sheets is not None:
            from sparx_agency.tasks.planning.objnav_benchmark_runtime import frontier_view_sheets as sheets
            written = sheets.write_sheets(replayed, args.sheets / directory.name, limit=args.max_sheets)
            print("%d contact sheet%s under %s" % (len(written), "" if len(written) == 1 else "s", args.sheets / directory.name))
    if len(results) > 1:
        print(metrics.report(metrics.pooled(results)))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"episodes": results, "pooled": metrics.pooled(results) if len(results) > 1 else None},
                                        indent=1, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
