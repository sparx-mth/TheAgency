#!/usr/bin/env python3
"""
Visual pipeline test: occupancy → inflation → SDF → Voronoi topology
                      → door separation → room–object graph.

Creates a synthetic apartment and plots every pipeline stage.
All algorithmic code lives in sparx_agency.core — this file is
only synthetic data + visualization + orchestration.

Place at: sparx_agency/tasks/mapping/test_topology_pipeline.py
"""
import os
import sys
import time
import types

# Stub out ROS packages so we can import sparx_agency.core.mapping
# without a full ROS workspace (mapping/__init__.py chains into sensor_msgs).
class _RosStub(types.ModuleType):
    """Module stub that returns a dummy for any attribute access."""
    def __getattr__(self, name):
        return type(name, (), {"__init__": lambda *a, **k: None})

for _mod in ["sensor_msgs", "sensor_msgs.msg",
             "std_msgs", "std_msgs.msg",
             "nav_msgs", "nav_msgs.msg",
             "geometry_msgs", "geometry_msgs.msg",
             "rospy", "rclpy", "rclpy.node",
             "cv_bridge", "tf2_ros"]:
    sys.modules.setdefault(_mod, _RosStub(_mod))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt

from sparx_agency.core.mapping.costmap.inflation import inflate_occupancy, InflationParams
from sparx_agency.core.mapping.costmap.sdf import compute_sdf, boundary_cost_field
from sparx_agency.core.mapping.topology import (
    extract_voronoi_graph, TopologyParams, get_junctions, get_dead_ends,
    separate_rooms, DoorInfo, RoomSeparationParams,
    ObjectInfo, NodeType, assign_objects_to_rooms, build_room_object_graph,
    get_room_nodes, get_objects_in_room,
)

# ========================= SYNTHETIC APARTMENT =============================

def create_apartment_occupancy(shape=(200, 300)) -> np.ndarray:
    """
    Draw a synthetic apartment: outer walls, 4 rooms, a corridor,
    door openings, and furniture.  Returns uint8 grid where 0=free
    and each nonzero value is a unique object ID (wall segment or furniture).
    """
    H, W = shape
    occ = np.zeros((H, W), dtype=np.uint8)
    wall = 2
    _id = [0]

    def box(r1, r2, c1, c2):
        _id[0] += 1
        occ[r1:r2, c1:c2] = _id[0]

    # Outer walls (4 separate segments)
    box(0, wall, 0, W)          # top wall
    box(H - wall, H, 0, W)      # bottom wall
    box(0, H, 0, wall)           # left wall
    box(0, H, W - wall, W)       # right wall

    # Vertical divider at col 140
    div_c = 140
    box(0, H, div_c, div_c + wall)

    # Corridor band rows 85..115
    corr_top, corr_bot = 85, 115
    box(corr_top, corr_top + wall, 0, div_c)
    box(corr_bot, corr_bot + wall, 0, div_c)
    box(corr_top, corr_top + wall, div_c, W)

    # Door openings (gaps)
    occ[corr_top:corr_top + wall, 45:60] = 0
    occ[corr_bot:corr_bot + wall, 45:60] = 0
    occ[40:55, div_c:div_c + wall] = 0
    occ[130:145, div_c:div_c + wall] = 0

    # ── Room 0 — Living room (top-left) ──
    box(25, 35, 15, 25)     # TV
    box(35, 42, 55, 72)     # Table
    box(20, 30, 55, 80)     # Sofa

    # ── Room 1 — Bedroom (top-right) ──
    box(30, 55, 180, 220)   # Bed
    box(30, 38, 225, 233)   # Night lamp
    box(10, 35, 260, 275)   # Wardrobe

    # ── Room 2 — Bathroom (bottom-left) ──
    box(140, 165, 15, 40)   # Bathtub
    box(130, 138, 60, 72)   # Toilet

    # ── Room 3 — Kitchen (bottom-right) ──
    box(140, 155, 185, 215) # Kettle
    box(120, 130, 245, 290) # Oven
    box(165, 175, 175, 195) # Refrigerator

    # Room 4 — Corridor has no furniture

    return occ


def create_apartment_objects() -> list[ObjectInfo]:
    """
    Define tangible objects matching the furniture boxes drawn in
    :func:`create_apartment_occupancy`.  Positions are box centres.
    """
    return [
        # ── Room 0 — Living room ──
        ObjectInfo("tv",    np.array([30.0,  20.0]),  (25, 35, 15, 25),   "tv"),
        ObjectInfo("sofa",  np.array([25.0,  67.5]),  (20, 30, 55, 80),   "sofa"),
        ObjectInfo("table", np.array([38.5,  63.5]),  (35, 42, 55, 72),   "table"),
        # ── Room 1 — Bedroom ──
        ObjectInfo("bed",      np.array([42.5, 200.0]), (30, 55, 180, 220), "bed"),
        ObjectInfo("Night lamp",  np.array([34.0, 229.0]), (30, 38, 225, 233), "Night lamp"),
        ObjectInfo("wardrobe", np.array([22.5, 267.5]), (10, 35, 260, 275), "wardrobe"),
        # ── Room 2 — Bathroom ──
        ObjectInfo("bathtub", np.array([152.5, 27.5]), (140, 165, 15, 40), "bathtub"),
        ObjectInfo("toilet",  np.array([134.0, 66.0]), (130, 138, 60, 72), "toilet"),
        # ── Room 3 — Kitchen ──
        ObjectInfo("kettle",       np.array([147.5, 200.0]), (140, 155, 185, 215), "kettle"),
        ObjectInfo("oven",         np.array([125.0, 267.5]), (120, 130, 245, 290), "oven"),
        ObjectInfo("refrigerator", np.array([170.0, 185.0]), (165, 175, 175, 195), "refrigerator"),
    ]


def create_apartment_doors() -> list[DoorInfo]:
    """
    Define the 4 door openings matching the gaps in
    :func:`create_apartment_occupancy`.
    """
    return [
        DoorInfo(position=np.array([86.0,  52.0]),  size=np.array([0.10, 0.40])),   # corridor → living room
        DoorInfo(position=np.array([116.0, 52.0]),  size=np.array([0.10, 0.40])),   # corridor → bathroom
        DoorInfo(position=np.array([47.0,  141.0]), size=np.array([0.40, 0.10])),   # living room → bedroom
        DoorInfo(position=np.array([137.0, 141.0]), size=np.array([0.40, 0.10])),   # bathroom → kitchen
    ]


# ============================== VISUALIZATION ===============================

def plot_occupancy(ax, occ, title):
    ax.imshow(occ != 0, cmap="gray_r", interpolation="nearest")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_axis_off()


def plot_inflated(ax, occ_orig, occ_inflated, title):
    rgb = np.ones((*occ_inflated.shape, 3))
    rgb[occ_inflated != 0] = [1.0, 0.7, 0.7]
    rgb[occ_orig != 0] = [0.0, 0.0, 0.0]
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_axis_off()


def plot_sdf(ax, sdf, title):
    vmax = np.abs(sdf).max()
    im = ax.imshow(sdf, cmap="RdBu", vmin=-vmax, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="meters")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_axis_off()


def plot_boundary_cost(ax, cost, title):
    im = ax.imshow(cost, cmap="hot", vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cost")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_axis_off()


def plot_voronoi_graph(ax, occ, G, title):
    rgb = np.ones((*occ.shape, 3)) * 0.95
    rgb[occ != 0] = [0.2, 0.2, 0.2]
    ax.imshow(rgb, interpolation="nearest")

    if len(G.nodes) == 0:
        ax.set_title(title + " (empty)", fontsize=11, fontweight="bold")
        ax.set_axis_off()
        return

    junctions = set(get_junctions(G))
    dead_ends = set(get_dead_ends(G))
    passthrough = set(G.nodes) - junctions - dead_ends

    for u, v in G.edges():
        ax.plot([u[1], v[1]], [u[0], v[0]], color="#4488cc", linewidth=0.6, alpha=0.7)

    def scatter_set(nodes, color, size, label):
        if nodes:
            arr = np.array(list(nodes))
            ax.scatter(arr[:, 1], arr[:, 0], c=color, s=size, zorder=5,
                       label=label, edgecolors="none")

    scatter_set(passthrough, "#4488cc", 3, f"Pass-through [{len(passthrough)}]")
    scatter_set(junctions, "#ff3333", 30, f"Junctions (deg≥3) [{len(junctions)}]")
    scatter_set(dead_ends, "#33cc33", 18, f"Dead-ends (deg=1) [{len(dead_ends)}]")

    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_axis_off()


def plot_components(ax, occ, inflated, G, title):
    rgb = np.ones((*occ.shape, 3)) * 0.95
    rgb[inflated != 0] = [0.85, 0.85, 0.85]
    rgb[occ != 0] = [0.2, 0.2, 0.2]
    ax.imshow(rgb, interpolation="nearest")

    if len(G.nodes) == 0:
        ax.set_title(title + " (empty)", fontsize=11, fontweight="bold")
        ax.set_axis_off()
        return

    junctions = set(get_junctions(G))
    dead_ends = set(get_dead_ends(G))
    passthrough = set(G.nodes) - junctions - dead_ends

    for u, v in G.edges():
        ax.plot([u[1], v[1]], [u[0], v[0]], color="#4488cc", linewidth=0.6, alpha=0.7)

    def scatter_set(nodes, color, size, label):
        if nodes:
            arr = np.array(list(nodes))
            ax.scatter(arr[:, 1], arr[:, 0], c=color, s=size, zorder=5,
                       label=label, edgecolors="none")

    scatter_set(passthrough, "#4488cc", 3, f"Pass-through [{len(passthrough)}]")
    scatter_set(junctions, "#ff3333", 30, f"Junctions (deg≥3) [{len(junctions)}]")
    scatter_set(dead_ends, "#33cc33", 18, f"Dead-ends (deg=1) [{len(dead_ends)}]")

    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_axis_off()


def plot_door_field(ax, occ, door_field, doors, title):
    """Overlay door probability heatmap on the occupancy grid."""
    rgb = np.ones((*occ.shape, 3)) * 0.95
    rgb[occ != 0] = [0.2, 0.2, 0.2]
    ax.imshow(rgb, interpolation="nearest")

    masked = np.ma.masked_where(door_field < 1e-6, door_field)
    im = ax.imshow(masked, cmap="magma", interpolation="nearest", alpha=0.7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="density")

    for d in doors:
        ax.plot(d.position[1], d.position[0], "c*", markersize=10, zorder=6)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_axis_off()


def plot_room_components(ax, occ, inflated, G_sep, title):
    """Color each connected component (room) in a distinct color."""
    rgb = np.ones((*occ.shape, 3)) * 0.95
    rgb[inflated != 0] = [0.85, 0.85, 0.85]
    rgb[occ != 0] = [0.2, 0.2, 0.2]
    ax.imshow(rgb, interpolation="nearest")

    if len(G_sep.nodes) == 0:
        ax.set_title(title + " (empty)", fontsize=11, fontweight="bold")
        ax.set_axis_off()
        return

    components = list(nx.connected_components(G_sep))
    cmap = plt.cm.Set2(np.linspace(0, 1, max(len(components), 1)))

    for ci, comp in enumerate(components):
        color = cmap[ci % len(cmap)][:3]
        sub = G_sep.subgraph(comp)
        for u, v in sub.edges():
            ax.plot([u[1], v[1]], [u[0], v[0]], color=color, linewidth=1.0, alpha=0.8)
        arr = np.array(list(comp))
        ax.scatter(arr[:, 1], arr[:, 0], c=[color], s=6, zorder=5,
                   label=f"Room {ci} [{len(comp)}]", edgecolors="none")

    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_axis_off()


def plot_room_object_graph(ax, occ, inflated, G_sep, rog, objects, title):
    """
    Rooms coloured by component, objects plotted as labelled markers,
    room centres marked with stars.
    """
    rgb = np.ones((*occ.shape, 3)) * 0.95
    rgb[inflated != 0] = [0.90, 0.90, 0.90]
    rgb[occ != 0] = [0.2, 0.2, 0.2]
    ax.imshow(rgb, interpolation="nearest")

    components = list(nx.connected_components(G_sep))
    cmap_rooms = plt.cm.Set2(np.linspace(0, 1, max(len(components), 1)))

    # Draw room Voronoi edges (thin, muted)
    for ci, comp in enumerate(components):
        color = cmap_rooms[ci % len(cmap_rooms)][:3]
        sub = G_sep.subgraph(comp)
        for u, v in sub.edges():
            ax.plot([u[1], v[1]], [u[0], v[0]], color=color,
                    linewidth=0.5, alpha=0.35)

    # Room centre stars
    for rname in get_room_nodes(rog):
        rd = rog.nodes[rname]
        rc, cc = rd["pos_map"]
        room_id = rd["room_id"]
        color = cmap_rooms[room_id % len(cmap_rooms)][:3]
        ax.plot(cc, rc, marker="*", color=color, markersize=14,
                markeredgecolor="k", markeredgewidth=0.5, zorder=7)
        ax.annotate(rname, (cc, rc), fontsize=7, fontweight="bold",
                    color=color, ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points")

    # Objects as labelled dots, coloured by room
    for obj in objects:
        if obj.room_id is None or obj.room_id < 0:
            continue
        color = cmap_rooms[obj.room_id % len(cmap_rooms)][:3]
        r, c = obj.position
        ax.plot(c, r, "o", color=color, markersize=7,
                markeredgecolor="k", markeredgewidth=0.6, zorder=6)
        ax.annotate(obj.name, (c, r), fontsize=6,
                    ha="left", va="top", color="k",
                    xytext=(3, -3), textcoords="offset points")

        # Thin line from object to its closest Voronoi node
        if obj.closest_vor_node is not None:
            vr, vc = obj.closest_vor_node
            ax.plot([c, vc], [r, vr], ":", color=color,
                    linewidth=0.7, alpha=0.5)

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_axis_off()


def plot_voronoi_cells(ax, occ, title):
    """Voronoi tessellation: each wall segment and furniture piece gets its own cell."""
    H, W = occ.shape
    occ_bool = occ != 0

    _, idx = distance_transform_edt(~occ_bool, return_indices=True)
    tessellation = occ[idx[0], idx[1]]
    tessellation[occ_bool] = 0

    n_labels = int(occ.max())
    cmap = plt.cm.tab20(np.linspace(0, 1, max(n_labels, 1)))
    rgb = np.ones((H, W, 3)) * 0.95
    for i in range(1, n_labels + 1):
        rgb[tessellation == i] = cmap[(i - 1) % len(cmap)][:3]

    boundary = np.zeros((H, W), dtype=bool)
    boundary[1:, :] |= tessellation[1:, :] != tessellation[:-1, :]
    boundary[:, 1:] |= tessellation[:, 1:] != tessellation[:, :-1]
    rgb[boundary & ~occ_bool] = [0.35, 0.35, 0.35]

    rgb[occ_bool] = [0.2, 0.2, 0.2]

    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_axis_off()


# ================================= MAIN ====================================

def main():
    resolution = 0.05  # 5 cm per cell

    # ── Step 0: Synthetic occupancy ──
    print("Creating synthetic apartment occupancy...")
    occ = create_apartment_occupancy(shape=(200, 300))
    print(f"  Grid: {occ.shape},  occupied cells: {(occ != 0).sum()}")

    # ── Step 1: Inflation ──
    print("Running inflation...")
    t0 = time.perf_counter()
    inflated = inflate_occupancy(occ, resolution=resolution,
                                 params=InflationParams(radius_m=0.15))
    t_inflate = time.perf_counter() - t0
    print(f"  Inflation: {t_inflate*1000:.1f} ms")

    # ── Step 2: SDF ──
    print("Computing SDF...")
    t0 = time.perf_counter()
    sdf = compute_sdf(inflated, resolution=resolution)
    t_sdf = time.perf_counter() - t0
    print(f"  SDF: {t_sdf*1000:.1f} ms,  range: [{sdf.min():.3f}, {sdf.max():.3f}] m")

    # ── Step 3: Boundary cost field ──
    print("Computing boundary cost field...")
    t0 = time.perf_counter()
    cost = boundary_cost_field(inflated, resolution=resolution, distance_scale_m=1.5)
    t_cost = time.perf_counter() - t0
    print(f"  Cost field: {t_cost*1000:.1f} ms")

    # ── Step 4: Voronoi topology ──
    print("Extracting Voronoi topology graph...")
    t0 = time.perf_counter()
    G = extract_voronoi_graph(inflated, resolution=resolution,
                              params=TopologyParams(sdf_scale_m=1.5,
                                                    inner_inflate_m=0.15,
                                                    sparsify_dist_m=0.3,
                                                    min_component=4))
    t_vor = time.perf_counter() - t0
    print(f"  Voronoi: {t_vor*1000:.1f} ms  |  "
          f"Nodes: {len(G.nodes)},  Edges: {len(G.edges)}")

    # ── Step 5: Room separation ──
    doors = create_apartment_doors()
    print("Separating rooms via door probability...")
    t0 = time.perf_counter()
    G_sep, door_field = separate_rooms(
        G, grid_shape=occ.shape, doors=doors, resolution=resolution,
        params=RoomSeparationParams(edge_score_threshold=0.05, min_component=4),
    )
    t_sep = time.perf_counter() - t0
    room_components = list(nx.connected_components(G_sep))
    print(f"  Room separation: {t_sep*1000:.1f} ms  |  "
          f"Rooms found: {len(room_components)}")

    # ── Step 6: Room–object graph ──
    objects = create_apartment_objects()
    print("Building room–object graph...")
    t0 = time.perf_counter()
    assign_objects_to_rooms(objects, G_sep)
    rog = build_room_object_graph(G_sep, objects)
    t_rog = time.perf_counter() - t0

    total_ms = (t_inflate + t_sdf + t_cost + t_vor + t_sep + t_rog) * 1000
    print(f"  Room–object graph: {t_rog*1000:.1f} ms")
    print(f"  Total pipeline: {total_ms:.1f} ms\n")

    # Print the room–object hierarchy
    print("Room–Object Hierarchy")
    print("=" * 40)
    for rname in sorted(get_room_nodes(rog)):
        rd = rog.nodes[rname]
        obj_names = get_objects_in_room(rog, rd["room_id"])
        print(f"  {rname}  (centre: {rd['pos_map']})")
        if obj_names:
            for oname in obj_names:
                od = rog.nodes[oname]
                print(f"    ├─ {oname:15s}  class={od['semantic_class']:12s}  "
                      f"pos={od['pos_map']}")
        else:
            print(f"    └─ (no objects)")
    print()

    out_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Figure 1: Core pipeline stages 1–6 ──
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle("Topology Pipeline: Occupancy → Inflation → SDF → Voronoi",
                  fontsize=14, fontweight="bold")

    plot_occupancy(axes1[0, 0], occ, "1. Raw Occupancy")
    plot_inflated(axes1[0, 1], occ, inflated, "2. Inflated Occupancy")
    plot_sdf(axes1[0, 2], sdf, "3. Signed Distance Field")
    plot_boundary_cost(axes1[1, 0], cost, "4. Boundary Cost Field")
    plot_voronoi_graph(axes1[1, 1], inflated, G, "5. Voronoi Graph")
    plot_components(axes1[1, 2], occ, inflated, G,
                    "6. Voronoi Graph + Inflation")

    fig1.tight_layout()
    out1 = os.path.join(out_dir, "topology_pipeline_test.png")
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved pipeline plot       → {out1}")

    # ── Figure 2: Room separation + Room–object graph ──
    fig2, axes2 = plt.subplots(1, 3, figsize=(21, 6))
    fig2.suptitle("Room Separation & Object Assignment",
                  fontsize=14, fontweight="bold")

    plot_door_field(axes2[0], occ, door_field, doors,
                    "7. Door Probability Field")
    plot_room_components(axes2[1], occ, inflated, G_sep,
                         "8. Separated Rooms")
    plot_room_object_graph(axes2[2], occ, inflated, G_sep, rog, objects,
                           "9. Room–Object Graph")

    fig2.tight_layout()
    out2 = os.path.join(out_dir, "topology_room_objects.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved room-object plot    → {out2}")

    plt.show()


if __name__ == "__main__":
    main()