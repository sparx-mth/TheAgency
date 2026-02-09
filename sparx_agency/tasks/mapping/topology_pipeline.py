#!/usr/bin/env python3
"""
Visual pipeline test: occupancy → inflation → SDF → Voronoi topology.

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

    # Furniture — Living room
    box(25, 35, 15, 25)     # TV
    box(35, 42, 55, 72)     # coffee table
    box(20, 30, 55, 80)     # sofa

    # Furniture — Bedroom
    box(30, 55, 180, 220)   # bed
    box(30, 38, 225, 233)   # nightstand
    box(10, 35, 260, 275)   # wardrobe

    # Furniture — Bathroom
    box(140, 165, 15, 40)   # bathtub
    box(130, 138, 60, 72)   # sink

    # Furniture — Kitchen
    box(140, 155, 185, 215) # island
    box(120, 130, 245, 290) # counter
    box(165, 175, 175, 195) # small table

    return occ


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


def plot_voronoi_cells(ax, occ, title):
    """Voronoi tessellation: each wall segment and furniture piece gets its own cell."""
    H, W = occ.shape
    occ_bool = occ != 0

    # occ already has unique IDs per object — propagate to free space
    _, idx = distance_transform_edt(~occ_bool, return_indices=True)
    tessellation = occ[idx[0], idx[1]]
    tessellation[occ_bool] = 0

    n_labels = int(occ.max())
    cmap = plt.cm.tab20(np.linspace(0, 1, max(n_labels, 1)))
    rgb = np.ones((H, W, 3)) * 0.95
    for i in range(1, n_labels + 1):
        rgb[tessellation == i] = cmap[(i - 1) % len(cmap)][:3]

    # Cell boundaries
    boundary = np.zeros((H, W), dtype=bool)
    boundary[1:, :] |= tessellation[1:, :] != tessellation[:-1, :]
    boundary[:, 1:] |= tessellation[:, 1:] != tessellation[:, :-1]
    rgb[boundary & ~occ_bool] = [0.35, 0.35, 0.35]

    rgb[occ_bool] = [0.2, 0.2, 0.2]

    ax.imshow(rgb, interpolation="nearest")
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


# ================================= MAIN ====================================

def main():
    resolution = 0.05  # 5 cm per cell

    # Step 0: Synthetic occupancy
    print("Creating synthetic apartment occupancy...")
    occ = create_apartment_occupancy(shape=(200, 300))
    print(f"  Grid: {occ.shape},  occupied cells: {occ.sum()}")

    # Step 1: Inflation
    print("Running inflation...")
    t0 = time.perf_counter()
    inflated = inflate_occupancy(occ, resolution=resolution,
                                 params=InflationParams(radius_m=0.15))
    t_inflate = time.perf_counter() - t0
    print(f"  Inflation: {t_inflate*1000:.1f} ms")

    # Step 2: SDF
    print("Computing SDF...")
    t0 = time.perf_counter()
    sdf = compute_sdf(inflated, resolution=resolution)
    t_sdf = time.perf_counter() - t0
    print(f"  SDF: {t_sdf*1000:.1f} ms,  range: [{sdf.min():.3f}, {sdf.max():.3f}] m")

    # Step 3: Boundary cost field
    print("Computing boundary cost field...")
    t0 = time.perf_counter()
    cost = boundary_cost_field(inflated, resolution=resolution, distance_scale_m=1.5)
    t_cost = time.perf_counter() - t0
    print(f"  Cost field: {t_cost*1000:.1f} ms")

    # Step 4: Voronoi topology
    print("Extracting Voronoi topology graph...")
    t0 = time.perf_counter()
    G = extract_voronoi_graph(inflated, resolution=resolution,
                              params=TopologyParams(sdf_scale_m=1.5,
                                                    inner_inflate_m=0.15,
                                                    sparsify_dist_m=0.3,
                                                    min_component=4))
    t_vor = time.perf_counter() - t0

    junctions = get_junctions(G)
    dead_ends = get_dead_ends(G)
    components = list(nx.connected_components(G))
    print(f"  Voronoi: {t_vor*1000:.1f} ms")
    print(f"  Nodes: {len(G.nodes)},  Edges: {len(G.edges)}")
    print(f"  Junctions: {len(junctions)},  Dead-ends: {len(dead_ends)}")
    print(f"  Connected components: {len(components)}")
    print(f"  Total pipeline: {(t_inflate + t_sdf + t_cost + t_vor)*1000:.1f} ms")

    out_dir = os.path.dirname(os.path.abspath(__file__))

    # Figure 1: Pipeline stages 1–6
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
    print(f"\nSaved pipeline plot to: {out1}")

    # Figure 2: Graph + Tessellation side by side
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle("Voronoi Graph & Tessellation",
                  fontsize=14, fontweight="bold")

    plot_components(axes2[0], occ, inflated, G, "Voronoi Graph + Inflation")
    plot_voronoi_cells(axes2[1], occ, "Voronoi Tessellation")

    fig2.tight_layout()
    out2 = os.path.join(out_dir, "topology_voronoi_comparison.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved comparison plot to: {out2}")

    plt.show()


if __name__ == "__main__":
    main()