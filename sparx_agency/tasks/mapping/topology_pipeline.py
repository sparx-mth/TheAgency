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

from sparx_agency.core.mapping.costmap.inflation import inflate_occupancy, InflationParams
from sparx_agency.core.mapping.costmap.sdf import compute_sdf, boundary_cost_field
from sparx_agency.core.mapping.topology import (
    extract_voronoi_graph, TopologyParams, get_junctions, get_dead_ends,
)

# ========================= SYNTHETIC APARTMENT =============================

def create_apartment_occupancy(shape=(200, 300)) -> np.ndarray:
    """
    Draw a synthetic apartment: outer walls, 4 rooms, a corridor,
    door openings, and furniture.  Returns binary uint8 grid (0=free, 1=occ).
    """
    H, W = shape
    occ = np.zeros((H, W), dtype=np.uint8)
    wall = 2

    def box(r1, r2, c1, c2):
        occ[r1:r2, c1:c2] = 1

    # Outer walls
    box(0, wall, 0, W)
    box(H - wall, H, 0, W)
    box(0, H, 0, wall)
    box(0, H, W - wall, W)

    # Vertical divider at col 140
    div_c = 140
    box(0, H, div_c, div_c + wall)

    # Corridor band rows 85..115
    corr_top, corr_bot = 85, 115
    box(corr_top, corr_top + wall, 0, div_c)
    box(corr_bot, corr_bot + wall, 0, div_c)
    box(corr_top, corr_top + wall, div_c, W)

    # Door openings (gaps)
    occ[corr_top:corr_top + wall, 45:60] = 0   # living → corridor
    occ[corr_bot:corr_bot + wall, 45:60] = 0   # corridor → bathroom
    occ[40:55, div_c:div_c + wall] = 0          # living → bedroom
    occ[130:145, div_c:div_c + wall] = 0        # kitchen → right side

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
    ax.imshow(occ, cmap="gray_r", interpolation="nearest")
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
    rgb[occ != 0] = [0.15, 0.15, 0.15]
    ax.imshow(rgb, interpolation="nearest")

    components = list(nx.connected_components(G))
    junctions = get_junctions(G)
    dead_ends = get_dead_ends(G)

    if components:
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(components), 2)))
        for ci, comp in enumerate(components):
            c = colors[ci % len(colors)]
            for u, v in G.edges():
                if u in comp:
                    ax.plot([u[1], v[1]], [u[0], v[0]], color=c, linewidth=1.0, alpha=0.8)
            arr = np.array(list(comp))
            ax.scatter(arr[:, 1], arr[:, 0], c=[c], s=5, zorder=4)

    if junctions:
        jarr = np.array(junctions)
        ax.scatter(jarr[:, 1], jarr[:, 0], c="red", s=40, zorder=6,
                   marker="*", label=f"Junctions ({len(junctions)})")
    if dead_ends:
        darr = np.array(dead_ends)
        ax.scatter(darr[:, 1], darr[:, 0], c="lime", s=25, zorder=6,
                   marker="^", label=f"Dead-ends ({len(dead_ends)})")

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

    # Plot all stages
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Topology Pipeline: Occupancy → Inflation → SDF → Voronoi",
                 fontsize=14, fontweight="bold")

    plot_occupancy(axes[0, 0], occ, "1. Raw Occupancy")
    plot_inflated(axes[0, 1], occ, inflated, "2. Inflated Occupancy")
    plot_sdf(axes[0, 2], sdf, "3. Signed Distance Field")
    plot_boundary_cost(axes[1, 0], cost, "4. Boundary Cost Field")
    plot_voronoi_graph(axes[1, 1], inflated, G, "5. Voronoi Graph")
    plot_components(axes[1, 2], occ, inflated, G,
                    f"6. Components ({len(components)}) + Junctions")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "topology_pipeline_test.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()