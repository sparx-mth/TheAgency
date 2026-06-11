#!/usr/bin/env python3
"""
Generate 5 synthetic house/office occupancy maps with furniture objects.

Each map is saved as:
  maps/<map_name>/occupancy.npy      – uint8 grid (0=free, nonzero=obstacle ID)
  maps/<map_name>/objects.json       – list of objects with position & bbox (NO room association)
  maps/<map_name>/doors.json         – list of door openings
  maps/<map_name>/metadata.json      – grid shape, resolution, description
  maps/<map_name>/preview.png        – visual preview of the map

Objects are placed on the grid and listed with their (row, col) position and bounding box,
but they are NOT associated with any room. Room assignment is done algorithmically downstream.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class MapBuilder:
    """Incremental builder for an occupancy grid with auto-incrementing IDs."""

    def __init__(self, shape: tuple[int, int]):
        self.shape = shape
        self.occ = np.zeros(shape, dtype=np.uint8)
        self._id = 0
        self.objects: list[dict] = []
        self.doors: list[dict] = []

    def wall(self, r1: int, r2: int, c1: int, c2: int):
        """Draw a wall segment (not tracked as a named object)."""
        self._id += 1
        self.occ[r1:r2, c1:c2] = self._id

    def door(self, r1: int, r2: int, c1: int, c2: int,
             pos_row: float, pos_col: float):
        """Clear a gap in the wall (door opening) and record the door."""
        self.occ[r1:r2, c1:c2] = 0
        self.doors.append({
            "position": [pos_row, pos_col],
            "size": [0.10, 0.40],
        })

    def place(self, name: str, r1: int, r2: int, c1: int, c2: int,
              semantic_class: str):
        """Place a furniture/object block and record it."""
        self._id += 1
        self.occ[r1:r2, c1:c2] = self._id
        center_r = (r1 + r2) / 2.0
        center_c = (c1 + c2) / 2.0
        self.objects.append({
            "name": name,
            "position": [center_r, center_c],
            "bbox": [r1, r2, c1, c2],
            "semantic_class": semantic_class,
        })

    def outer_walls(self, thickness: int = 2):
        """Draw the 4 outer walls."""
        H, W = self.shape
        self.wall(0, thickness, 0, W)            # top
        self.wall(H - thickness, H, 0, W)        # bottom
        self.wall(0, H, 0, thickness)             # left
        self.wall(0, H, W - thickness, W)         # right


# ─────────────────────────────────────────────────────────────────────────────
# Map 1 — Family Apartment  (4 rooms + corridor)
# ─────────────────────────────────────────────────────────────────────────────

def build_map_family_apartment() -> MapBuilder:
    """
    200×300 apartment: living room (top-left), bedroom (top-right),
    bathroom (bottom-left), kitchen (bottom-right), horizontal corridor.
    """
    m = MapBuilder((200, 300))
    w = 2
    m.outer_walls(w)

    # Vertical divider at col 140
    m.wall(0, 200, 140, 142)

    # Corridor rows 85-115
    m.wall(85, 87, 0, 140)        # corridor top-left
    m.wall(115, 117, 0, 140)      # corridor bottom-left
    m.wall(85, 87, 142, 300)      # corridor top-right

    # Door openings (4 doors)
    m.door(85, 87, 45, 60, 86.0, 52.0)        # corridor → living room
    m.door(115, 117, 45, 60, 116.0, 52.0)     # corridor → bathroom
    m.door(40, 55, 140, 142, 47.0, 141.0)     # living → bedroom
    m.door(130, 145, 140, 142, 137.0, 141.0)  # bathroom → kitchen

    # Living room objects
    m.place("tv",           25, 35,  15,  25,  "tv")
    m.place("sofa",         20, 30,  55,  80,  "sofa")
    m.place("coffee_table", 35, 42,  55,  72,  "table")

    # Bedroom objects
    m.place("bed",          30, 55,  180, 220, "bed")
    m.place("nightstand",   30, 38,  225, 233, "nightstand")
    m.place("wardrobe",     10, 35,  260, 275, "wardrobe")

    # Bathroom objects
    m.place("bathtub",      140, 165, 15,  40,  "bathtub")
    m.place("toilet",       130, 138, 60,  72,  "toilet")

    # Kitchen objects
    m.place("kettle",       140, 155, 185, 215, "kettle")
    m.place("oven",         120, 130, 245, 290, "oven")
    m.place("refrigerator", 165, 175, 175, 195, "refrigerator")

    return m


# ─────────────────────────────────────────────────────────────────────────────
# Map 2 — Small Office  (3 offices + reception + meeting room)
# ─────────────────────────────────────────────────────────────────────────────

def build_map_small_office() -> MapBuilder:
    """
    250×350 office floor: reception area (center), 3 private offices (left),
    meeting room (bottom-right), hallway connecting them.
    """
    m = MapBuilder((250, 350))
    w = 2
    m.outer_walls(w)

    # Left vertical wall at col 120 (offices on left)
    m.wall(0, 250, 120, 122)

    # Horizontal walls splitting left side into 3 offices
    m.wall(80, 82, 0, 120)     # between office 1 and 2
    m.wall(165, 167, 0, 120)   # between office 2 and 3

    # Right side: meeting room wall at row 150
    m.wall(150, 152, 122, 350)

    # Door: office 1 → hallway
    m.door(30, 50, 120, 122, 40.0, 121.0)
    # Door: office 2 → hallway
    m.door(110, 130, 120, 122, 120.0, 121.0)
    # Door: office 3 → hallway
    m.door(195, 215, 120, 122, 205.0, 121.0)
    # Door: meeting room
    m.door(150, 152, 200, 220, 151.0, 210.0)

    # Office 1 objects
    m.place("desk_1",          20, 32,  20,  55,  "desk")
    m.place("office_chair_1",  35, 43,  30,  45,  "chair")
    m.place("filing_cabinet",  10, 30,  90, 105,  "cabinet")
    m.place("monitor_1",       22, 28,  25,  35,  "monitor")

    # Office 2 objects
    m.place("desk_2",          100, 112, 20,  55,  "desk")
    m.place("office_chair_2",  115, 123, 30,  45,  "chair")
    m.place("bookshelf",       90,  130, 95, 110,  "bookshelf")
    m.place("printer",         140, 150, 20,  45,  "printer")

    # Office 3 objects
    m.place("desk_3",          185, 197, 20,  55,  "desk")
    m.place("office_chair_3",  200, 208, 30,  45,  "chair")
    m.place("plant_1",         175, 182, 95, 105,  "plant")

    # Reception (top-right)
    m.place("reception_desk",  30,  50,  200, 270, "desk")
    m.place("sofa_reception",  10,  22,  260, 310, "sofa")
    m.place("water_cooler",    70,  82,  300, 315, "water_cooler")

    # Meeting room (bottom-right)
    m.place("conference_table", 175, 195, 180, 280, "table")
    m.place("whiteboard",       160, 200, 325, 335, "whiteboard")
    m.place("projector",        155, 162, 220, 235, "projector")

    return m


# ─────────────────────────────────────────────────────────────────────────────
# Map 3 — Studio Apartment  (open plan + bathroom)
# ─────────────────────────────────────────────────────────────────────────────

def build_map_studio_apartment() -> MapBuilder:
    """
    150×200 studio: one large open room (living/kitchen/sleeping),
    a separate bathroom, and a small storage closet.
    """
    m = MapBuilder((150, 200))
    w = 2
    m.outer_walls(w)

    # Bathroom wall (bottom-right corner): rows 95-150, cols 130-200
    m.wall(95, 97, 130, 200)     # horizontal wall
    m.wall(95, 150, 128, 130)    # vertical wall (short segment)

    # Storage closet wall (top-right corner): rows 0-45, cols 150-200
    m.wall(43, 45, 148, 200)     # horizontal wall
    m.wall(0, 45, 148, 150)      # vertical wall

    # Door: bathroom
    m.door(95, 97, 145, 160, 96.0, 152.0)
    # Door: storage closet
    m.door(43, 45, 165, 180, 44.0, 172.0)

    # Main room objects (open plan — living + kitchen + sleeping)
    m.place("bed",            15,  40,  10,  50,  "bed")
    m.place("nightstand",     15,  25,  55,  65,  "nightstand")
    m.place("desk",           55,  67,  10,  45,  "desk")
    m.place("desk_chair",     70,  78,  20,  35,  "chair")
    m.place("sofa",           50,  62,  80, 120,  "sofa")
    m.place("tv",             40,  48,  90, 105,  "tv")
    m.place("coffee_table",   65,  72,  85, 112,  "table")
    m.place("kitchen_counter", 120, 132, 10,  60, "counter")
    m.place("stove",          120, 130, 65,  85,  "stove")
    m.place("refrigerator",   105, 118, 10,  25,  "refrigerator")
    m.place("dining_table",   100, 112, 80, 115,  "table")

    # Bathroom objects
    m.place("shower",         105, 135, 160, 185, "shower")
    m.place("toilet",         100, 110, 135, 148, "toilet")
    m.place("sink_bathroom",  115, 122, 135, 148, "sink")

    # Storage closet objects
    m.place("vacuum_cleaner", 10,  20,  170, 182, "appliance")
    m.place("storage_shelf",  10,  38,  185, 195, "shelf")

    return m


# ─────────────────────────────────────────────────────────────────────────────
# Map 4 — L-Shaped House  (6 rooms)
# ─────────────────────────────────────────────────────────────────────────────

def build_map_l_shaped_house() -> MapBuilder:
    """
    300×300 L-shaped house: master bedroom, kid's room, living room,
    kitchen, bathroom, garage. The L is formed by the garage extending down-right.
    """
    m = MapBuilder((300, 300))
    w = 2
    m.outer_walls(w)

    # The L shape: cut out top-right block (rows 0-140, cols 180-300)
    # Fill that area as solid (impassable) to create L
    m.wall(0, 140, 178, 300)

    # --- Interior walls ---
    # Vertical divider at col 150 (left wing rooms)
    m.wall(0, 300, 150, 152)
    # Horizontal divider at row 140 (top/bottom on left wing)
    m.wall(138, 140, 0, 150)
    # Right side: horizontal at row 200 (splits living room / garage area)
    m.wall(198, 200, 152, 300)
    # Right side: vertical at col 230 (kitchen vs living)
    m.wall(140, 200, 228, 230)

    # Doors
    m.door(138, 140, 55, 75, 139.0, 65.0)       # master bedroom → kid's room
    m.door(60, 80, 150, 152, 70.0, 151.0)        # master bedroom → living room
    m.door(200, 220, 150, 152, 210.0, 151.0)     # kid's room → garage
    m.door(198, 200, 180, 200, 199.0, 190.0)     # living room → garage
    m.door(138, 140, 152, 300, 139.0, 200.0)     # (gap in L wall for living room access)
    # That last door is actually the opening at row 140 for right side
    # Let me redo: at row 140, cols 152-300, there's no wall since we only walled
    # 0-150. So the opening is natural. Let me add a wall and door for kitchen.
    m.door(198, 200, 260, 280, 199.0, 270.0)     # kitchen → garage

    # Master bedroom (top-left, rows 0-138, cols 0-150)
    m.place("king_bed",        30,  65,  20,  75,  "bed")
    m.place("dresser",         15,  30,  100, 135, "dresser")
    m.place("master_lamp",     30,  38,  80,  90,  "lamp")
    m.place("master_mirror",   70,  100, 120, 135, "mirror")

    # Kid's room (bottom-left, rows 140-300, cols 0-150)
    m.place("kids_bed",        160, 185, 15,  55,  "bed")
    m.place("toy_box",         200, 215, 15,  40,  "toy_box")
    m.place("study_desk",      250, 262, 20,  60,  "desk")
    m.place("kids_bookshelf",  250, 290, 110, 140, "bookshelf")
    m.place("kids_lamp",       160, 168, 60,  70,  "lamp")

    # Living room (right-top of L, rows 140-198, cols 152-228)
    m.place("sofa_living",     155, 168, 160, 200, "sofa")
    m.place("tv_stand",        145, 155, 175, 210, "tv")
    m.place("living_table",    175, 185, 165, 200, "table")

    # Kitchen (right-top of L, rows 140-198, cols 230-300)
    m.place("fridge",          145, 165, 270, 290, "refrigerator")
    m.place("kitchen_sink",    145, 155, 240, 260, "sink")
    m.place("oven_large",      175, 190, 270, 292, "oven")
    m.place("microwave",       165, 172, 240, 258, "microwave")

    # Garage (bottom-right, rows 200-300, cols 152-300)
    m.place("workbench",       220, 235, 160, 210, "workbench")
    m.place("toolbox",         220, 232, 215, 235, "toolbox")
    m.place("bicycle",         270, 290, 250, 275, "bicycle")

    # Bathroom — carved out of kid's room corner (bottom-left, small)
    # Add bathroom walls inside kid's room area
    m.wall(250, 252, 65, 150)   # horizontal wall
    m.wall(252, 300, 63, 65)    # vertical wall
    m.door(250, 252, 80, 95, 251.0, 87.0)   # bathroom door

    m.place("bath_toilet",     265, 275, 75,  90,  "toilet")
    m.place("bath_sink",       258, 266, 100, 115, "sink")
    m.place("bath_shower",     265, 290, 120, 145, "shower")

    return m


# ─────────────────────────────────────────────────────────────────────────────
# Map 5 — Coworking Space  (open area + pods + kitchen + bathroom)
# ─────────────────────────────────────────────────────────────────────────────

def build_map_coworking_space() -> MapBuilder:
    """
    200×400 coworking space: large open hot-desk area, 2 phone pods,
    a lounge, a kitchenette, and restrooms.
    """
    m = MapBuilder((200, 400))
    w = 2
    m.outer_walls(w)

    # Left block: restrooms (rows 0-80, cols 0-70)
    m.wall(0, 80, 68, 70)        # vertical wall separating restrooms from open (restroom height only)
    m.wall(78, 80, 0, 70)        # horizontal wall bottom of restrooms

    # Phone pods: two small rooms top-right corner
    # Pod 1: rows 0-50, cols 310-400
    m.wall(0, 100, 308, 310)     # vertical wall left of pods (pods only, not full height)
    m.wall(48, 50, 310, 400)     # horizontal wall bottom of pod 1
    # Pod 2: rows 50-100, cols 310-400
    m.wall(98, 100, 310, 400)    # horizontal wall bottom of pod 2

    # Kitchenette: bottom-left, rows 130-200, cols 0-120
    m.wall(128, 130, 0, 120)     # horizontal wall top of kitchen
    m.wall(128, 200, 118, 120)   # vertical wall right of kitchen

    # Lounge: bottom-right, rows 130-200, cols 280-400
    m.wall(128, 130, 280, 400)   # top wall of lounge
    m.wall(128, 200, 278, 280)   # left wall of lounge

    # Doors
    m.door(78, 80, 30, 50, 79.0, 40.0)          # restroom → open area
    m.door(15, 35, 308, 310, 25.0, 309.0)        # open area → pod corridor
    m.door(48, 50, 340, 360, 49.0, 350.0)        # pod 1 → pod corridor
    m.door(98, 100, 340, 360, 99.0, 350.0)       # pod 2 → pod corridor
    m.door(128, 130, 50, 70, 129.0, 60.0)        # kitchenette → open area
    m.door(128, 130, 330, 350, 129.0, 340.0)     # lounge → open area

    # Restroom objects
    m.place("restroom_toilet_1", 15, 28, 10,  28,  "toilet")
    m.place("restroom_toilet_2", 40, 53, 10,  28,  "toilet")
    m.place("restroom_sink",     20, 30, 42,  58,  "sink")
    m.place("hand_dryer",        55, 65, 45,  55,  "dryer")

    # Phone pod 1 objects
    m.place("pod1_desk",         10, 25, 325, 365, "desk")
    m.place("pod1_chair",        28, 38, 335, 355, "chair")
    m.place("pod1_monitor",      12, 20, 335, 355, "monitor")

    # Phone pod 2 objects
    m.place("pod2_desk",         58, 73, 325, 365, "desk")
    m.place("pod2_chair",        76, 86, 335, 355, "chair")

    # Open hot-desk area (center, large)
    m.place("hotdesk_1",         20,  32,  100, 140, "desk")
    m.place("hotdesk_2",         20,  32,  160, 200, "desk")
    m.place("hotdesk_3",         20,  32,  220, 260, "desk")
    m.place("hotdesk_chair_1",   35,  43,  110, 130, "chair")
    m.place("hotdesk_chair_2",   35,  43,  170, 190, "chair")
    m.place("hotdesk_chair_3",   35,  43,  230, 250, "chair")
    m.place("standing_desk",     85,  95,  150, 190, "desk")
    m.place("shared_printer",    110, 122, 240, 265, "printer")
    m.place("plant_lobby",       60,  70,  90,  100, "plant")

    # Kitchenette objects
    m.place("coffee_machine",    140, 150, 10,  30,  "coffee_machine")
    m.place("microwave_kitchen", 140, 150, 40,  60,  "microwave")
    m.place("mini_fridge",       160, 175, 10,  28,  "refrigerator")
    m.place("kitchen_sink_cw",   140, 150, 75, 100,  "sink")
    m.place("snack_shelf",       175, 192, 80, 105,  "shelf")

    # Lounge objects
    m.place("couch_1",           145, 160, 300, 345, "sofa")
    m.place("couch_2",           170, 185, 300, 345, "sofa")
    m.place("bean_bag",          145, 158, 360, 380, "bean_bag")
    m.place("lounge_tv",         135, 145, 340, 370, "tv")
    m.place("lounge_table",      160, 170, 350, 385, "table")

    return m


# ─────────────────────────────────────────────────────────────────────────────
# Saving & Visualization
# ─────────────────────────────────────────────────────────────────────────────

def save_map(name: str, description: str, builder: MapBuilder,
             resolution: float = 0.05):
    """Save a map to disk: occupancy.npy, objects.json, doors.json, metadata.json, preview.png."""
    out_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    occ = builder.occ
    objects = builder.objects
    doors = builder.doors

    # 1. Occupancy grid
    np.save(os.path.join(out_dir, "occupancy.npy"), occ)

    # 2. Objects (position + bbox, NO room association)
    with open(os.path.join(out_dir, "objects.json"), "w") as f:
        json.dump(objects, f, indent=2)

    # 3. Doors
    with open(os.path.join(out_dir, "doors.json"), "w") as f:
        json.dump(doors, f, indent=2)

    # 4. Metadata
    meta = {
        "name": name,
        "description": description,
        "grid_shape": list(occ.shape),
        "resolution_m": resolution,
        "num_objects": len(objects),
        "num_doors": len(doors),
        "occupied_cells": int((occ != 0).sum()),
        "object_names": [o["name"] for o in objects],
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 5. Preview image
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"{name}: {description}", fontsize=13, fontweight="bold")

    # Panel 1: raw occupancy (walls + objects as black)
    axes[0].imshow(occ != 0, cmap="gray_r", interpolation="nearest")
    axes[0].set_title("Occupancy Grid", fontsize=10)
    axes[0].set_axis_off()

    # Panel 2: occupancy + object labels + door markers
    rgb = np.ones((*occ.shape, 3)) * 0.95
    rgb[occ != 0] = [0.2, 0.2, 0.2]
    axes[1].imshow(rgb, interpolation="nearest")

    for obj in objects:
        r, c = obj["position"]
        r1, r2, c1, c2 = obj["bbox"]
        rect = Rectangle((c1, r1), c2 - c1, r2 - r1,
                          linewidth=1.2, edgecolor="tab:blue",
                          facecolor="tab:blue", alpha=0.3)
        axes[1].add_patch(rect)
        axes[1].annotate(obj["name"], (c, r), fontsize=5, ha="center",
                         va="center", color="white", fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.15",
                                   fc="tab:blue", alpha=0.7, lw=0))

    for door in doors:
        dr, dc = door["position"]
        axes[1].plot(dc, dr, "c*", markersize=10, zorder=6)
        axes[1].annotate("door", (dc, dr), fontsize=5, color="cyan",
                         xytext=(3, -6), textcoords="offset points")

    axes[1].set_title("Objects & Doors (no room labels)", fontsize=10)
    axes[1].set_axis_off()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "preview.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  ✓ Saved '{name}' → {out_dir}")
    print(f"    Grid: {occ.shape}, Objects: {len(objects)}, Doors: {len(doors)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

MAP_REGISTRY = [
    ("family_apartment",
     "4-room apartment: living room, bedroom, bathroom, kitchen + corridor",
     build_map_family_apartment),
    ("small_office",
     "Office floor: 3 private offices, reception area, meeting room",
     build_map_small_office),
    ("studio_apartment",
     "Open-plan studio with bathroom and storage closet",
     build_map_studio_apartment),
    ("l_shaped_house",
     "L-shaped 6-room house: master bedroom, kid's room, living, kitchen, bathroom, garage",
     build_map_l_shaped_house),
    ("coworking_space",
     "Coworking: open hot-desk area, 2 phone pods, kitchenette, lounge, restrooms",
     build_map_coworking_space),
]


def main():
    print("=" * 60)
    print("GENERATING 5 SYNTHETIC MAPS")
    print("=" * 60)

    for name, description, builder_fn in MAP_REGISTRY:
        print(f"\n[{name}]")
        builder = builder_fn()
        save_map(name, description, builder)

    print(f"\nAll maps saved to: {OUTPUT_DIR}")
    print("\nSaved files per map:")
    print("  occupancy.npy   – uint8 occupancy grid (0=free, >0=obstacle ID)")
    print("  objects.json    – object list with position & bbox (NO room IDs)")
    print("  doors.json      – door positions")
    print("  metadata.json   – grid shape, resolution, summary")
    print("  preview.png     – visual preview")


if __name__ == "__main__":
    main()