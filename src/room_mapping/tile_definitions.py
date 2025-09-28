"""
tile_definitions.py

Simple tile type and color definitions for room mapping.
"""

# Tile type constants
class TileType:
    FREE_SPACE = 0
    WALL = 1
    ENTRY_POINT = 2
    DOOR_CLOSED = 3
    DOOR_OPEN = 4
    WINDOW = 5
    OUT_OF_BOUNDS = 6
    CHAIR = 7
    TABLE = 8
    COUCH = 9
    TV = 10
    BED = 11
    DESK = 12
    PLANT = 13
    CABINET = 14
    APPLIANCE = 15
    UNKNOWN = -1

# Map object names to tile types
OBJECT_TO_TILE = {
    'chair': TileType.CHAIR,
    'table': TileType.TABLE,
    'couch': TileType.COUCH,
    'sofa': TileType.COUCH,
    'tv': TileType.TV,
    'television': TileType.TV,
    'monitor': TileType.TV,
    'bed': TileType.BED,
    'desk': TileType.DESK,
    'plant': TileType.PLANT,
    'cabinet': TileType.CABINET,
    'refrigerator': TileType.APPLIANCE,
    'microwave': TileType.APPLIANCE,
}

# Colors for rendering (RGB)
TILE_COLORS = {
    TileType.FREE_SPACE: (200, 200, 200),    # Light gray
    TileType.WALL: (100, 100, 100),          # Dark gray
    TileType.ENTRY_POINT: (0, 255, 255),     # Cyan
    TileType.DOOR_CLOSED: (255, 0, 0),       # Red
    TileType.DOOR_OPEN: (0, 200, 0),         # Green
    TileType.WINDOW: (0, 0, 255),            # Blue
    TileType.OUT_OF_BOUNDS: (0, 0, 0),       # Black
    TileType.CHAIR: (139, 69, 19),           # Brown
    TileType.TABLE: (101, 67, 33),           # Dark brown
    TileType.COUCH: (128, 0, 128),           # Purple
    TileType.TV: (64, 64, 64),               # Dark gray
    TileType.BED: (255, 182, 193),           # Light pink
    TileType.DESK: (160, 82, 45),            # Sienna
    TileType.PLANT: (0, 128, 0),             # Green
    TileType.CABINET: (92, 51, 23),          # Dark wood
    TileType.APPLIANCE: (192, 192, 192),     # Silver
    TileType.UNKNOWN: (50, 50, 50),          # Very dark gray
}

# Object sizes in meters (width, depth, height)
# Based on standard furniture dimensions
OBJECT_SIZES = {
    TileType.CHAIR: (0.5, 0.5, 0.9),        # Standard dining chair
    TileType.TABLE: (1.2, 0.8, 0.75),       # Dining table
    TileType.COUCH: (2.0, 0.9, 0.85),       # 3-seat couch
    TileType.TV: (1.2, 0.3, 0.7),           # TV on stand
    TileType.BED: (1.5, 2.0, 0.6),          # Queen bed
    TileType.DESK: (1.2, 0.6, 0.75),        # Office desk
    TileType.PLANT: (0.3, 0.3, 1.0),        # Potted plant
    TileType.CABINET: (0.8, 0.4, 1.8),      # Storage cabinet
    TileType.APPLIANCE: (0.6, 0.6, 1.7),    # Refrigerator
}

# Average real-world heights in meters (for distance estimation)
OBJECT_HEIGHTS = {
    'chair': 0.9,      # Back of chair
    'table': 0.75,     # Table surface
    'couch': 0.85,     # Back of couch
    'sofa': 0.85,
    'tv': 0.7,         # TV with stand
    'television': 0.7,
    'monitor': 0.4,    # Desktop monitor
    'bed': 0.6,        # Mattress height
    'desk': 0.75,      # Desk surface
    'plant': 1.0,      # Medium potted plant
    'cabinet': 1.8,    # Tall cabinet
    'refrigerator': 1.7,
    'microwave': 0.3,
    'person': 1.7,     # Average person height
    'door': 2.0,       # Standard door
}