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
    SUITCASE = 16
    BOX = 17
    SOCKET = 18
    BOTTLE = 19
    MOUSE = 20
    WEAPON = 21
    KEYBOARD = 22
    COMPUTER = 23
    REFRIGERATOR = 24
    PLASTIC_CHAIR = 25
    CARDBOARD_BOX = 26
    UNKNOWN = -1

# Map object names to tile types
OBJECT_TO_TILE = {
    # Seating
    'chair': TileType.CHAIR,
    'plastic chair': TileType.PLASTIC_CHAIR,

    # Tables
    'table': TileType.TABLE,

    # Seating (large)
    'couch': TileType.COUCH,
    'sofa': TileType.COUCH,

    # Display devices
    'tv': TileType.TV,
    'television': TileType.TV,
    'monitor': TileType.TV,

    # Bedroom
    'bed': TileType.BED,

    # Work furniture
    'desk': TileType.DESK,

    # Plants
    'plant': TileType.PLANT,

    # Storage furniture
    'cabinet': TileType.CABINET,

    # Appliances
    'refrigerator': TileType.REFRIGERATOR,
    'microwave': TileType.APPLIANCE,

    # Travel/Storage
    'suitcase': TileType.SUITCASE,

    # Containers
    'box': TileType.BOX,
    'cardboard box': TileType.CARDBOARD_BOX,

    # Computer equipment
    'computer': TileType.COMPUTER,
    'keyboard': TileType.KEYBOARD,
    'mouse': TileType.MOUSE,

    # Electrical
    'socket': TileType.SOCKET,

    # Small items
    'bottle': TileType.BOTTLE,

    # Weapons
    'weapon': TileType.WEAPON,
    'gun': TileType.WEAPON,
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
    TileType.SUITCASE: (139, 90, 43),        # Tan
    TileType.BOX: (205, 133, 63),            # Peru/cardboard
    TileType.SOCKET: (255, 140, 0),          # Dark orange
    TileType.BOTTLE: (135, 206, 250),        # Light sky blue
    TileType.MOUSE: (105, 105, 105),         # Dim gray
    TileType.WEAPON: (128, 0, 0),            # Maroon
    TileType.KEYBOARD: (47, 79, 79),         # Dark slate gray
    TileType.COMPUTER: (70, 130, 180),       # Steel blue
    TileType.REFRIGERATOR: (240, 248, 255),  # Alice blue (white-ish for fridge)
    TileType.PLASTIC_CHAIR: (255, 255, 224), # Light yellow (plastic look)
    TileType.CARDBOARD_BOX: (210, 180, 140), # Tan (lighter cardboard)
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
    TileType.APPLIANCE: (0.6, 0.6, 1.7),    # Generic appliance
    TileType.SUITCASE: (0.7, 0.45, 0.25),   # Large suitcase lying flat
    TileType.BOX: (0.4, 0.4, 0.4),          # Medium box
    TileType.SOCKET: (0.1, 0.05, 0.1),      # Wall socket
    TileType.BOTTLE: (0.08, 0.08, 0.25),    # Water bottle
    TileType.MOUSE: (0.12, 0.08, 0.04),     # Computer mouse
    TileType.WEAPON: (0.3, 0.1, 0.2),       # Handgun size
    TileType.KEYBOARD: (0.45, 0.15, 0.03),  # Standard keyboard
    TileType.COMPUTER: (0.2, 0.4, 0.4),     # Desktop tower
    TileType.REFRIGERATOR: (0.6, 0.65, 1.7), # Standard refrigerator
    TileType.PLASTIC_CHAIR: (0.45, 0.45, 0.85), # Plastic chair (slightly smaller)
    TileType.CARDBOARD_BOX: (0.45, 0.35, 0.35), # Medium cardboard box
}

# Average real-world heights in meters (for distance estimation)
OBJECT_HEIGHTS = {
    # Seating
    'chair': 0.9,              # Back of chair
    'plastic chair': 0.85,     # Plastic chair slightly lower

    # Tables
    'table': 0.75,             # Table surface

    # Seating (large)
    'couch': 0.85,             # Back of couch
    'sofa': 0.85,

    # Display
    'tv': 0.7,                 # TV with stand
    'television': 0.7,
    'monitor': 0.4,            # Desktop monitor

    # Bedroom
    'bed': 0.6,                # Mattress height

    # Work furniture
    'desk': 0.75,              # Desk surface

    # Plants
    'plant': 1.0,              # Medium potted plant

    # Storage furniture
    'cabinet': 1.8,            # Tall cabinet

    # Appliances
    'refrigerator': 0.5,
    'microwave': 0.3,

    # Travel/Storage
    'suitcase': 0.7,           # Large suitcase standing

    # Containers
    'box': 0.4,                # Medium box
    'cardboard box': 0.35,     # Medium cardboard box

    # Computer equipment
    'computer': 0.4,           # Desktop computer tower
    'keyboard': 0.03,          # Keyboard thickness
    'mouse': 0.04,             # Mouse height

    # Electrical
    'socket': 0.1,             # Wall socket

    # Small items
    'bottle': 0.25,            # Water bottle

    # Weapons
    'weapon': 0.2,             # Generic weapon
    'gun': 0.2,                # Handgun

    # Other reference heights
    'person': 1.7,             # Average person height
    'door': 2.0,               # Standard door
}

# Tile type names for display
TILE_NAMES = {
    TileType.FREE_SPACE: "Free Space",
    TileType.WALL: "Wall",
    TileType.ENTRY_POINT: "Entry/Camera",
    TileType.DOOR_CLOSED: "Door (Closed)",
    TileType.DOOR_OPEN: "Door (Open)",
    TileType.WINDOW: "Window",
    TileType.OUT_OF_BOUNDS: "Out of Bounds",
    TileType.CHAIR: "Chair",
    TileType.TABLE: "Table",
    TileType.COUCH: "Couch",
    TileType.TV: "TV",
    TileType.BED: "Bed",
    TileType.DESK: "Desk",
    TileType.PLANT: "Plant",
    TileType.CABINET: "Cabinet",
    TileType.APPLIANCE: "Appliance",
    TileType.SUITCASE: "Suitcase",
    TileType.BOX: "Box",
    TileType.SOCKET: "Socket",
    TileType.BOTTLE: "Bottle",
    TileType.MOUSE: "Mouse",
    TileType.WEAPON: "Weapon",
    TileType.KEYBOARD: "Keyboard",
    TileType.COMPUTER: "Computer",
    TileType.REFRIGERATOR: "Refrigerator",
    TileType.PLASTIC_CHAIR: "Plastic Chair",
    TileType.CARDBOARD_BOX: "Cardboard Box",
    TileType.UNKNOWN: "Unknown",
}

# List of all vocabulary labels supported
OPEN_VOCAB_LABELS = [
    "desk", "cabinet", "tv", "table", "couch", "plant", "bed",
    "suitcase", "table", "socket", "refrigerator", "bottle",
    "mouse", "weapon", "chair", "keyboard", "computer", "box",
    "cardboard box", "gun", "plastic chair"
]