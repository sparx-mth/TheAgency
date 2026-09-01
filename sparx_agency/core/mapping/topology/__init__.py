# core/mapping/topology/__init__.py
from .voronoi import extract_voronoi_graph, TopologyParams
from .graph_utils import sparsify_graph, get_junctions, get_dead_ends
from .room_separation import (
    separate_rooms,
    compute_door_probability_field,
    DoorInfo,
    RoomSeparationParams,
)
from .room_object_graph import (
    NodeType,
    ObjectInfo,
    assign_objects_to_rooms,
    build_room_object_graph,
    get_room_nodes,
    get_objects_in_room,
    get_object_node,
)
from .room_segmentation import (
    UNKNOWN,
    FREE_MAX,
    OCC_MIN,
    RoomSegmentationParams,
    RoomStats,
    compute_rooms,
    door_disk_mask,
    heal_free_mask,
)
from .room_merge import merge_basins_by_dynamics
from .room_watershed import WatershedRoomParams, segment_rooms_watershed
from .room_registry import RoomRegistry, TrackedRoom
# SHADOWING: this binds the FUNCTION room_adjacency onto the package, which
# permanently hides the SUBMODULE of the same name -- `import
# sparx_agency.core.mapping.topology.room_adjacency as ra` yields the
# function, not the module. Reach the module only as
# `from sparx_agency.core.mapping.topology.room_adjacency import ...`.
from .room_adjacency import iter_label_borders, room_adjacency
from .room_stats import (
    discover_doors,
    door_room_pairs,
    link_doors,
    count_frontier_clusters,
    room_at_cell,
    room_color,
)
from .llm_client import LLMClient, LLMConfig
from .room_classifier import DEFAULT_LABEL_SET, RoomLabel, RoomTypeClassifier
from .search_oracle import OracleResult, OracleRoom, SearchOracle
from .target_matcher import MatchResult, TargetMatcher, fallback_match
