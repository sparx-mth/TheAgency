# Mapping Agents: Preconditions & Postconditions

This document describes the **4 agents** implemented for hierarchical house mapping.  
Each agent has defined **Preconditions** (when it can be activated) and **Postconditions** (when it is considered complete).

---

## 1. Wall-Following Agent
**File:** `wall_following_agent.py`

**Purpose:** Finds the closest wall, approaches it, and follows along its length in both directions.

**Preconditions:**
- A wall is detected within sensor range 
- Agent is not already following a wall

**Postconditions:**
- Wall has been followed to one end
- 180° turn completed
- Wall has been followed back to the other end
- State reaches DONE

**States:**
- FIND_WALL → APPROACH_WALL → FOLLOW_WALL → TURN_AROUND → FOLLOW_BACK → DONE

---

## 2. Doorway Entry Agent
**File:** `doorway_traversal_agent.py`

**Purpose:** Detects doorways (free spaces with walls on opposite sides) and navigates through them.

**Preconditions:**
- A doorway pattern is detected (free space with walls on opposite sides)
- Doorway has not been previously visited
- Path to doorway is navigable

**Postconditions:**
- Agent has passed through the doorway (position changed after entering)
- Doorway is marked as visited
- State returns to FINDING for next doorway

**States:**
- FINDING → APPROACHING → ENTERING → COMPLETE (cycles back to FINDING)

---

## 3. Room Frontier Agent
**File:** `room_frontier_agent.py`

**Purpose:** Explores room interiors using frontier-based exploration while detecting and avoiding doorways.

**Preconditions:**
- Agent is inside a room or open space
- Unexplored frontiers exist (known cells adjacent to unknown cells)
- Detected doorways are marked as forbidden zones

**Postconditions:**
- All accessible frontiers within the room have been explored
- No more unknown cells adjacent to known free space (within room boundaries)
- Agent never steps on detected doorways

**Key Features:**
- Real-time doorway detection (wall-space-wall patterns)
- Maintains forbidden_doorways set to prevent crossing thresholds
- Plans paths that avoid doorway positions

---

## 4. A* Navigation Agent
**File:** `a_star_navigation_agent.py`

**Purpose:** Navigate to specific goal positions using A* pathfinding.

**Preconditions:**
- A valid goal position is provided
- Current position is known
- Map information is available (treats unknown as passable)

**Postconditions:**
- Agent reaches the goal position (within tolerance)
- Returns STAY action when at goal
- Replans if path becomes blocked

**Key Features:**
- Optimistic planning (assumes unknown cells are passable)
- Dynamic replanning when obstacles discovered
- Falls back to exploration when no goal is set

---

## Agent Hierarchy and Coordination

The agents can be combined in a hierarchical manner:

1. **Room-level exploration**: Room Frontier Agent explores within room boundaries
2. **Inter-room navigation**: Doorway Entry Agent moves between rooms
3. **Perimeter mapping**: Wall-Following Agent traces structural boundaries
4. **Goal-directed navigation**: A* Navigation Agent for specific target locations

Each agent operates with clear activation conditions and completion criteria, allowing for seamless transitions between exploration strategies based on the current mapping context.