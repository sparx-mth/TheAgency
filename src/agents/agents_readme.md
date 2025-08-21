# Mapping Agents: Preconditions & Postconditions

This document describes the **7 agents** used for hierarchical house mapping.  
Each agent has defined **Preconditions** (when it can be activated) and **Postconditions** (when it is considered complete).

---

## 1. Wall-Following Agent

**Preconditions:**
- A wall is detected adjacent to the agent (left, right, or front) within sensor range.

**Postconditions:**
- **End of wall**
- **Corner reached** 
- **Gap/Opening detected** 


---

## 2. Enter Doorway / Enter Room Agent

**Preconditions:**
- A doorway or gap in the wall is detected within sensor range.

**Postconditions:**
- The agent passes through the doorway.
- The environment changes into a more open space (room) or another corridor.

---

## 3. Corridor Traversal Agent

**Preconditions:**
- Parallel walls are detected on both sides.
- Corridor width is relatively narrow compared to an open room.

**Postconditions:**
- Reached an intersection (branching point).
- Reached a doorway leading into a room.
- Reached the end of the corridor (wall directly ahead).

---

## 4. Exit Room / Exit Doorway Agent

**Preconditions:**
- The agent is currently inside a bounded room.
- One or more exit doorways are known/detected.

**Postconditions:**
- The agent passes through an exit doorway.
- Current position is now outside the room (corridor or another room).

---

## 5. Room Scanning Agent (without leaving)

**Preconditions:**
- The agent is inside a room (open area vs corridor).

**Postconditions:**
- Room area has been fully scanned (coverage ≥ X%, e.g. 90%).
- The agent remains inside the room, near an exit doorway (ready to continue).

---

## 6. Return-to-Base Agent

**Preconditions:**
- Base location is known (origin or checkpoint).
- Current position is known within the partial map.

**Postconditions:**
- Agent reaches the base location (within tolerance ε).
- System can reset/end task or start a new one.

---

## 7. Go-to-Specific-Point Agent

**Preconditions:**
- A target point exists in the map.
- Path to target is navigable (no known blockages).

**Postconditions:**
- Agent reaches the target point within tolerance ε.
- Current position matches the target location.

---
