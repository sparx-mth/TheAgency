import numpy as np
from typing import List, Tuple, Optional
from sparx_agency.tasks.planning.slam_simulator.constants import TileType

def generate_random_map(
    width: int,
    height: int,
    wall_density: float = 0.15,
    num_entry_points: int = 3
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=np.int8)

    # Border walls
    grid[0, :] = TileType.WALL
    grid[-1, :] = TileType.WALL
    grid[:, 0] = TileType.WALL
    grid[:, -1] = TileType.WALL

    # Random internal walls
    num_walls = int(width * height * wall_density)
    for _ in range(num_walls):
        x = np.random.randint(1, width - 1)
        y = np.random.randint(1, height - 1)
        grid[y, x] = TileType.WALL

    # Entry points
    placed = 0
    attempts = 0
    while placed < num_entry_points and attempts < 1000:
        x = np.random.randint(1, width - 1)
        y = np.random.randint(1, height - 1)
        if grid[y, x] == TileType.FREE_SPACE:
            grid[y, x] = TileType.ENTRY_POINT
            placed += 1
        attempts += 1

    return grid

def load_map(path: str) -> np.ndarray:
    return np.loadtxt(path, dtype=np.int8)

def find_entry_points(grid: np.ndarray) -> List[Tuple[int, int]]:
    points = []
    height, width = grid.shape
    for y in range(height):
        for x in range(width):
            if grid[y, x] == TileType.ENTRY_POINT:
                points.append((x, y))
    return points

def compute_reachable_mask(grid: np.ndarray, start_points: List[Tuple[int, int]]) -> np.ndarray:
    height, width = grid.shape
    reachable = np.zeros((height, width), dtype=bool)
    visited = np.zeros((height, width), dtype=bool)
    queue = list(start_points)

    while queue:
        x, y = queue.pop(0)
        if visited[y, x]:
            continue
        visited[y, x] = True
        reachable[y, x] = True

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                tile = grid[ny, nx]
                if tile not in (TileType.WALL, TileType.DOOR_CLOSED, TileType.OUT_OF_BOUNDS):
                    queue.append((nx, ny))

    # Mark visible walls/doors as reachable (can be observed)
    for y in range(height):
        for x in range(width):
            if grid[y, x] in (TileType.WALL, TileType.DOOR_CLOSED):
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height and reachable[ny, nx]:
                        reachable[y, x] = True
                        break

    return reachable