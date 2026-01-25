# Multi-Agent SLAM Simulator

A Gymnasium-compatible environment for training RL agents on Simultaneous Localization and Mapping (SLAM) tasks. Multiple drones explore an unknown grid map, discovering tiles while avoiding collisions.

## Installation

```bash
pip install gymnasium numpy pygame
```

## Quick Start

```python
from slam_simulator import SLAMEnv

env = SLAMEnv(width=20, height=20, num_agents=2, render_mode='human')
obs, info = env.reset(seed=42)

for _ in range(500):
    actions = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(actions)
    env.render()
    if terminated or truncated:
        break

env.close()
```

## Architecture

```
slam_simulator/
├── __init__.py          # Package exports
├── constants.py         # Enums, colors, default configs
├── drone.py             # Drone state management
├── env.py               # Main Gymnasium environment
├── map_generator.py     # Random map generation & utilities
├── renderer.py          # Pygame visualization
├── sensors/
│   ├── __init__.py
│   ├── base.py          # Abstract sensor interface
│   └── camera.py        # Camera sensor (FOV-based raycasting)
└── wrappers/
    ├── __init__.py
    ├── discrete_action.py  # MultiDiscrete → Discrete conversion
    └── curriculum.py       # Progressive difficulty wrapper
```

## Environment Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width` | int | 32 | Grid width (ignored if `map_path` set) |
| `height` | int | 32 | Grid height (ignored if `map_path` set) |
| `num_agents` | int | 3 | Number of drones |
| `max_steps` | int | 1000 | Episode step limit |
| `map_path` | str | None | Path to load predefined map |
| `randomize` | bool | True | Generate random maps each reset |
| `render_mode` | str | None | `'human'` or `'rgb_array'` |
| `sensor_config` | dict | None | Per-drone sensor overrides |
| `rewards` | dict | None | Custom reward values |

## Observation Space

```python
{
    'global_map': Box(-1, 6, (height, width), int8),  # Shared discovered map
    'positions': Box(0, max_dim, (num_agents, 2), int32),  # (x, y) per drone
    'facings': Box(0, 3, (num_agents,), int32),  # Direction index per drone
    'active': Box(0, 1, (num_agents,), int8),  # Active status per drone
}
```

## Action Space

`MultiDiscrete([4] * num_agents)` — Each drone has 4 actions:

| Action | Value | Description |
|--------|-------|-------------|
| FORWARD | 0 | Move one cell forward |
| TURN_LEFT | 1 | Rotate 90° counter-clockwise |
| TURN_RIGHT | 2 | Rotate 90° clockwise |
| STAY | 3 | Do nothing |

## Tile Types

| Type | Value | Description |
|------|-------|-------------|
| UNKNOWN | -1 | Unexplored |
| FREE_SPACE | 0 | Walkable |
| WALL | 1 | Blocks movement & vision |
| ENTRY_POINT | 2 | Drone spawn location |
| DOOR_CLOSED | 3 | Blocks movement & vision |
| DOOR_OPEN | 4 | Walkable |
| WINDOW | 5 | Blocks movement, allows vision |
| OUT_OF_BOUNDS | 6 | Invalid area |

## Rewards

| Event | Default | Description |
|-------|---------|-------------|
| `discovery` | +0.1 | Per newly discovered cell |
| `collision` | -1.0 | Hitting wall/drone |
| `step` | -0.001 | Per timestep (encourages efficiency) |
| `completion` | +10.0 | Bonus for full map discovery |

Custom rewards:
```python
env = SLAMEnv(rewards={'discovery': 0.5, 'collision': -2.0, 'step': 0, 'completion': 50.0})
```

## Sensors

Default: `CameraSensor(max_range=10, fov_deg=90, num_rays=30)`

Custom per-drone sensors:
```python
from slam_simulator.sensors import CameraSensor

env = SLAMEnv(sensor_config={
    0: CameraSensor(max_range=15, fov_deg=120),
    1: CameraSensor(max_range=5, fov_deg=360),
})
```

## Info Dictionary

Returned by `step()` and `reset()`:

```python
{
    'step': int,           # Current timestep
    'progress': float,     # Discovery progress (0.0 - 1.0)
    'discovered_cells': int,
    'total_reachable': int,
    'collisions': [int],   # Per-drone collision counts
}
```

## Episode Termination

- **Terminated**: All reachable cells discovered
- **Truncated**: `max_steps` reached

## Custom Maps

Save as space-separated integers:
```
1 1 1 1 1
1 0 0 0 1
1 0 2 0 1
1 0 0 0 1
1 1 1 1 1
```

Load:
```python
env = SLAMEnv(map_path='my_map.txt', randomize=False)
```

## Wrappers

### DiscreteActionWrapper

Converts `MultiDiscrete([4,4,4])` to `Discrete(64)` for DQN compatibility.

```python
from slam_simulator import SLAMEnv, DiscreteActionWrapper

env = SLAMEnv(num_agents=2)  # MultiDiscrete([4, 4])
env = DiscreteActionWrapper(env)  # Discrete(16)

action = env.action_space.sample()  # Single int 0-15
obs, reward, done, trunc, info = env.step(action)

# Decode action to see per-agent actions
print(env.decode(action))  # e.g., [0, 2] = [FORWARD, TURN_RIGHT]
```

### CurriculumWrapper

Progressive learning: reveals most of the map, keeps a small square hidden.

```python
from slam_simulator import SLAMEnv, CurriculumWrapper

env = SLAMEnv(width=32, height=32, num_agents=1)
env = CurriculumWrapper(env, hidden_size=8, random_position=True)

# Start with 8x8 hidden area
obs, info = env.reset()
print(info['curriculum'])  # {'hidden_size': 8, 'hidden_position': (x, y), ...}

# Increase difficulty over time
env.set_hidden_size(16)  # Now 16x16 hidden
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hidden_size` | int | 8 | Size of hidden square |
| `random_position` | bool | False | Randomize hidden area position |
| `fixed_position` | tuple | None | Fixed (x, y) for hidden area |
