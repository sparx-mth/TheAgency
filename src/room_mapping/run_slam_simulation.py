#!/usr/bin/env python3
"""
run_slam_with_mission.py - Run SLAM simulation with LLM mission execution

This script:
1. Loads the house map from pixel_room_mapper
2. Creates the Mission Executor agent
3. Monitors for new missions from the LLM
4. Executes missions autonomously
"""

import sys
import os
import json
import numpy as np
import pygame
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import SLAM environment components
from src.environments.base.slam_env import MultiAgentSLAMEnv
from src.environments.base.constants import Action, TILE_SIZE, FPS, DIRECTION_DELTAS, TileType

# Import the mission executor
from mission_executor_agent import MissionExecutorAgent


class DynamicTileManager:
    """Manages dynamic tile types from unified_rooms.json"""

    def __init__(self, tile_registry: Dict[str, int]):
        self.tile_registry = tile_registry
        self.id_to_name = {v: k for k, v in tile_registry.items()}

        # Map dynamic tiles to SLAM TileType enum
        self.tile_to_slam_type = {}
        for name, tile_id in tile_registry.items():
            name_lower = name.lower()
            if 'wall' in name_lower:
                self.tile_to_slam_type[tile_id] = TileType.WALL
            elif 'door' in name_lower:
                self.tile_to_slam_type[tile_id] = TileType.DOOR_OPEN
            elif 'entry' in name_lower:
                self.tile_to_slam_type[tile_id] = TileType.ENTRY_POINT
            elif name_lower == 'free_space':
                self.tile_to_slam_type[tile_id] = TileType.FREE_SPACE
            elif name_lower == 'camera':
                self.tile_to_slam_type[tile_id] = TileType.FREE_SPACE  # Camera positions are free
            else:
                # Most furniture/objects are walls for collision
                self.tile_to_slam_type[tile_id] = TileType.WALL

        # Define colors for visualization
        self.tile_colors = {
            0: (240, 240, 240),  # free_space - light gray
            1: (50, 50, 50),  # wall - dark gray
            2: (0, 255, 0),  # camera - green
            3: (139, 69, 19),  # door - brown
            4: (139, 90, 43),  # table - tan
            5: (165, 42, 42),  # chair - brown-red
            6: (100, 100, 200),  # table and monitor - blue-gray
            7: (75, 75, 150),  # tv - dark blue
            8: (255, 255, 0),  # entry point - yellow
            9: (128, 0, 128),  # bicycle - purple
            10: (80, 80, 80),  # black suitcase - dark gray
            11: (0, 128, 255),  # bottle - light blue
            12: (255, 128, 0),  # mug - orange
            13: (100, 100, 200),  # monitor - blue
            14: (64, 64, 64),  # keyboard - gray
            15: (150, 50, 150),  # bicycle and chair - purple-red
            16: (80, 80, 160),  # keyboard and monitor - gray-blue
            -1: (20, 20, 20),  # unknown - very dark gray
        }

    def get_color(self, tile_id: int) -> Tuple[int, int, int]:
        return self.tile_colors.get(tile_id, (150, 150, 150))

    def to_slam_type(self, tile_id: int) -> int:
        return self.tile_to_slam_type.get(tile_id, TileType.WALL)


class MissionControlledSLAMEnv(MultiAgentSLAMEnv):
    """SLAM environment with mission executor agent and enhanced visualization"""

    def __init__(self, tile_manager: DynamicTileManager, spawn_pos: Tuple[int, int],
                 original_map: np.ndarray, *args, **kwargs):
        self.tile_manager = tile_manager
        self.spawn_pos = spawn_pos
        self.original_map = original_map

        # Mission executor agent
        self.mission_agent = MissionExecutorAgent(num_agents=1)

        # Mission monitoring
        self.last_mission_check = 0
        self.check_interval = 2.0  # Check for new missions every 2 seconds

        super().__init__(*args, **kwargs)

    def reset(self, seed=None, options=None):
        """Override reset to spawn drone at specific position"""
        obs, info = super().reset(seed=seed, options=options)

        # Override drone position
        if len(self.drones) > 0:
            self.drones[0].pos = self.spawn_pos
            self.drones[0].path_history = [self.spawn_pos]
            print(f"Spawned drone at position: {self.spawn_pos}")

            # Reset mission agent
            self.mission_agent.reset()

            # Update observations
            obs = self._get_observations()

        return obs, info

    def get_mission_action(self, observations: Dict, info: Dict) -> np.ndarray:
        """Get action from mission executor agent"""
        current_time = time.time()

        # Check for new missions periodically
        if current_time - self.last_mission_check > self.check_interval:
            if os.path.exists("agent_commands.txt"):
                print("Checking for new mission...")
            self.last_mission_check = current_time

        # Get action from mission executor
        return self.mission_agent.get_actions(observations, info)

    def render(self):
        """Enhanced render with mission status"""
        if self.render_mode is None:
            return None

        # Initialize pygame if needed
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.font = pygame.font.SysFont("Arial", 12)
            self.info_font = pygame.font.SysFont("Arial", 10)

            # Create larger screen for mission info
            screen_width = self.width * TILE_SIZE * 2 + 50
            screen_height = self.height * TILE_SIZE + 200
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption("SLAM Mission Executor")
            self.clock = pygame.time.Clock()

        # Clear screen
        self.screen.fill((30, 30, 30))

        # Draw original map (left)
        for y in range(self.height):
            for x in range(self.width):
                tile = self.original_map[y, x]
                color = self.tile_manager.get_color(tile)
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1)
                pygame.draw.rect(self.screen, color, rect)

        # Draw observed map (right)
        offset_x = self.width * TILE_SIZE + 50
        for y in range(self.height):
            for x in range(self.width):
                if self.global_map[y, x] == TileType.UNKNOWN:
                    color = self.tile_manager.get_color(-1)
                else:
                    tile = self.original_map[y, x]
                    color = self.tile_manager.get_color(tile)

                rect = pygame.Rect(offset_x + x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1)
                pygame.draw.rect(self.screen, color, rect)

        # Draw goal position if it exists
        metrics = self.mission_agent.get_metrics()
        # Use final_goal if available, otherwise use navigation agent's goal
        goal = None
        if hasattr(self.mission_agent, 'final_goal') and self.mission_agent.final_goal:
            goal = self.mission_agent.final_goal
        elif hasattr(self.mission_agent.navigation_agent, 'goal') and self.mission_agent.navigation_agent.goal:
            goal = self.mission_agent.navigation_agent.goal

        if goal:
            goal_color = (0, 255, 255)  # Cyan

            # Draw on both maps
            for offset in [0, offset_x]:
                # Draw goal as a star or target
                goal_x = offset + goal[0] * TILE_SIZE + TILE_SIZE // 2
                goal_y = goal[1] * TILE_SIZE + TILE_SIZE // 2

                # Draw crosshair pattern for goal
                pygame.draw.line(self.screen, goal_color,
                                 (goal_x - 8, goal_y), (goal_x + 8, goal_y), 2)
                pygame.draw.line(self.screen, goal_color,
                                 (goal_x, goal_y - 8), (goal_x, goal_y + 8), 2)
                pygame.draw.circle(self.screen, goal_color, (goal_x, goal_y), 10, 2)

                # Draw a smaller filled circle in center
                pygame.draw.circle(self.screen, goal_color, (goal_x, goal_y), 3)

        # Draw drone
        if len(self.drones) > 0:
            drone = self.drones[0]
            if drone.active:
                drone_color = (255, 0, 0)  # Red

                # Draw on both maps
                for offset in [0, offset_x]:
                    center_x = offset + drone.pos[0] * TILE_SIZE + TILE_SIZE // 2
                    center_y = drone.pos[1] * TILE_SIZE + TILE_SIZE // 2
                    pygame.draw.circle(self.screen, drone_color, (center_x, center_y), 6)

                    # Draw facing direction
                    dx, dy = DIRECTION_DELTAS[drone.facing]
                    end_x = center_x + dx * TILE_SIZE // 3
                    end_y = center_y + dy * TILE_SIZE // 3
                    pygame.draw.line(self.screen, (255, 255, 0), (center_x, center_y), (end_x, end_y), 2)

                    # Draw line from drone to goal if goal exists
                    goal = None
                    if hasattr(self.mission_agent, 'final_goal') and self.mission_agent.final_goal:
                        goal = self.mission_agent.final_goal
                    elif hasattr(self.mission_agent.navigation_agent,
                                 'goal') and self.mission_agent.navigation_agent.goal:
                        goal = self.mission_agent.navigation_agent.goal

                    if goal:
                        goal_x = offset + goal[0] * TILE_SIZE + TILE_SIZE // 2
                        goal_y = goal[1] * TILE_SIZE + TILE_SIZE // 2
                        # Draw a faint line connecting drone to goal
                        pygame.draw.line(self.screen, (100, 100, 100),
                                         (center_x, center_y), (goal_x, goal_y), 1)

        # Draw labels
        true_label = self.font.render("True Map", True, (255, 255, 255))
        obs_label = self.font.render("Observed Map", True, (255, 255, 255))
        self.screen.blit(true_label, (10, self.height * TILE_SIZE + 10))
        self.screen.blit(obs_label, (offset_x + 10, self.height * TILE_SIZE + 10))

        # Draw mission status
        y_offset = self.height * TILE_SIZE + 35

        # Get mission metrics
        metrics = self.mission_agent.get_metrics()

        # Get current goal if exists
        goal_str = "None"
        if hasattr(self.mission_agent, 'final_goal') and self.mission_agent.final_goal:
            goal = self.mission_agent.final_goal
            goal_str = f"({goal[0]}, {goal[1]})"
        elif hasattr(self.mission_agent.navigation_agent, 'goal') and self.mission_agent.navigation_agent.goal:
            goal = self.mission_agent.navigation_agent.goal
            goal_str = f"({goal[0]}, {goal[1]})"

        # Mission information
        mission_info = [
            f"Mission Status: {'LOADED' if metrics.get('mission_loaded') else 'NO MISSION'}",
            f"Steps: {metrics.get('completed_steps', 0)}/{metrics.get('total_steps', 0)}",
            f"Active Agent: {metrics.get('active_agent', 'None')}",
            f"Current Action: {metrics.get('current_action', 'None')[:50] if metrics.get('current_action') else 'None'}",
            "",
            f"Drone Position: {self.drones[0].pos if self.drones else 'N/A'}",
            f"Goal Position: {goal_str} (shown as cyan crosshair)",
            f"Facing: {self.drones[0].facing if self.drones else 'N/A'}",
            f"Step: {self.current_step}/{self.max_steps}",
        ]

        for i, text in enumerate(mission_info):
            if text:  # Skip empty lines
                rendered_text = self.info_font.render(text, True, (200, 200, 200))
                self.screen.blit(rendered_text, (10, y_offset + i * 15))

        # Draw progress bar
        discovered = np.sum((self.global_map != -1) & self.reachable_mask)
        progress = discovered / self.total_reachable if self.total_reachable > 0 else 0.0

        bar_y = y_offset + len(mission_info) * 15 + 10
        bar_width = self.width * TILE_SIZE * 2 + 30
        bar_height = 20

        pygame.draw.rect(self.screen, (60, 60, 60), (10, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, (0, 200, 0), (10, bar_y, int(bar_width * progress), bar_height))

        progress_text = self.font.render(
            f"Exploration: {progress * 100:.1f}% ({discovered}/{self.total_reachable})",
            True, (255, 255, 255)
        )
        self.screen.blit(progress_text, (10, bar_y + 25))

        # Update display
        pygame.display.flip()
        self.clock.tick(FPS)

        return None


def load_house_data():
    """Load house map and tile registry"""
    with open("unified_rooms.json", 'r') as f:
        data = json.load(f)
        tile_registry = data.get("tile_registry", {})

    house_map = np.loadtxt("house_map.txt", dtype=np.int8)

    return house_map, tile_registry


def main():
    """Run SLAM simulation with mission executor"""
    print("=" * 60)
    print("SLAM MISSION EXECUTOR")
    print("=" * 60)

    # Check required files
    if not os.path.exists("unified_rooms.json"):
        print("ERROR: unified_rooms.json not found!")
        return

    if not os.path.exists("house_map.txt"):
        print("ERROR: house_map.txt not found!")
        return

    # Load house data
    print("Loading house data...")
    house_map, tile_registry = load_house_data()
    print(f"Map size: {house_map.shape}")
    print(f"Tile types: {len(tile_registry)}")

    # Create tile manager
    tile_manager = DynamicTileManager(tile_registry)

    # Convert to SLAM-compatible map
    slam_map = np.zeros_like(house_map)
    for y in range(house_map.shape[0]):
        for x in range(house_map.shape[1]):
            tile_id = house_map[y, x]
            slam_map[y, x] = tile_manager.to_slam_type(tile_id)

    # Save temporary SLAM map
    temp_map_file = "temp_slam_map.txt"
    np.savetxt(temp_map_file, slam_map, fmt='%d')

    # Spawn position
    spawn_pos = (27, 34)

    # Verify spawn position
    if house_map[spawn_pos[1], spawn_pos[0]] not in [0, 8, 2]:  # free_space, entry_point, camera
        print(f"WARNING: Spawn position {spawn_pos} is not free! Finding nearby free space...")
        found = False
        for radius in range(1, 5):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = spawn_pos[1] + dy, spawn_pos[0] + dx
                    if 0 <= ny < house_map.shape[0] and 0 <= nx < house_map.shape[1]:
                        if house_map[ny, nx] in [0, 8, 2]:
                            spawn_pos = (nx, ny)
                            print(f"New spawn position: {spawn_pos}")
                            found = True
                            break
                if found:
                    break
            if found:
                break

    print(f"Drone spawning at: {spawn_pos}")

    # Create environment
    env = MissionControlledSLAMEnv(
        tile_manager=tile_manager,
        spawn_pos=spawn_pos,
        original_map=house_map,
        map_path=temp_map_file,
        num_agents=1,
        max_steps=5000,
        render_mode='human',
        default_sensor_params={
            'max_range': 5,
            'fov_deg': 90
        }
    )

    # Reset environment
    obs, info = env.reset()

    print("\n" + "=" * 60)
    print("MISSION EXECUTOR STARTED!")
    print("=" * 60)
    print("\nMonitoring for missions from agent_commands.txt")
    print("Use the web interface to send missions, or")
    print("manually create agent_commands.txt with numbered steps:")
    print("  1. Activate NavigationAgent to navigate to hallway")
    print("  2. Activate ScanAgent to scan the room")
    print("  etc.")
    print("\nPress Q to quit")
    print("=" * 60)

    # Initialize pygame
    pygame.init()
    pygame.display.init()
    pygame.font.init()

    # Force initial render
    env.render()

    # Main loop
    running = True
    terminated = False
    truncated = False
    step_count = 0

    while running:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

        if not terminated and not truncated:
            # Get action from mission executor
            action = env.get_mission_action(obs, info)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            # Print updates every 10 steps
            if step_count % 10 == 0:
                metrics = env.mission_agent.get_metrics()
                if metrics.get('mission_loaded'):
                    print(
                        f"Step {step_count}: Executing step {metrics.get('completed_steps', 0) + 1}/{metrics.get('total_steps', 0)}")

        # Render
        env.render()

        # Check termination
        if terminated:
            print("\n" + "=" * 40)
            print("EXPLORATION COMPLETE!")
            print(f"Steps taken: {step_count}")
            print("=" * 40)
            terminated = False  # Continue running

        elif truncated:
            print("\n" + "=" * 40)
            print("MAX STEPS REACHED!")
            print("=" * 40)
            truncated = False  # Continue running

    # Cleanup
    env.close()

    # Remove temp file
    if os.path.exists(temp_map_file):
        os.remove(temp_map_file)

    print("\nSimulation ended.")


if __name__ == "__main__":
    main()