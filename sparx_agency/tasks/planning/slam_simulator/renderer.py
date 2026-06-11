import numpy as np
from typing import Optional, List
from sparx_agency.tasks.planning.slam_simulator.constants import TILE_SIZE, FPS, TILE_COLORS, DRONE_COLORS, DIRECTION_DELTAS
from sparx_agency.tasks.planning.slam_simulator.drone import Drone

class Renderer:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.screen = None
        self.clock = None
        self.font = None

    def _init_pygame(self):
        import pygame
        pygame.init()
        pygame.display.init()
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 12)
        screen_w = self.width * TILE_SIZE * 2 + 50
        screen_h = self.height * TILE_SIZE + 100
        self.screen = pygame.display.set_mode((screen_w, screen_h))
        pygame.display.set_caption("SLAM Environment")
        self.clock = pygame.time.Clock()

    def render(
        self,
        true_map: np.ndarray,
        global_map: np.ndarray,
        drones: List[Drone],
        progress: float,
        step: int,
        max_steps: int,
        mode: str = 'human'
    ) -> Optional[np.ndarray]:
        import pygame

        if self.screen is None:
            self._init_pygame()

        self.screen.fill((30, 30, 30))
        offset_x = self.width * TILE_SIZE + 50

        # Draw maps
        for y in range(self.height):
            for x in range(self.width):
                # True map
                color = TILE_COLORS.get(int(true_map[y, x]), (150, 150, 150))
                rect = pygame.Rect(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1)
                pygame.draw.rect(self.screen, color, rect)

                # Global map
                color = TILE_COLORS.get(int(global_map[y, x]), (150, 150, 150))
                rect = pygame.Rect(offset_x + x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1)
                pygame.draw.rect(self.screen, color, rect)

        # Draw drones
        for drone in drones:
            if not drone.active:
                continue
            color = DRONE_COLORS[drone.drone_id % len(DRONE_COLORS)]
            for off in [0, offset_x]:
                cx = off + drone.pos[0] * TILE_SIZE + TILE_SIZE // 2
                cy = drone.pos[1] * TILE_SIZE + TILE_SIZE // 2
                pygame.draw.circle(self.screen, color, (cx, cy), 6)
                dx, dy = DIRECTION_DELTAS[drone.facing]
                pygame.draw.line(self.screen, (255, 0, 0), (cx, cy),
                                 (cx + dx * 8, cy + dy * 8), 2)

        # Labels
        self.screen.blit(self.font.render("True Map", True, (255, 255, 255)),
                         (10, self.height * TILE_SIZE + 10))
        self.screen.blit(self.font.render("Observed Map", True, (255, 255, 255)),
                         (offset_x + 10, self.height * TILE_SIZE + 10))

        # Progress bar
        bar_y = self.height * TILE_SIZE + 40
        bar_w = self.width * TILE_SIZE * 2 + 30
        pygame.draw.rect(self.screen, (60, 60, 60), (10, bar_y, bar_w, 20))
        pygame.draw.rect(self.screen, (0, 200, 0), (10, bar_y, int(bar_w * progress), 20))
        text = f"Progress: {progress*100:.1f}% | Step: {step}/{max_steps}"
        self.screen.blit(self.font.render(text, True, (255, 255, 255)), (10, bar_y + 25))

        pygame.display.flip()
        self.clock.tick(FPS)

        if mode == 'rgb_array':
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), (1, 0, 2))
        return None

    def close(self):
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.screen = None