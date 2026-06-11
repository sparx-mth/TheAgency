from dataclasses import dataclass, field
from typing import Tuple, List, Optional
from sparx_agency.tasks.planning.slam_simulator.constants import DIRECTIONS, DIRECTION_DELTAS

@dataclass
class Drone:
    drone_id: int
    pos: Tuple[int, int]
    facing: str = 'NORTH'
    active: bool = True
    collision_count: int = 0
    total_discoveries: int = 0

    def get_facing_idx(self) -> int:
        return DIRECTIONS.index(self.facing)

    def turn(self, direction: str) -> None:
        idx = self.get_facing_idx()
        if direction == 'LEFT':
            self.facing = DIRECTIONS[(idx - 1) % 4]
        else:
            self.facing = DIRECTIONS[(idx + 1) % 4]

    def get_forward_pos(self) -> Tuple[int, int]:
        dx, dy = DIRECTION_DELTAS[self.facing]
        return (self.pos[0] + dx, self.pos[1] + dy)

    def move_to(self, new_pos: Tuple[int, int]) -> None:
        self.pos = new_pos

    def add_collision(self) -> None:
        self.collision_count += 1

    def add_discoveries(self, count: int) -> None:
        self.total_discoveries += count