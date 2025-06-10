import math
from src.planner.simulation.sensors.base_sensor import BaseSensor
from src.planner.simulation.simulation_constants import WALL, DOOR_CLOSED


class BresenhamFOVSensor(BaseSensor):
    def __init__(self, radius):
        self.radius = radius

    def sense(self, pos, facing, env):
        def bresenham(x0, y0, x1, y1):
            dx = abs(x1 - x0)
            dy = -abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx + dy
            while True:
                yield x0, y0
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x0 += sx
                if e2 <= dx:
                    err += dx
                    y0 += sy

        cx, cy = pos
        observations = []

        for offset_y in range(-self.radius, self.radius + 1):
            for offset_x in range(-self.radius, self.radius + 1):
                x, y = cx + offset_x, cy + offset_y
                if not (0 <= x < env.width and 0 <= y < env.height):
                    continue
                if offset_x ** 2 + offset_y ** 2 > self.radius ** 2:
                    continue

                for lx, ly in bresenham(cx, cy, x, y):
                    if not (0 <= lx < env.width and 0 <= ly < env.height):
                        break
                    val = env.get_tile(lx, ly)
                    observations.append((lx, ly, val))
                    if val in {WALL, DOOR_CLOSED}:
                        break

        return observations
