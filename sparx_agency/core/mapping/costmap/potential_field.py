import numpy as np
import cv2
from typing import Tuple
from sparx_agency.core.mapping.interfaces.costmap import Costmap

class PotentialFieldCalculator:
    def __init__(
            self,
            zeta: float = 0.5,  # ζ: Attractive gain
            eta: float = 15.0,  # η: Repulsive gain
            rho_0: float = 2.0,  # ρ₀: Distance of influence (meters)
    ):
        self.zeta = zeta
        self.eta = eta
        self.rho_0 = rho_0

    def calculate_force(self, q: np.ndarray, q_goal: np.ndarray, costmap: Costmap) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates F_att and F_rep based on the ProbabilisticGridCostmap.
        q: Current [x, y] in meters.
        q_goal: Target [x, y] in meters.
        """
        # 1. Attractive Force: F_att = ζ * (q_goal - q)
        f_att = self.zeta * (q_goal - q)

        # 2. Repulsive Force: F_rep
        f_rep = np.array([0.0, 0.0], dtype=np.float32)

        grid_spec, grid_data = costmap.get_grid()
        res = grid_spec.resolution

        # We find occupied cells (value 100) in the grid
        # To be efficient, we only look at a local window around the drone
        window_px = int(self.rho_0 / res) + 2
        gx = int((q[0] - grid_spec.origin_x) / res)
        gy = int((q[1] - grid_spec.origin_y) / res)

        y_min, y_max = max(0, gy - window_px), min(grid_data.shape[0], gy + window_px)
        x_min, x_max = max(0, gx - window_px), min(grid_data.shape[1], gx + window_px)

        # Extract obstacles from the local area
        local_grid = grid_data[y_min:y_max, x_min:x_max]
        obs_indices = np.where(local_grid == 100)

        for oy, ox in zip(obs_indices[0], obs_indices[1]):
            # Convert grid index back to world meters
            q_obs = np.array([
                (ox + x_min) * res + grid_spec.origin_x,
                (oy + y_min) * res + grid_spec.origin_y
            ])

            diff = q - q_obs
            rho_q = np.linalg.norm(diff)

            if 0 < rho_q <= self.rho_0:
                grad_rho = diff / rho_q  # Direction away from obstacle
                # Equation 4.4 from book
                scalar = self.eta * (1.0 / rho_q - 1.0 / self.rho_0) * (1.0 / rho_q ** 2)
                f_rep += scalar * grad_rho

        return f_att, f_rep


    @staticmethod
    def draw_potential_field(image, q_drone, q_goal, force_vec):
        """
        Visualizes the force on the RGB frame.
        """
        # Draw Goal (Target)
        cv2.circle(image, tuple(q_goal.astype(int)), 8, (0, 255, 0), -1)

        # Draw Drone Position
        cv2.circle(image, tuple(q_drone.astype(int)), 5, (255, 0, 0), -1)

        # Draw Resultant Force Arrow
        # We scale the force vector so it's visible (e.g., * 20 pixels)
        end_point = (q_drone + force_vec * 20).astype(int)
        cv2.arrowedLine(image, tuple(q_drone), tuple(end_point), (0, 255, 255), 2)

        return image