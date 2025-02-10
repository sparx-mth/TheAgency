import math
import numpy as np


class Utils:
    @staticmethod
    def calculate_distance(vector1, vector2):
        if len(vector1) != 3 or len(vector2) != 3:
            raise ValueError("Both vectors should have three components")

        # Calculate the squared differences of each component
        squared_diffs = [(vector1["x"] - vector2["x"]) ** 2,
                         (vector1["y"] - vector2["y"]) ** 2,
                         (vector1["z"] - vector2["z"]) ** 2]

        # Calculate the sum of squared differences
        sum_squared_diffs = sum(squared_diffs)

        # Calculate the square root of the sum
        distance = math.sqrt(sum_squared_diffs)

        return distance

    @staticmethod
    def calculate_direction_vector(vector1, vector2):
        """
        Calculates the normalized direction vector between two 3D vectors.

        Args:
            vector1 (list or np.array): The first 3D vector.
            vector2 (list or np.array): The second 3D vector.

        Returns:
            dict: The normalized direction vector from vector1 to vector2 in the format {"x": x_val, "y": y_val, "z": z_val}.
        """
        vector1 = np.array(vector1)
        vector2 = np.array(vector2)

        direction_vector = vector2 - vector1
        direction_vector /= np.linalg.norm(direction_vector)

        direction_vector_dict = {"x": direction_vector[0], "y": direction_vector[1], "z": direction_vector[2]}

        return direction_vector_dict

    @staticmethod
    def calculate_height_differences(my_loc, agents_locs):
        my_height = my_loc[1]
        height_differences = [my_height - agent_loc[1] for agent_loc in agents_locs]
        return height_differences
