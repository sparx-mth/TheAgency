"""Poses that cover a room evenly, for asking what the best place and direction to look from would be: one position per square of floor, each with several headings."""
import numpy as np


def pose_grid(room_cells, spacing_m, headings, cap, seed):
    """``[(position, yaw)]``: the floor cell nearest the middle of each ``spacing_m`` square, at most ``cap`` squares (chosen at random), each looked at from ``headings`` equally spaced yaws."""
    squares = {}
    for cell in room_cells:
        squares.setdefault((int(cell[0] // spacing_m), int(cell[2] // spacing_m)), []).append(cell)
    chosen = []
    for key in sorted(squares):
        middle = (np.array(key) + 0.5) * spacing_m
        chosen.append(min(squares[key], key=lambda c: np.hypot(c[0] - middle[0], c[2] - middle[1])))
    order = np.random.default_rng(seed).permutation(len(chosen))[:cap]
    yaws = [2 * np.pi * i / headings for i in range(headings)]
    return [(chosen[i], yaw) for i in sorted(order) for yaw in yaws]
