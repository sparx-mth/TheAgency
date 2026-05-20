import numpy as np

original_map_path = 'occ_grid_int8.npy'
new_map_path = 'rotated_occ_grid_int8.npy'

grid = np.load(original_map_path)

rotated_grid = np.rot90(grid, k=-1)

np.save(new_map_path, rotated_grid)

