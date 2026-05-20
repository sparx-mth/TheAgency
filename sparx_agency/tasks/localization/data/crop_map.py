import numpy as np

grid = np.load('occ_grid_int8.npy')
valid_indices = np.where(grid != -1)

if valid_indices[0].size > 0:
    y_min, y_max = np.min(valid_indices[0]), np.max(valid_indices[0])
    x_min, x_max = np.min(valid_indices[1]), np.max(valid_indices[1])

    padding = 2
    y_min = max(0, y_min - padding)
    y_max = min(grid.shape[0] - 1, y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(grid.shape[1] - 1, x_max + padding)

    cropped_grid = grid[y_min:y_max+1, x_min:x_max+1]
    np.save('cropped_occ_grid_int8.npy', cropped_grid)

    print(f"Original shape: {grid.shape}")
    print(f"Cropped shape: {cropped_grid.shape}")
    print(f"Crop Offsets (y_min, x_min): ({y_min}, {x_min})")
else:
    print("The map is entirely empty/unknown.")
