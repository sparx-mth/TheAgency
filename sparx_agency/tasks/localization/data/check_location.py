import numpy as np
grid = np.load('/home/shirb/GIT/TheAgency/sparx_agency/tasks/localization/data/cropped_occ_grid_int8.npy')
free_space_indices = np.where(grid == 0)
rows = free_space_indices[0]
cols = free_space_indices[1]
min_row_idx = np.argmin(rows)
tip_x = rows[min_row_idx]
tip_y = cols[min_row_idx]
 
print(f"The white tip is located at Matrix Cell: X (row) = {tip_x}, Y (col) = {tip_y}")
