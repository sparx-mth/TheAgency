import numpy as np
import json

old_map_path = "/home/user1/GIT/TheAgency/sparx_agency/tasks/localization/data/cropped_occ_grid_int8.npy"
old_metadata_path = "/home/user1/GIT/TheAgency/sparx_agency/tasks/localization/data/occ_metadata.json"

new_map_path = "/home/user1/GIT/TheAgency/sparx_agency/tasks/localization/data/cropped_occ_grid_int8_res_0_1.npy"
new_metadata_path = "/home/user1/GIT/TheAgency/sparx_agency/tasks/localization/data/occ_metadata_res_0_1.json"

old_res = 0.05
new_res = 0.1
scale = int(new_res / old_res)  # 2

occ = np.load(old_map_path)

H, W = occ.shape

H_new = H // scale
W_new = W // scale

occ_cropped = occ[:H_new * scale, :W_new * scale]

blocks = occ_cropped.reshape(H_new, scale, W_new, scale)

new_occ = np.zeros((H_new, W_new), dtype=np.int8)

has_obstacle = np.any(blocks == 100, axis=(1, 3))
has_unknown = np.any(blocks == -1, axis=(1, 3))

new_occ[has_unknown] = -1
new_occ[has_obstacle] = 100

np.save(new_map_path, new_occ)

print("Old shape:", occ.shape)
print("New shape:", new_occ.shape)
print("Saved:", new_map_path)