import numpy as np
from PIL import Image

grid = np.load('cropped_occ_grid_int8.npy')

img_array = np.zeros(grid.shape, dtype=np.uint8)

img_array[grid == 0] = 255

img_array[grid == 100] = 0

img_array[grid == -1] = 127

img_array = np.flipud(img_array)


img = Image.fromarray(img_array)
#img.save('cropped_map_visual.jpg') 
img.save('cropped_map_visual.png') # מומלץ יותר למפות
print("Image saved successfully!")