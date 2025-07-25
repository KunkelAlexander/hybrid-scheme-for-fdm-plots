import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


# Load your images (Replace these lines with your actual image loading code)
image1 = plt.imread('data/figure_2/Data_000045_Lv_00_Slice_y_density_x1.png')
image2 = plt.imread('data/figure_2/Data_000045_Lv_00_Slice_y_Phase_x1.png')
image3 = plt.imread('data/figure_2/Data_000045_Lv_00_Slice_y_points_per_wavelength_x1.png')

sl1 = np.s_[20:-50, 250:-150]

fig = plt.figure(figsize=(10, 12), dpi=600)
gs = gridspec.GridSpec(1, 3, wspace=0.1)

# Plotting images
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(image1[sl1])
ax1.axis('off')

ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(image2[sl1])
ax2.axis('off')

ax3 = fig.add_subplot(gs[0, 2])
ax3.imshow(image3[sl1])
ax3.axis('off')

plt.savefig("figures/figure_2.pdf", bbox_inches='tight')
plt.close()