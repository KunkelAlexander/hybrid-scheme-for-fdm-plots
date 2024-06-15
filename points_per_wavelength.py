import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import os
import glob
import matplotlib.patheffects as PathEffects

# Set the color cycle to use colors from the 'magma' colormap
def set_cycle(num_colors = 10, start=0):
    colors = plt.cm.magma(np.linspace(0, 1, num_colors)[start:])
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

plot_dpi = 600
plot_width = 12
plot_height_single = 3
plot_height_double = 6

set_cycle()

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import gridspec

# Load your images (Replace these lines with your actual image loading code)
image1 = plt.imread('results/points_per_wavelength/Data_000045_Lv_00_Slice_y_density_x1.png')
image2 = plt.imread('results/points_per_wavelength/Data_000045_Lv_00_Slice_y_Phase_x1.png')
image3 = plt.imread('results/points_per_wavelength/Data_000045_Lv_00_Slice_y_points_per_wavelength_x1.png')

sl1 = np.s_[20:-50, 250:-150]

from matplotlib import gridspec
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

plt.savefig("plots/point_per_wavelength.pdf", bbox_inches='tight')
plt.close()
