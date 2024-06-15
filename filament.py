import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import os
import glob

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

# Function to add an inset showing a zoomed part of the image
def add_zoomed_inset(ax, image, zoom_factor, location="upper right"):
    # Assuming you want to zoom in on the center of the image
    inset_ax = ax.inset_axes([0.0, 0.0, 0.35, 0.35])  # Inset axes; adjust location and size as needed
    inset_ax.spines['bottom'].set_color('w')
    inset_ax.spines['top'].set_color('w')
    inset_ax.spines['right'].set_color('w')
    inset_ax.spines['left'].set_color('w')


    num_rows, num_cols = image.shape[:2]

    center_row, center_col = num_rows // 2, num_cols // 2
    extent = [center_col - num_cols / (2 * zoom_factor) + 30, center_col + num_cols / (2 * zoom_factor) + 30,
              center_row - num_rows / (2 * zoom_factor) + 250, center_row + num_rows / (2 * zoom_factor) + 250]
    print(extent)
    inset_ax.imshow(image,extent=extent,origin='upper')
    inset_ax.set_xticklabels([])
    inset_ax.set_yticklabels([])
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])

    zoomed_area_bounds = [extent[0], extent[2], extent[1] - extent[0], extent[3] - extent[2]]

    ax.indicate_inset_zoom(inset_ax, edgecolor="white")

# Load your images (Replace these lines with your actual image loading code)
ids = [18, 19, 24, 30]
row = [[], [], [], []]
row[0] = [plt.imread(f'results/filament/Data_{id:06d}_Proj_y_density.png') for id in ids]
row[1] = [plt.imread(f'results/filament/Data_{id:06d}_Proj_y_density_grid.png') for id in ids]
row[2] = [plt.imread(f'results/filament/Data_{id:06d}_000002_Slice_x_density.png') for id in ids]
row[3] = [plt.imread(f'results/filament/Data_{id:06d}_000002_Slice_x_phase_wrapped.png') for id in ids]


sl1 = np.s_[60:-60, 100:-495]
sl2 = np.s_[60:-60, 100:-50]

from matplotlib import gridspec
fig = plt.figure(figsize=(6, 6), dpi=600)
#width_ratios = [image1[sl1].shape[1], image3[sl2].shape[1]]  # Assuming image1 and image3 define max width in each col
width_ratios = [row[0][0][sl1].shape[1], row[0][0][sl1].shape[1], row[0][0][sl1].shape[1], row[0][0][sl2].shape[1]]  # Assuming image1 and image3 define max width in each col
#gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1], width_ratios=width_ratios, hspace=0.0, wspace=0.00)
gs = gridspec.GridSpec(4, 4, height_ratios=[1, 1, 1, 1], width_ratios=width_ratios, hspace=0.0, wspace=0.00)

for j in range(len(row)):
    for i, im in enumerate(row[j]):
        ax = fig.add_subplot(gs[j, i])
        # Plotting images
        ax.imshow(im[sl1] if i != 3 else im[sl2])
        ax.axis('off')


plt.savefig("plots/filament.pdf", bbox_inches='tight')
plt.close()
