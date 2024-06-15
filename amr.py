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

# Function to add an inset showing a zoomed part of the image
def add_zoomed_inset(ax, image, zoom_factor, inset_extent = [0.1, 0.1, 0.37, 0.37], location="upper right"):
    # Assuming you want to zoom in on the center of the image
    inset_ax = ax.inset_axes(inset_extent)  # Inset axes; adjust location and size as needed
    inset_ax.spines['bottom'].set_color('w')
    inset_ax.spines['top'].set_color('w')
    inset_ax.spines['right'].set_color('w')
    inset_ax.spines['left'].set_color('w')


    num_rows, num_cols = image.shape[:2]
    print(num_rows, num_cols)

    center_row, center_col = num_rows // 2, num_cols // 2
    shift1 = -5
    shift2 = -2
    extent = [1075+shift1, 1300+shift2, 1175+shift1, 1400+shift2]
    #extent = [center_col - num_cols / (2 * zoom_factor) + 30, center_col + num_cols / (2 * zoom_factor) + 30,
    #          center_row - num_rows / (2 * zoom_factor) + 250, center_row + num_rows / (2 * zoom_factor) + 250]
    print(extent)
    inset_ax.imshow(image,extent=extent,origin='upper')
    inset_ax.set_xticklabels([])
    inset_ax.set_yticklabels([])
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])

    zoomed_area_bounds = [1000, 600, extent[1] - extent[0], extent[3] - extent[2]]

    ax.indicate_inset_zoom(inset_ax, edgecolor="white")

# Load your images (Replace these lines with your actual image loading code)
image1 = plt.imread('results/large_box/proj_madelung.png')
image2 = plt.imread('results/large_box/slice_madelung.png')
image3 = plt.imread('results/large_box/proj_wave.png')
image4 = plt.imread('results/large_box/slice_wave.png')
image5 = plt.imread('results/large_box/zoom_slice_madelung.png')  # Image for zoom-in in second column
image6 = plt.imread('results/large_box/zoom_slice_wave.png')  # Image for zoom-in in second column

sl1 = np.s_[50:-60, 100:-585]
sl2 = np.s_[50:-60, 100:-50]
sl3 = np.s_[200:-130, 210:-585]

from matplotlib import gridspec
fig = plt.figure(figsize=(6, 5.5), dpi=600)
width_ratios = [image1[sl1].shape[1], image3[sl2].shape[1]]  # Assuming image1 and image3 define max width in each col
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=width_ratios, hspace=0.0, wspace=0.00)

# Plotting images
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(image1[sl1])
ax1.axis('off')

ax2 = fig.add_subplot(gs[1, 0])
ax2.imshow(image2[sl1])
ax2.axis('off')

# Adding zoom-in insets
add_zoomed_inset(ax2, image5[sl3], 10, inset_extent=[0.017, 0.023, 0.365, 0.365], location="upper right")

ax3 = fig.add_subplot(gs[0, 1])
ax3.imshow(image3[sl2])
ax3.axis('off')

ax4 = fig.add_subplot(gs[1, 1])
ax4.imshow(image4[sl2])
ax4.axis('off')
add_zoomed_inset(ax4, image6[sl3], 10, inset_extent=[-0.02+0.01, 0.03, 0.35, 0.35], location="upper right")


# Add color definitions
c1, c2, c3, c4, c5, c6, c7 = plt.cm.magma(np.array(np.linspace(0, 1, 9)[1:8]))

# Add vertical boxes
fig.patches.extend([
    Rectangle((0.13, 0.115), 0.34, 0.756, fill=False, edgecolor=c3, linewidth=3, transform=fig.transFigure, figure=fig),
    Rectangle((0.48, 0.115), 0.338, 0.756, fill=False, edgecolor=c6, linewidth=3, transform=fig.transFigure, figure=fig)
])
# Add text annotations
ax1.text(0.5, 0.8, 'Fluid', transform=ax1.transAxes, fontsize=12, color=c3, ha='center', va='bottom',
         bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'), weight='bold')

ax3.text(0.42, 0.8, 'Wave', transform=ax3.transAxes, fontsize=12, color=c6, ha='center', va='bottom',
         bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'), weight='bold')

#text1 = ax1.text(0.5, 0.8, 'Fluid', transform=ax1.transAxes, fontsize=12, color=c4, ha='center', va='bottom', weight='bold')
#text1.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
#
#text2 = ax3.text(0.42, 0.8, 'Wave', transform=ax3.transAxes, fontsize=12, color=c5, ha='center', va='bottom', weight='bold')
#text2.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])


plt.savefig("plots/large_box.pdf", bbox_inches='tight')
plt.close()
