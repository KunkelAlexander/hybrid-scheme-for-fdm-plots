import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox, TransformedBbox, blended_transform_factory
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
zooms = [1, 5, 20, 100]
row = [[], [], [], []]
row[0] = [plt.imread(f'results/halo/Data_000045_Lv_10_Slice_z_density_x{zoom}.png') for zoom in zooms]
row[1] = [plt.imread(f'results/halo/Data_000045_Lv_10_Slice_z_density_wave_grid_x{zoom}.png') for zoom in zooms]
row[2] = [plt.imread(f'results/halo/Data_000045_Lv_10_Slice_z_Phase_x{zoom}.png') for zoom in zooms]
row[3] = [plt.imread(f'results/halo/Data_000045_Lv_10_Slice_z_Phase_wave_grid_x{zoom}.png') for zoom in zooms]


sl1 = np.s_[100:-40, 100:-745]
sl2 = np.s_[100:-40, 100:-50]


from matplotlib import gridspec

#width_ratios = [image1[sl1].shape[1], image3[sl2].shape[1]]  # Assuming image1 and image3 define max width in each col
width_ratios = [row[0][0][sl1].shape[1], row[0][0][sl1].shape[1], row[0][0][sl1].shape[1], row[0][0][sl2].shape[1]]  # Assuming image1 and image3 define max width in each col
#gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1], width_ratios=width_ratios, hspace=0.0, wspace=0.00)



# Initial full image dimensions (adjust these to your image dimensions)
image_height, image_width = row[0][0][sl1].shape[:2]  # Assumes all images have the same dimensions

fig = plt.figure(figsize=(6, 6), dpi=600)
gs = gridspec.GridSpec(4, 4, height_ratios=[1, 1, 1, 1], width_ratios=width_ratios, hspace=0.0, wspace=0.00)
axes = [[fig.add_subplot(gs[i, j]) for j in range(4)] for i in range(4)]

def calculate_zoom_rectangle(width, height, zoom_factor):
    new_width = width / zoom_factor
    new_height = height / zoom_factor
    x_start = (width - new_width) / 2
    y_start = (height - new_height) / 2
    return (x_start, y_start, new_width, new_height)

rel_zooms = [zooms[1]/zooms[0], zooms[2]/zooms[1], zooms[3]/zooms[2], 1]

# Generate rectangles based on zoom levels
rectangles = [calculate_zoom_rectangle(image_width, image_height, z) for z in rel_zooms]


for i in range(4):
    for j in range(4):
        ax = axes[i][j]
        ax.imshow(row[i][j][sl1] if j != 3 else row[i][j][sl2])


for i in range(4):
    for j in range(4):
        ax = axes[i][j]
        # Draw a rectangle on the current axis, if not the last column
        if j < 3:
            rect = Rectangle((rectangles[j][0], rectangles[j][1]), rectangles[j][2], rectangles[j][3],
                             linewidth=0.5, edgecolor="white", facecolor='none')
            ax.add_patch(rect)

        # Draw lines to the next subplot
        if j < 3:  # Avoid drawing lines in the last column
            # Calculate points directly from axes positions
            trans = ax.transData.transform
            inv_fig_trans = fig.transFigure.inverted().transform

            # Current rectangle upper right and lower right
            ur = trans((rectangles[j][0] + rectangles[j][2], rectangles[j][1]))
            lr = trans((rectangles[j][0] + rectangles[j][2], rectangles[j][1] + rectangles[j][3]))

            # Convert these points to figure coordinates
            fig_ur = inv_fig_trans(ur)
            fig_lr = inv_fig_trans(lr)

            # Next subplot's upper left and lower left as figure coordinates
            next_ax_bounds = axes[i][j+1].get_position()
            next_ul = [next_ax_bounds.x0, next_ax_bounds.y1]
            next_ll = [next_ax_bounds.x0, next_ax_bounds.y0]

            print(fig_ur, fig_lr, next_ul, next_ll)

            # Draw lines in figure coordinates
            fig.add_artist(plt.Line2D([fig_ur[0], next_ul[0]], [fig_ur[1], next_ul[1]], color='white', lw=0.5))
            fig.add_artist(plt.Line2D([fig_lr[0], next_ll[0]], [fig_lr[1], next_ll[1]], color='white', lw=0.5))

        ax.axis('off')


plt.savefig("plots/halo.pdf", bbox_inches='tight')
plt.close()
