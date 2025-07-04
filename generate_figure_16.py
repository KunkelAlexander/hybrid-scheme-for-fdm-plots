import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import os
import glob

import matplotlib.patheffects as PathEffects


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import gridspec

# Set the color cycle to use colors from the 'magma' colormap
def set_cycle(num_colors = 10, start=0):
    colors = plt.cm.magma(np.linspace(0, 1, num_colors)[start:])
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)

plot_dpi = 600
plot_width = 12
plot_height_single = 3
plot_height_double = 6

set_cycle()
sl1 = np.s_[100:-40, 100:-850]
sl2 = np.s_[100:-40, 100:-50]

# Load your images (Replace these lines with your actual image loading code)
ids = [18, 19, 24, 30]
row = [[], [], [], []]
row[0] = [plt.imread(f'data/figure_16/Data_{id:06d}_Proj_y_density.png') for id in ids]
row[1] = [plt.imread(f'data/figure_16/Data_{id:06d}_Proj_y_density_grid.png') for id in ids]
row[2] = [plt.imread(f'data/figure_16/Data_{id:06d}_000002_Slice_x_density.png') for id in ids]
row[3] = [plt.imread(f'data/figure_16/Data_{id:06d}_000002_Slice_x_phase_wrapped.png') for id in ids]

fig = plt.figure(figsize=(6, 6), dpi=600)
width_ratios = [row[0][0][sl1].shape[1], row[0][0][sl1].shape[1], row[0][0][sl1].shape[1], row[0][0][sl2].shape[1]]  # Assuming image1 and image3 define max width in each col
gs = gridspec.GridSpec(4, 4, height_ratios=[1, 1, 1, 1], width_ratios=width_ratios, hspace=0.0, wspace=0.01)

row_titles = ["Projection", "Wave\nGrid", "Zoom-in\nDensity\nSlice", "Zoom-in\nPhase\nSlice"]

axes = []
for j in range(len(row)):
    row_ax = []
    for i, im in enumerate(row[j]):
        ax = fig.add_subplot(gs[j, i])
        row_ax.append(ax)
        # Plotting images
        ax.imshow(im[sl1] if i != 3 else im[sl2])
        ax.axis('off')
        # Adding white box around the subplots
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
            spine.set_linewidth(4)

    axes.append(row_ax)



for j in range(len(row)):
    for i, im in enumerate(row[j]):
        ax = axes[j][i]
        # Adding row-wise titles as annotations in the top center of the left plot of each row
        if i == 0:
            ax.annotate(row_titles[j], xy=(0.1, 0.7), xycoords="axes fraction", fontsize=7, fontweight="bold",
                        ha='left', va='center', zorder=10, color="white")

        ## Adding specific annotations
        if j == 1 and i == 1:
            ax.annotate('Switch from\nfluid to wave', xy=(0.8, 0.25), xytext=(0.5, 0.5), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='white'),
                        fontsize=6, ha='center', va='center', color='white')
#
        if j == 2 and i == 2:
            ax.annotate('Destructive\ninterference', xy=(0.4, 0.4), xytext=(0.4, 0.12), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='white'),
                        fontsize=6, ha='center', va='center', color='white')


plt.savefig("figures/figure_16.pdf", bbox_inches='tight')
plt.close()