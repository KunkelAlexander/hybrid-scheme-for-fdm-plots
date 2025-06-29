import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import yt



# Set the color cycle to use colors from the 'magma' colormap
def set_cycle(num_colors = 10, start=0):
    colors = plt.cm.magma(np.linspace(0, 1, num_colors)[start:])
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)
    return colors

plot_dpi = 600
plot_width = 12
plot_height_single = 3
plot_height_double = 6

set_cycle()


colormap = 'magma'  # Define the colormap for the plots
# Create a series of datasets based on data files with indices in the specified range
dataset = yt.load("data/figure_3/Data_000001")

current_axis = "z"

# Create a new figure for the current slice
fig = plt.figure(dpi=600, figsize=(plot_width, 5))

# Create a grid of axes for multiple plots
grid = AxesGrid(
    fig,
    (0.075, 0.075, 0.85, 0.85),
    nrows_ncols=(2, 2),
    axes_pad=(0.2, 0.0),
    label_mode="L",
    share_all=True,
    cbar_location="right",
    cbar_mode="edge",
    direction="row",
    cbar_size="3%",
    cbar_pad="0%",
)

# Define the fields to plot
fields_to_plot = [
    ("gas", "density"),
    ("gamer", "Phase"),
]

# Create a slice plot for the current dataset and field
slice_plot = yt.SlicePlot(dataset, current_axis, fields_to_plot)
slice_plot.set_log(("gamer", "Phase"), False)

slice_plot.annotate_grids(periodic=False)

for field in fields_to_plot:
    slice_plot.set_cmap(field, colormap)

# For each plotted field, associate it with the corresponding AxesGrid axes
for i, field in enumerate(fields_to_plot):
    plot = slice_plot.plots[field]
    plot.figure = fig
    plot.axes = grid[2 * i].axes
    plot.cax = grid.cbar_axes[i]

# Create a second slice plot for comparison
slice_plot_2 = yt.SlicePlot(dataset, current_axis, fields_to_plot)
slice_plot_2.set_log(("gamer", "Phase"), False)
slice_plot_2.annotate_grids(periodic=False)

for field in fields_to_plot:
    slice_plot_2.set_cmap(field, colormap)

# Associate the second slice plot with the AxesGrid axes
for i, field in enumerate(fields_to_plot):
    plot = slice_plot_2.plots[field]
    plot.figure = fig
    plot.axes = grid[2 * i + 1].axes

# Redraw the plot on the AxesGrid axes
slice_plot._setup_plots()
slice_plot_2._setup_plots()

# Assuming 'z' axis slices, the size of the plot, and other parameters
axis = 'z'
width = 1#dataset.domain_width.min().value  # Adjust based on your dataset

# Create Fixed Resolution Buffers (FRBs) for density and AMR level visualization
res = [800, 800]  # Adjust resolution based on your needs

# Extracting AMR level data
# Note: Modify this part according to how you can get AMR level data from your dataset
amr_data = slice_plot.data_source.to_frb(width, res, height=width)

# Visualizing AMR levels
# Assume `amr_data["index", "grid_level"]` provides the AMR level data, you may need to adjust this
amr_levels = amr_data["index", "grid_level"]
level_0_mask = amr_levels == 0
level_1_mask = amr_levels == 1

# Initialize an RGBA image with full transparency (alpha = 0)
amr_image_rgba = np.zeros((res[0], res[1], 4), dtype=np.float32)  # Last dimension for RGBA

# Set colors with alpha for AMR levels
# Here, 0.5 is used as an example alpha value for both levels, adjust according to your preferences
light_grey_rgba = [0.8, 0.8, 0.8, 0.9]  # Light grey with alpha for level 0
dark_grey_rgba  = [0.8, 0.8, 0.8, 0.7]  # Dark grey with alpha for level 1

# Apply colors with alpha based on AMR level masks
amr_image_rgba[level_0_mask, :] = light_grey_rgba
amr_image_rgba[np.logical_not(level_0_mask), :] = dark_grey_rgba


# Plotting the RGBA image with alpha
# Using the same extent as before to align with other plots
grid[0].imshow(amr_image_rgba, origin='lower', extent=[-0.5, 0.5, -0.5, 0.5])
grid[2].imshow(amr_image_rgba, origin='lower', extent=[-0.5, 0.5, -0.5, 0.5])

dy = 0.06

# Add annotations or other plot elements as needed
txt = grid[0].annotate(r"$\rho$", (-0.4, 0 + dy), fontsize=30, c="w")
txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
#grid[0].annotate(r"$\rho$", (-0.4, 0), fontsize=28, c="white")
txt = grid[0].annotate(r"$\psi$", (0.08, 0 + dy), fontsize=30, c="w")
txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])

grid[0].text(-0.15, 0.1 + dy, r"$\sqrt{\rho/m} e^{iS/\hbar}$",
            ha="center", va="center", rotation=0, size=10,
            bbox=dict(boxstyle="rarrow,pad=0.3",
                      fc="mistyrose", ec="purple", lw=2))
grid[0].text(-0.15, -0.1 + dy, r"$m |\psi|^2$",
            ha="center", va="center", rotation=0, size=10,
            bbox=dict(boxstyle="larrow,pad=0.3",
                      fc="mistyrose", ec="purple", lw=2))

txt = grid[2].annotate(r"$S/\hbar$", (-0.42, 0 + dy), fontsize=24, c="w")
txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
txt = grid[2].annotate(r"$\psi$", (0.08, 0 + dy), fontsize=24, c="w")
txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
grid[2].text(-0.15, 0.1 + dy, r"$\sqrt{\rho/m} e^{iS/\hbar}$",
            ha="center", va="center", rotation=0, size=10,
            bbox=dict(boxstyle="rarrow,pad=0.3",
                      fc="mistyrose", ec="purple", lw=2))
grid[2].text(-0.15, -0.11 + dy, r"atan2$(\Im(\psi), \Re(\psi)) +  2 n \pi$",
            ha="center", va="center", rotation=0, size=10,
            bbox=dict(boxstyle="larrow,pad=0.3",
                      fc="mistyrose", ec="purple", lw=2))


txt = grid[1].annotate("Vortices",
            xy=(0.158, 0.14), xycoords='data',
            xytext=(-0.1, -0.3), textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), c="w", fontsize=14)
txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
grid[1].annotate("Vortices",
            xy=(0.05, 0.048), xycoords='data',
            xytext=(-0.1, -0.3), textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), c="w", fontsize=14, alpha=1)


txt = grid[3].annotate("Phase winding",
            xy=(0.158, 0.14), xycoords='data',
            xytext=(-0.1, -0.3), textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), c="w", fontsize=14)
txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])
grid[3].annotate("Phase winding",
            xy=(0.05, 0.048), xycoords='data',
            xytext=(-0.1, -0.3), textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), c="w", fontsize=14, alpha=1)

grid[0].axis("off")
grid[1].axis("off")
grid[2].axis("off")
grid[3].axis("off")
grid.cbar_axes[0].get_yaxis().labelpad = 6
grid.cbar_axes[0].get_yaxis().set_ticks([1e-2, 1e-1, 1, 1e1])
grid.cbar_axes[0].tick_params(labelsize=12)
grid.cbar_axes[0].set_ylabel(r"Density $\rho$", rotation=90, fontsize=12)

grid.cbar_axes[1].get_yaxis().labelpad = 2
grid.cbar_axes[1].set_ylabel(r"Phase $S/\hbar$", rotation=90, fontsize=12)
grid.cbar_axes[1].get_yaxis().set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r"$-\pi$", r"$-\frac{\pi}{2}$", "0", r"$\frac{\pi}{2}$", r"$\pi$"])
grid.cbar_axes[1].tick_params(labelsize=14)
# Get the DumpID from dataset parameters and save the plot
dump_id = dataset.parameters["DumpID"]
plt.savefig("figure_3.pdf", bbox_inches='tight')

plt.close()
