import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import AxesGrid
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
dataset = yt.load("data/figure_12/Data_000001")

current_axis = "z"

# Create a new figure for the current slice
fig = plt.figure(dpi=plot_dpi, figsize=(plot_width, plot_height_double))

# Create a grid of axes for multiple plots
grid = AxesGrid(
    fig,
    #(0.1, 0.1, 0.8, 0.8),
    (0, 0, 1, 1),
    nrows_ncols=(1, 4),
    axes_pad=(0.53, 0.0),
    label_mode="L",
    share_all=True,
    cbar_location="right",
    cbar_mode="each",
    direction="row",
    cbar_size="5%",
    cbar_pad="2%",
)


def reim2phase(field, data):
    return np.arctan2(data["gamer", "Imag"], data["gamer", "Real"])

yt.add_field(("gamer", "Phase"), function=reim2phase, sampling_type="local", units="")


# Define the fields to plot
fields_to_plot = [
    ("gas", "density"),
    ("gamer", "Phase"),
    ("gamer", "Real"),
    ("gamer", "Imag"),
]

# Create a slice plot for the current dataset and field
slice_plot = yt.SlicePlot(dataset, current_axis, fields_to_plot, center=(0.25, 0.25, 0))
slice_plot.set_log(("gamer", "Phase"), False)
slice_plot.set_log(("gamer", "Real"), False)
slice_plot.set_log(("gamer", "Imag"), False)

for field in fields_to_plot:
    slice_plot.set_cmap(field, colormap)

# For each plotted field, associate it with the corresponding AxesGrid axes
for i, field in enumerate(fields_to_plot):
    plot = slice_plot.plots[field]
    plot.figure = fig
    plot.axes = grid[i].axes
    plot.cax = grid.cbar_axes[i]

# Redraw the plot on the AxesGrid axes
slice_plot._setup_plots()

# Assuming 'z' axis slices, the size of the plot, and other parameters
axis = 'z'
width = 1

grid[0].axis("off")
grid[1].axis("off")
grid[2].axis("off")
grid[3].axis("off")
grid[0].set_title(r"(a) Density $\rho$", fontsize=16)
grid[1].set_title(r"(b) Phase $S/\hbar$", fontsize=16)
grid[2].set_title(r"(c) Real Part $\Re(\psi)$", fontsize=16)
grid[3].set_title(r"(d) Imaginary Part $\Im(\psi)$", fontsize=16)

grid.cbar_axes[0].get_yaxis().set_ticks([1e-3, 1e-2, 1e-1, 1])
grid.cbar_axes[0].set_ylabel(r" ", rotation=90)
grid.cbar_axes[0].tick_params(labelsize=16)

# Add ticks and labels for the first colorbar axis
grid.cbar_axes[1].get_yaxis().set_ticks([-np.pi/2, 0, np.pi/2])
grid.cbar_axes[1].get_yaxis().set_ticklabels([r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$'])
grid.cbar_axes[1].set_ylabel(" ", rotation=90)
grid.cbar_axes[1].tick_params(labelsize=16)

# Add ticks and labels for the third colorbar axis
grid.cbar_axes[2].set_ylabel(" ", rotation=90)
grid.cbar_axes[2].tick_params(labelsize=16)

# Add ticks and labels for the fourth colorbar axis
grid.cbar_axes[3].set_ylabel(" ", rotation=90)
grid.cbar_axes[3].tick_params(labelsize=16)

# Get the DumpID from dataset parameters and save the plot
dump_id = dataset.parameters["DumpID"]
plt.savefig("figures/figure_12.pdf", bbox_inches='tight')

# Close the current figure to release resources
plt.close()
