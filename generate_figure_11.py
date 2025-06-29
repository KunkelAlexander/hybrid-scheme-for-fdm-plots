
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import matplotlib.gridspec as gridspec

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

#### FIRST SUBPLOT ###
def load_and_filter_data(file_path, index_of_interest):
    """Load data from a CSV file and filter it by the specified index."""
    df = pd.read_csv(file_path)
    return df[df['Index'] == index_of_interest]

def plot_density(ax, df, shift, label, l2, linestyle, marker=None, plot_points=False):
    """Plot density data on the provided axis."""
    data = np.roll(df[label], shift)
    line, = ax.plot(df['Coord.'], data, label=l2, linestyle=linestyle, lw=2 if linestyle=="-" else 1.5, alpha=0.8)
    if plot_points:
        if label != "FC-Gram FFT":
            ax.plot(df['Coord.'][::4], data[::4], linestyle='', marker=marker, markersize=4, alpha=0.8, color=line.get_color())
        else:
            ax.plot(df['Coord.'][2::4], data[2::4], linestyle='', marker=marker, markersize=4, alpha=0.8, color=line.get_color())


def plot_error(ax, df, shift):
    """Plot error data on the provided axis."""
    error = np.roll(abs(df['Numerical'] - df['Analytical']), shift)
    ax.plot(df['Coord.'][::4], error[::4], linestyle='--', marker='o', markersize=4, alpha=0.8)

# Define constants and file paths
CSV_FILENAME = 'data/figure_11/combined_density_0_10_t=0.5.csv'
INDEX_OF_INTEREST = 5
SHIFT_LEFT = -18

# Load and prepare data
df = load_and_filter_data(CSV_FILENAME, INDEX_OF_INTEREST)

### SECOND SUBPLOT ###

# Define labels for numerical results
labels = {
    "fd_2_results": "4th-order FD",
    "fd_4_results": "6th-order FD",
    "gramfe_fft_results": "FC-Gram FFT",
    "gramfe_mat_results": "FC-Gram Matrix",
    "base_fft_results": "FFT",
}
set_cycle(7)


font_size = 12
# Set the font size globally
plt.rcParams.update({
    'font.size': font_size,       # Set the base font size
    'axes.titlesize': font_size,  # Set the font size of the axes title
    'axes.labelsize': font_size,  # Set the font size of the axes labels
    'xtick.labelsize': font_size, # Set the font size of the x tick labels
    'ytick.labelsize': font_size, # Set the font size of the y tick labels
    'legend.fontsize': 9, # Set the font size of the legend
    'figure.titlesize': font_size # Set the font size of the figure title
})


# Set up the figure and subplots
fig = plt.figure(figsize=(12, 3), dpi=600)

main_gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.2)  # Two columns, different widths
nested_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=main_gs[0], height_ratios=[3, 1], hspace=0.05)
ax1 = fig.add_subplot(nested_gs[0])  # First subplot in the nested GridSpec
ax2 = fig.add_subplot(nested_gs[1], sharex=ax1)  # Second subplot in the nested GridSpec
ax3 = fig.add_subplot(main_gs[1])    # Regular plot in the second column

#Skip first colour
ax2.plot([],[])

# Plot analytical density
df_analytical = df[df['Source'] == df['Source'].unique()[0]].sort_values("Coord.")
plot_density(ax1, df_analytical, SHIFT_LEFT, 'Analytical', 'Analytical', '-', plot_points=False)


# Plot numerical densities and errors
for source, label in labels.items():
    if source == "base_fft_results":
        continue
    df_source = df[df['Source'] == source].sort_values("Coord.")
    plot_density(ax1, df_source, SHIFT_LEFT, 'Numerical', label, '--', 'o', plot_points=True)
    plot_error(ax2, df_source, SHIFT_LEFT)

# Configure axis labels and scales
ax1.set_ylabel('Density')
ax1.legend(fontsize="small")
ax1.set_xlim(-0.1, 1.0)
ax2.set_xlabel('x')
ax2.set_ylabel('L$_1$ Error')
ax2.set_yscale('log')
ax2.set_ylim([1e-8, 100])
ax2.set_yticks([1e-1, 1e-4, 1e-7])

# Adjust labels and ticks
ax1.yaxis.set_label_coords(-0.12, 0.5)
ax2.yaxis.set_label_coords(-0.12, 0.5)
ax1.tick_params(axis='both', direction='in', top=True, right=True)
ax2.tick_params(axis='both', direction='in', top=True, right=True)

# Manage tick labels visibility
plt.setp(ax1.get_xticklabels(), visible=False)


df = pd.read_csv("data/figure_11/lt_stability_gaussian_gramfe_fft.csv")

set_cycle(7, start=1)

N_points = 64

source_map = {
    "GaussianWavePacketSlow": r"FC-Gram FFT: $k=0.08$",
    #"GaussianWavePacketSlowFilter": r"$142$ points/wavelength w filter",
    "GaussianWavePacketMedium": r"FC-Gram FFT: $k=0.6$",
    "GaussianWavePacketFast": r"FC-Gram FFT: $k=1.7$",

}

# Assign labels based on 'RunName'
df['Label'] = df['Source'].map(source_map)


# Plot data for each label
for label, group_df in df.groupby('Label'):
    x = np.arange(len(group_df['Time']))*10
    y = 0.5 * (group_df['Error(Real)'] + group_df['Error(Imag)'])


    # Calculate logarithmically spaced indices based on the length of k, avoiding log10(0)
    indices = np.geomspace(1, len(x), num=10, dtype=int) - 1  # Subtract 1 to adjust for index starting at 0
    # Ensure unique indices if overlapping occurs due to rounding and bounds
    unique_indices = np.unique(indices)
    line, = plt.plot(x, y, label=f'{label}', linestyle='-')
    ax3.plot(x[unique_indices], y.array[unique_indices], marker='o', linestyle='', c= line.get_color())

    if label == r'$7$ points/wavelength':
        y_ref = y.iloc[10] * (x/x[10])
        ax3.plot(x, y_ref, 'k--', alpha = 0.6, label="$x^1$")


df = pd.read_csv("data/figure_11/lt_stability_gaussian_gramfe_single_matmul.csv")

set_cycle(5, start=1)

N_points = 64

source_map = {
    "GaussianWavePacketSlow": r"FC-Gram Matrix: $k=0.08$",
    #"GaussianWavePacketSlowFilter": r"$142$ points/wavelength w filter",
    "GaussianWavePacketMedium": r"FC-Gram Matrix: $k=0.6$",
    "GaussianWavePacketFast": r"FC-Gram Matrix: $k=1.7$",

}

# Assign labels based on 'RunName'
df['Label'] = df['Source'].map(source_map)


# Plot data for each label
for label, group_df in df.groupby('Label'):
    x = np.arange(len(group_df['Time']))*1000
    y = 0.5 * (group_df['Error(Real)'] + group_df['Error(Imag)'])

    # Calculate logarithmically spaced indices based on the length of k, avoiding log10(0)
    indices = np.geomspace(1, len(x), num=10, dtype=int) - 1  # Subtract 1 to adjust for index starting at 0
    # Ensure unique indices if overlapping occurs due to rounding and bounds
    unique_indices = np.unique(indices)
    line, = plt.plot(x, y, label=f'{label}', linestyle='-')

#plt.title('Real and Imaginary Part Errors as a Function of Wavelength')
ax3.set_xlabel('# Time steps')
ax3.set_ylabel('$L_1$ Error')
ax3.set_yscale("log")
ax3.set_xscale("log")
ax3.set_xlim(1, 1e5)
ax3.set_ylim(1e-18, 1e6)
ax3.tick_params(axis='both', direction='in', top=True, right=True)

# Adding legend
plt.legend(loc="upper left", ncol=2)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save and print plot information
plt.savefig("figures/figure_11.pdf", bbox_inches='tight')
plt.close()