import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy
import numpy as np
import matplotlib.patheffects as PathEffects


redshifts     = np.array([100, 99, 90, 80, 70, 63, 50, 40, 31, 20, 15, 14, 13, 12, 11, 10, 9, 8, 7])
redshifts     = np.sort(redshifts)[::-1]
scale_factors = 1/(redshifts+1)

# Read the DataFrame from disk

input_file = "data/figure_15/fdm_spectra.csv"
fdm_df = pd.read_csv(input_file)
input_file = "data/figure_15/cdm_spectra.csv"
cdm_df = pd.read_csv(input_file)
combined_df = pd.concat([fdm_df, cdm_df], ignore_index=True)

# Define your custom order for the 'Method' column assuming correct mapping to your methods
method_order = ['fluid', 'nbody', 'spectral']  # Adjust the actual method names
combined_df['Method'] = pd.Categorical(combined_df['Method'], categories=method_order, ordered=True)

# Assuming 'Resolution' contains the 'N' value like 'n_64', 'n_128', etc.
# Sorting by 'Method' first and then by the numeric part of the 'Resolution'
combined_df['N_value'] = combined_df['Resolution'].apply(lambda x: int(x.split('_')[1]))
combined_df = combined_df.sort_values(by=['Method', 'N_value'])


# Set the color cycle to 'magma'
num_colors = 12
colors = plt.cm.magma([0, 0.45, 0.6, 0.75, 0.85])
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

# Find the unique Spec IDs in the DataFrame
spec_ids = combined_df['Spec ID'].unique()

# Filter out the reference spectrum (Spec ID 0)
a_ref  = scale_factors[0]


font_size = 12
# Set the font size globally
plt.rcParams.update({
    'font.size': font_size,       # Set the base font size
    'axes.titlesize': 14,  # Set the font size of the axes title
    'axes.labelsize': font_size,  # Set the font size of the axes labels
    'xtick.labelsize': font_size, # Set the font size of the x tick labels
    'ytick.labelsize': 10, # Set the font size of the y tick labels
    'legend.fontsize': font_size, # Set the font size of the legend
    'figure.titlesize': 14 # Set the font size of the figure title
})


# Set up figure with custom layout
fig = plt.figure(figsize=(12, 7), dpi=600)

# Create a GridSpec with 8 rows (for 4 pairs of k-P and error plots) and 2 columns
gs = gridspec.GridSpec(5, 2, height_ratios=[2, 0.6, 0.5, 2, 0.6], hspace=0.0, wspace=0.1)

legend_added = False

# Define marker and marker size mappings for different methods
marker_dict = {
    'nbody': 's',  # Example method names
    'gramfe': 's',
    'finite_difference': '^',
    'fluid': '^',
    'spectral': 's',
}
size_dict = {
    'nbody': 3,
    'gramfe': 5,
    'finite_difference': 5,
    'fluid': 4,
    'spectral': 5,
}

lw_dict = {
    'nbody': 1.5,
    'gramfe': 1.5,
    'finite_difference': 1.5,
    'fluid': 1.5,
    'spectral': 1.5,
}
alpha_dict = {
    'fluid': 0.7,
    'nbody': 0.8,
    'gramfe': 0.8,
    'finite_difference': 0.8,
    'spectral': 1.0,
}
method_name_dict = {
    'finite_difference': 'Finite Difference',
    'fluid': 'Madelung',
    'gramfe': 'FC-Gram',
    'nbody': 'N-Body',
    'spectral': 'FFT',
}
zorder_dict = {
    'finite_difference': 'Finite Difference',
    'fluid': 10,
    'gramfe': 'FC-Gram',
    'nbody': 5,
    'spectral': 0,
}



# Function to format resolution label
def format_resolution_label(resolution):
    # Extracts number from resolution string (e.g., 'n_128') and formats it
    number = resolution
    return f"$N = {number}$"


for i, spec_id in enumerate([5, 8, 10, 18]):
    a = scale_factors[spec_id]
    row = int(i / 2)
    col = (i % 2 != 0)
    ax1 = fig.add_subplot(gs[row + 2 * row, col])  # k-P plot for this pair
    ax2 = fig.add_subplot(gs[row + 2 * row + 1, col], sharex=ax1)  # Error plot directly below ax1

    # Remove x-tick labels for ax1, since ax2 will share the x-axis
    plt.setp(ax1.get_xticklabels(), visible=False)


    for (method, resolution), group_df in combined_df.groupby(['Method', 'N_value']):

        df = group_df[group_df['Spec ID'] == spec_id]
        df_ref = group_df[group_df['Spec ID'] == 0]

        if df.empty or df_ref.empty:
            continue

        if method == "finite_difference":
            continue
        if method == "gramfe":# and not resolution == "n_2048":
            continue
        if method == "fluid" and not resolution == 64:
            continue
        if method == "spectral" and not (resolution == 512 or resolution == 2048 or resolution ==1024):
            continue
        if method == "nbody" and not resolution == 512:
            continue

        k = df["k"].values
        P = df["Power"].values
        P_ref = (a / a_ref) ** 2 * df_ref['Power'].values

        # Apply method name mapping
        method_label = method_name_dict.get(method, method)  # Default to raw method name if not found
        # Format resolution label
        resolution_label = format_resolution_label(resolution)

        label = f"{method_label} {resolution_label}"+r"$^3$"
        fillstyle="full"
        # Select marker and size based on the method
        marker = marker_dict.get(method, 'o')  # Default to 'o' if method not in dict
        if method == "spectral":
            marker_d2 = {
                512: "o",
                1024: "o",
                2048: "o"
            }
            fillstyle_d2 = {
                512: "full",
                1024: "bottom",
                2048: "none"
            }
            marker = marker_d2[resolution]
            fillstyle = fillstyle_d2[resolution]
        size = size_dict.get(method, 4)  # Default to 4 if method not in dict
        lw = lw_dict.get(method, 4)  # Default to 4 if method not in dict
        alpha = alpha_dict.get(method, 4)  # Default to 4 if method not in dict
        zorder = zorder_dict.get(method, 4)  # Default to 4 if method not in dict


        # Calculate logarithmically spaced indices based on the length of k, avoiding log10(0)
        indices = np.geomspace(1, len(k), num=15, dtype=int) - 1  # Subtract 1 to adjust for index starting at 0

        # Ensure unique indices if overlapping occurs due to rounding
        unique_log_indices = np.unique(indices)

        # Plot the entire dataset with lines
        line, = ax1.plot(k[unique_log_indices], k[unique_log_indices]**3 * P[unique_log_indices], linestyle='-', lw = lw, alpha=alpha, zorder=zorder)
        # Overlay markers only on the subset of points
        ax1.plot(k[unique_log_indices], k[unique_log_indices]**3 * P[unique_log_indices], label=label, linestyle='', marker=marker, fillstyle=fillstyle, markersize=size, alpha=alpha, c=line.get_color(), zorder=zorder)


        error = np.abs(P - P_ref) / P_ref
        # Plot the entire dataset with lines
        ax2.plot(k[unique_log_indices], error[unique_log_indices], linestyle='-', marker=marker, fillstyle=fillstyle, markersize=size, alpha=alpha, lw = lw, c=line.get_color(), zorder=zorder)


        #ax1.plot(k, P, label=label, linestyle='--', marker='o', markersize=4, alpha=0.7) #linestyle='--', marker=marker, markersize=size, )
        if method == "spectral" and resolution == 2048:
            line = ax1.plot(k, k**3 * P_ref, c="k", lw=1.0, alpha=0.6, ls="--", zorder=10, label="Linear Theory")[0]

            # Add a white border to the line using patheffects
            line.set_path_effects([PathEffects.Stroke(linewidth=2.0, foreground='white'), PathEffects.Normal()])


    if not legend_added:
        ax1.legend(loc='lower left', fontsize=9, ncol=1)
        legend_added = True

    ax1.set_title(f'z = {redshifts[spec_id]}', fontsize=14)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlim(0.5, 100)  # Adjust as needed
    #ax1.grid(True, which="both", ls="--")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_ylim(1e-5, 100)  # Adjust as needed
    ax2.grid(True, which="both", axis="y", ls="--")

    # Enable ticks on the top and right axes
    ax1.tick_params(axis='x', which='both', direction='in', top=True)  # Enable top ticks
    ax1.tick_params(axis='y', which='both', direction='in', right=True)  # Enable right ticks
    ax2.tick_params(axis='x', which='both', direction='in', top=True)  # Enable top ticks
    ax2.tick_params(axis='y', which='both', direction='in', right=True)  # Enable right ticks

    if row == 0:
        ax1.set_ylim(1e-9, 1e0)  # Adjust as needed
    if row == 1:  # Only set xlabel for the last row of error plots
        ax1.set_ylim(1e-6, 1e2)  # Adjust as needed
        ax2.set_xlabel(r'$k$ $h^{-1}$ Mpc')

    if col == 0:
        ax1.set_ylabel(r'$k^3$ $P(k)$',labelpad=20)
        ax2.set_ylabel(r'$\frac{|P(k) - P_{lin}(k)|}{P_{lin}(k)}$', labelpad=20)
        # Adjust y-axis label position
        ax1.yaxis.set_label_coords(-0.1, 0.5)
        ax2.yaxis.set_label_coords(-0.1, 0.5)

        # Set the locations for the y-ticks
        ax2.set_yticks([1e-3, 1])

        # Set the labels for the y-ticks
        ax2.set_yticklabels([r"$0.1\%$", r"$100\%$"])
    if col == 1:  # Remove y-ticks and labels for the second column
        ax1.set_yticklabels([])
        ax2.set_yticklabels([])
        ax2.set_yticks([1e-3, 1])
        ax2.grid(True, which='both', axis="y", linestyle='--')


# Remove padding between k-P and error plots while keeping padding between sets
gs.update(hspace=0.07)  # Adjust space between the k-P/error plot groups

plt.tight_layout()

plt.savefig('figure_15.pdf', bbox_inches='tight')  # Save the plot as a PNG file
plt.close()
