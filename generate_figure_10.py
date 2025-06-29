import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import os
import glob
from matplotlib.gridspec import GridSpec

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


set_cycle(8)

# Define labels for Gaussian data
labels_gaussian = {
    'base_fft_NX0_TOT_X': 'FFT',
    'fd2_no_cons_NX0_TOT_X': '4th-order FD',
    'fd4_no_cons_NX0_TOT_X': '6th-order FD',
    'gramfe_fft_NX0_TOT_X': 'FC-Gram FFT',
    'gramfe_matmul_NX0_TOT_X': 'FC-Gram Matrix',
    'fluid_NX0_TOT_X': 'Madelung'
}

# Define labels for Plane Wave data
labels_planewave = {
    'base_fft_NX0_TOT_X': 'FFT',
    'fd2_NX0_TOT_X': '4th-order FD',
    'fd4_NX0_TOT_X': '6th-order FD',
    'gramfe_fft_NX0_TOT_X': 'FC-Gram FFT',
    'gramfe_matmul_NX0_TOT_X': 'FC-Gram Matrix',
    'fluid_NX0_TOT_X': 'Madelung'
}

markers = {
    'FFT' : "",
    '4th-order FD': ".",
    '6th-order FD': "",
    'FC-Gram FFT': "",
    'FC-Gram Matrix': ".",
    'Madelung': "."
}
# Function to read data and assign labels
def prepare_data(file_path, labels_dict):
    df = pd.read_csv(file_path)
    df['Label'] = df['RunName'].map(labels_dict)
    return df

# Load data
df_gaussian  = prepare_data('data/figure_10/accuracy_gaussian_fixed_wavelength.csv', labels_gaussian)
df_planewave = prepare_data('data/figure_10/accuracy_planewave_fixed_wavelength.csv', labels_planewave)

# Set desired order of labels
desired_order = ['4th-order FD', '6th-order FD', 'FC-Gram Matrix', 'FC-Gram FFT', 'Madelung', 'FFT']
df_gaussian['Label'] = pd.Categorical(df_gaussian['Label'], categories=desired_order, ordered=True)
df_planewave['Label'] = pd.Categorical(df_planewave['Label'], categories=desired_order, ordered=True)

# Sort data
df_gaussian.sort_values('Resolution', inplace=True)
df_planewave.sort_values('Resolution', inplace=True)




font_size = 12
# Set the font size globally
plt.rcParams.update({
    'font.size': font_size,       # Set the base font size
    'axes.titlesize': font_size,  # Set the font size of the axes title
    'axes.labelsize': font_size,  # Set the font size of the axes labels
    'xtick.labelsize': font_size, # Set the font size of the x tick labels
    'ytick.labelsize': font_size, # Set the font size of the y tick labels
    'legend.fontsize': 10, # Set the font size of the legend
    'figure.titlesize': font_size # Set the font size of the figure title
})


# High-resolution plot setup
fig = plt.figure(figsize=(12, 3), dpi=600)

gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.5], wspace=0.0)

# Assign axes for the plots
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
legend_ax = fig.add_subplot(gs[2])
legend_ax.axis('off')

# Plot data for each label
for i, (test, df) in enumerate(zip(["Plane wave", "Gaussian"], [df_planewave, df_gaussian])):
    ax = [ax1, ax2][i]

    for label, group_df in df.groupby('Label'):

        if test == "Plane wave":
            x = group_df['Resolution']
            y = 0.5 * (group_df['Error(Dens)'] + group_df['Error(Real)'])

            line, = ax.plot(x, y, label=f'{label}', marker=markers[label], fillstyle="none", linestyle='-', lw=1.8, alpha = 0.8)

        elif test == "Gaussian":
            x = group_df['Resolution']
            y = 0.5 * (group_df['Error(Dens)'] + group_df['Error(Real)'])

            line, = ax.plot(x, y, label=f'{label}', marker=markers[label], fillstyle="none", linestyle='-', lw=1.8, alpha = 0.8)#,, lw=2)

        # Plotting reference lines based on label
        n_start = 15

        ls = ['4th-order FD', '6th-order FD', 'FC-Gram FFT']
        cs = [-2, -4, -11]
        if test == "Gaussian":
            ss = [5, 4, 3]
        else:
            ss = [3, 3, 3]


        for i in range(len(ls)):
            c = cs[i]
            l = ls[i]
            s = ss[i]
            if label == l:
                y_ref = y.iloc[s] * (x/x.iloc[s])**c
                ax.plot(x, y_ref, 'k--', alpha = 0.6, lw = 1.0)


    # Customize plot
    ax.set_xlabel('Points $N$')
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim([5e-16, 1e1])
    ax.set_xticks([2**i for i in range(4, 12)],[r"$2^{" + f"{i}" + r"}$" for i in range(4, 12)])
    # Ticks on the inside and on all sides of the plot
    ax.tick_params(direction='in', which='both', top=True, right=True)

ax1.set_ylabel('$L_1$ Error')
ax2.set_yticklabels([])


# Define and place the legend
scheme_handles, scheme_labels = ax1.get_legend_handles_labels()

accuracy_handles = [Line2D([0], [0], color='k', ls="--", lw=0.5, label=r'$x^{-4}, x^{-6}, x^{-13}$')]

scheme_handles, scheme_labels = ax1.get_legend_handles_labels()

# First legend for "Numerical schemes"
legend1 = legend_ax.legend(scheme_handles, scheme_labels, loc='upper left', bbox_to_anchor=(0.05, 1), title="Numerical schemes", handletextpad=0.5, handlelength=2, fancybox=True)

# Second legend for "Accuracy"
accuracy_handles = [Line2D([0], [0], color='k', ls="--", lw=1.0, label=r'$x^{-2}, x^{-4}, x^{-11}$')]
legend2 = legend_ax.legend(handles = accuracy_handles, loc='upper left', bbox_to_anchor=(0.05, 0.3), title="Accuracy", handletextpad=0.5, handlelength=2, fancybox=True)

legend_ax.add_artist(legend1)

# Disable axis of legend_ax
legend_ax.axis('off')
ax1.set_title("Plane Wave")
ax2.set_title("Traveling Gaussian")

# Show or save the plot
plt.savefig('figures/figure_10.pdf', bbox_inches='tight')  # Save the plot as a PNG file
plt.close()