import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import gridspec
from matplotlib.lines import Line2D
import pandas as pd



# Set the color cycle to use colors from the 'magma' colormap
def set_cycle(num_colors = 10, start=0):
    colors = plt.cm.magma(np.linspace(0, 1, num_colors)[start:])
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)
    return colors

plot_dpi = 600
plot_width = 12
plot_height_single = 3
plot_height_double = 6


colors = set_cycle(5, 1)
fns = [
    'data/figure_13/interpolation_plane_wave.csv',
    'data/figure_13/interpolation_vortex_pair.csv',
]


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

for i, fn in enumerate(fns):
    ax = [ax1, ax2][i]

    # Load the CSV file into a DataFrame
    df = pd.read_csv(fn)

    # Filter rows where OPT__FLU_INT_SCHEME equals OPT__REF_FLU_INT_SCHEME
    filtered_df = df[df['OPT__FLU_INT_SCHEME'] == df['OPT__REF_FLU_INT_SCHEME']]


    # Plot Mean_L1_Error as a function of PWave_NWavelength for different OPT__INT_PHASE options
    # Group by OPT__FLU_INT_SCHEME and OPT__INT_PHASE to plot each option as a separate line
    for (scheme, phase), group_data in filtered_df.groupby(['OPT__FLU_INT_SCHEME', 'OPT__INT_PHASE']):
        # Set line styles and labels based on phase
        if phase == 0:
            marker    = 'o'
        elif scheme == 8:
            marker = 's'
        else:
            marker = 'x'

        fillstyle = 'none'

        label_suffix = " - 1" if phase == 0 else " - 2"


        # Set labels based on scheme
        if scheme == 4:
            label = "2nd-order Poly."
            c     = colors[0]
            alpha = 0.6
            lw    = 1.5

            linestyle = '-'
        elif scheme == 6:
            label = "4th-order Poly."
            c     = colors[1]
            alpha = 0.8
            lw    = 1.5
            linestyle = '-'
        elif scheme == 8:
            label = "FC-Gram Matrix"
            c     = colors[2]
            alpha = 1.0
            lw    = 1.5
            linestyle = '-'

        #label += label_suffix

        if i == 0:
            x = 512 / group_data['PWave_NWavelength']
        elif i == 1:
            x = 64 / group_data["VorPairLin_kx"]

        y = group_data['Mean_L1_Error']

        ax.plot(x, y, c=c, alpha = alpha, lw = lw, label=label if phase == 0 else None, linestyle=linestyle, fillstyle=fillstyle, markersize=8)
        ax.plot(x, y, c=c, alpha = alpha, lw = lw, marker=marker, markevery=2, linestyle=linestyle, fillstyle=fillstyle, markersize=8)


        # Plotting reference lines based on label
        n_start = 15

        ls = ['2nd-order Poly.', '4th-order Poly.', 'FC-Gram Matrix']
        cs = [-3, -5, -12]
        if i == 0:
            ss = [12, 12, 12]
        elif i == 1:
            ss = [10, 10, 8]


        for i in range(len(ls)):
            c = cs[i]
            l = ls[i]
            s = ss[i]
            if label == l and phase == 0:
                y_ref = y.iloc[s] * (x.iloc[:s+1]/x.iloc[s])**c
                ax.plot(x.iloc[:s+1], y_ref, 'k--', alpha = 0.6, lw = 1.0)

    # Customize plot
    ax.set_xlabel('Points per Wavelength')
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim([5e-17, 1e1])
    # Ticks on the inside and on all sides of the plot
    ax.tick_params(direction='in', which='both', top=True, right=True)
ax1.set_ylabel('$L_1$ Error')
ax2.set_yticklabels([])

# Define and place the legend
#accuracy_heading = [Line2D([0], [0], color='none', label='Accuracy')]
accuracy_handles = [Line2D([0], [0], color='k', ls="--", lw=1.5, label=r'$x^{-4}, x^{-6}, x^{-13}$')]
#legend_ax.legend(handles = scheme_handles + accuracy_heading + accuracy_handles, loc='upper left', title="Numerical schemes", handletextpad=0.5, handlelength=2, fancybox=True)

# Assume ax1 and ax2 are already set up as in previous examples
scheme_handles, scheme_labels = ax1.get_legend_handles_labels()

#for h in scheme_handles: h.set_marker("")

# First legend for "Numerical schemes"
legend1 = legend_ax.legend(scheme_handles, scheme_labels, loc='upper left', bbox_to_anchor=(0.05, 1), title="Numerical schemes", handletextpad=0.5, handlelength=2, fancybox=True)

# Create legend for line styles
line_style_handles = [
    Line2D([0], [0], color='black', ls='', marker = "o", lw=1.0, label='Real/Imag', fillstyle="none"),
    Line2D([0], [0], color='black', ls='', marker = "x", lw=1.0, label='Dens/Phas', fillstyle="none"),
    Line2D([0], [0], color='black', ls='', marker = "s", lw=1.0, label='Adaptive',  fillstyle="none"),
]
legend2 = legend_ax.legend(handles = line_style_handles, loc='upper left', bbox_to_anchor=(0.05, 0.56), title="Mode", handletextpad=0.5, handlelength=2, fancybox=True)


# Second legend for "Accuracy"
accuracy_handles = [Line2D([0], [0], color='k', ls="--", lw=1.0, label=r'$x^{-3}, x^{-5}, x^{-12}$')]
legend3 = legend_ax.legend(handles = accuracy_handles, loc='upper left', bbox_to_anchor=(0.05, 0.1), title="Accuracy", handletextpad=0.5, handlelength=2, fancybox=True)

legend_ax.add_artist(legend1)
legend_ax.add_artist(legend2)
# Disable axis of legend_ax
legend_ax.axis('off')
ax1.set_title("Plane Wave")
ax2.set_title("Traveling Vortex Pair")

ax1.set_xlim([1, 530])
ax1.set_xticks([1, 2, 3, 10, 30, 100, 300], ['1', '2', '3', '10', '30', '100', '300'])

ax2.set_xlim([1, 64])
ax2.set_xticks([1, 2, 3, 10, 30], ['1', '2', '3', '10', '30'])

# Show or save the plot
plt.savefig('figures/figure_13.pdf', bbox_inches='tight')  # Save the plot as a PNG file
plt.close()
