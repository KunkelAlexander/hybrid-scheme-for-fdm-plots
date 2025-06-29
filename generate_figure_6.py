import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
import matplotlib.colors as mcolors
from scipy.special import j1

import colorsys

# Function to generate the complex wave function based on the Schrödinger equation
def generate_wave_function(x, y, VorPairLin_kx=1, VorPairLin_ky=1, time=0):
    VorPairRot_Omega = 90
    VorPairRot_Phase0 = 0
    VorPairRot_J1Amp = 5
    VorPairRot_BgAmp = 2
    dx = x
    dy = y
    phase = np.arctan2(dy, dx) - VorPairRot_Omega * time + VorPairRot_Phase0
    R = np.sqrt(dx ** 2 + dy ** 2)
    J1 = VorPairRot_J1Amp * j1(np.sqrt(2.0 * ELBDM_ETA * VorPairRot_Omega) * R)
    Re = VorPairRot_BgAmp - J1 * np.cos(phase)
    Im = -J1 * np.sin(phase)
    return Re + 1j * Im



font_size = 11
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

#
Time = 0.0  # Example time moment
ELBDM_ETA = 1.0  # Just an example value, adjust according to your setup

N = 512
Nh = 256
# Grid setup
x = np.linspace(-0.5, 0.5, N, endpoint=False)
y = np.linspace(-0.5, 0.5, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# Wave function computation
psi = generate_wave_function(X, Y)
Re, Im = np.real(psi), np.imag(psi)

# Density calculation
density = Re**2 + Im**2
phase   = np.arctan2(Im, Re)

plot_width = 12
plot_height_double = 5
plot_dpi = 300

fig = plt.figure(figsize=(plot_width, plot_height_double), dpi=plot_dpi)
gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1], hspace=0.25, wspace=0.0)

# Main subplot that spans all three rows in the first column
main_ax = fig.add_subplot(gs[:, 0])
main_ax.set_xticks([])
main_ax.set_yticks([])
main_ax.spines['top'].set_visible(False)
main_ax.spines['right'].set_visible(False)
main_ax.spines['bottom'].set_visible(False)
main_ax.spines['left'].set_visible(False)

# Nested GridSpec within the main subplot
nested_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=main_ax, hspace=0.15)

x0, x1 = 0.54, 0.74
y0, y1 = 0.48, 0.52

def desaturate_color(color, factor=0.5):
    """Desaturate a single color."""
    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    return colorsys.hls_to_rgb(h, l, s * factor)

def desaturate_cmap(cmap, factor=0.5):
    """Desaturate a colormap by the given factor."""
    return mcolors.LinearSegmentedColormap.from_list(
        f'{cmap.name}_desat',
        [desaturate_color(c, factor) for c in cmap(np.linspace(0, 1, cmap.N))]
    )

# Desaturate the 'magma' colormap
desaturated_magma = desaturate_cmap(plt.cm.magma, factor=1.0)
# Two subplots within the main subplot
data = [density, phase]
axes = []
cmap = "magma"
l2c = 'black'
l1c = '#FB6CFF'

ls1 = "solid"
ls2 = "dashed"

for i in range(2):
    ax = fig.add_subplot(nested_gs[i, 0])
    if i == 0:
        ax.imshow(np.log(data[i]), cmap=cmap, extent=(0, 1, 0, 1), origin="lower")
        ax.set_title(r"Density $\rho$")
    else:
        ax.imshow(data[i], cmap=cmap, extent=(0, 1, 0, 1), origin="lower")
        ax.set_title(r"Phase $S/\hbar$")
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    axes.append(ax)

    # Add zoom-in insets
    ax_inset = fig.add_axes([0.175, 0.53- i*0.41, 0.16, 0.18])
    if i == 0:
        ax_inset.imshow(np.log(data[i][int(y0*N):int(y1*N), int(x0*N):int(x1*N)]), cmap=cmap, extent=(x0, x1, y0, y1), origin="lower")
    else:
        ax_inset.imshow(data[i][int(y0*N):int(y1*N), int(x0*N):int(x1*N)], cmap=cmap, extent=(x0, x1, y0, y1), origin="lower")

    ax_inset.set_xticks([])
    ax_inset.set_yticks([])

    # Add dashed lines
    yh1 = 0.0
    yh2 = -2*0.001953125 # Determined by trial-and-error
    line_purple = ax_inset.axhline(yh1 + 0.503, color=l1c, linestyle=ls1, linewidth=1)
    line_orange = ax_inset.axhline(yh2 + 0.503, color=l2c, linestyle=ls2, linewidth=1)

    # Indicate zoom region
    ax.indicate_inset_zoom(ax_inset, edgecolor="black")



    txt = ax.annotate("Zoom-in",
                xy=(0.41, 0.08), xycoords='data',
                xytext=(0.41, 0.08), textcoords='data',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), c="w", fontsize=9)
    txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='k')])


# Create the right two columns for 1D data visualization
for i in range(2):
    for j in range(3):
        ax = fig.add_subplot(gs[j, i+1])
        if j == 0:
            ax.set_ylim([-0.1, 1.6]) if i == 0 else ax.set_ylim([-1.5*np.pi, 1.5*np.pi])
            if i == 1:
                ax.set_yticks([-np.pi, 0, np.pi], [r"$-\pi$", "$0$", r"$\pi$"])
        elif j == 1 or j == 2:
            ax.set_ylim([-1.6, 1.6]) if i == 0 else ax.set_ylim([-1.25*np.pi, 1.25*np.pi])
            if i == 1:
                ax.set_yticks([-np.pi, 0, np.pi], [r"$-\pi$", "$0$", r"$\pi$"])
        for p, y_index in enumerate([Nh, Nh-2]):
            x = np.linspace(0, 1, N)
            r = data[0][y_index, :].copy()
            s = data[1][y_index, :].copy()
            if j == 1:
                for k in range(1, len(s)):
                    if np.abs(s[k] - s[k-1]) > np.pi * 0.9:
                        s[k:] -= np.pi * np.sign(s[k] - s[k-1])
                        r[k:] *= -1
            elif j == 2:
                b = np.sqrt(r)
                r = b * np.cos(s)
                s = b * np.sin(s)
            y = r if i == 0 else s
            color = l1c if p == 0 else l2c
            alpha = 1.0 if p == 0 else 0.8
            ls    = ls1 if p == 0 else ls2
            ax.plot(x, y, lw=1.5, linestyle=ls, color=color, alpha=alpha)
        if i == 0 and j == 0:
            ax.legend(["Horizontal line through vortex", "Horizontal line slightly below vortex"])
        if i == 1:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
        label = [r"Density $\rho$", r"Phase $S/\hbar$", r"Singular gauge: $\rho'$", r"Singular gauge: $S'/\hbar$", r"Real part: $\Re(\psi)$", r"Imaginary part: $\Im(\psi)$"]
        ax.set_title(label[i + 2*j])
        ax.set_xlim(0.53, 0.75)
        ax.set_xticks([])
        if j == 2:
            ax.set_xlabel("x")  # Add x label only for the bottom plots

# Adding x and y labels for the left two panels
axes[0].set_ylabel("y")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")

plt.savefig("figure_6.pdf", bbox_inches='tight')
plt.close()
