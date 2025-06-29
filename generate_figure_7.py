import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import j1
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm

# Function to generate the complex wave function based on the Schrödinger equation
def generate_wave_function(x, y, VorPairLin_kx=1, VorPairLin_ky=1, time=0):
    VorPairRot_Omega = 90
    VorPairRot_Phase0 = 0
    VorPairRot_J1Amp = 1
    VorPairRot_BgAmp = 0
    ELBDM_ETA = 1

    dx = x
    dy = y
    phase = np.arctan2(dy, dx) - VorPairRot_Omega * time + VorPairRot_Phase0
    R = np.sqrt(dx ** 2 + dy ** 2)
    J1 = VorPairRot_J1Amp * j1(np.sqrt(2.0 * ELBDM_ETA * VorPairRot_Omega) * R)
    Re = VorPairRot_BgAmp - J1 * np.cos(phase)
    Im = -J1 * np.sin(phase)
    return Re + 1j * Im

N = 256
# Grid setup
x = np.linspace(-0.8, 0.8, N, endpoint=False)
y = np.linspace(-0.8, 0.8, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# Wave function computation
psi = generate_wave_function(X, Y)
Re, Im = np.real(psi), np.imag(psi)

# Density calculation
density = Re**2 + Im**2
phase = np.arctan2(Im, Re)

# Calculate C1
sqrt_density = np.sqrt(density)
C11 = np.diff(sqrt_density, n=2, axis=0)[:, 1:-1]
C12 = np.diff(sqrt_density, n=2, axis=1)[1:-1, :]
C1 = np.abs((C11 + C12) / 2 / sqrt_density[1:-1, 1:-1])

# Calculate C2
C21 = np.diff(phase, n=2, axis=0)[:, 1:-1]
C22 = np.diff(phase, n=2, axis=1)[1:-1, :]
C2 = np.abs((C21 + C22) / 2)

# Sum of C1_avg and C2_avg
C3 = ((C1 > 0.03) + (C2 > 1.0)) > 0

# Define the discrete colormap
cmap_discrete = mcolors.ListedColormap(['black', 'white'])
bounds_discrete = [0, 0.5, 1]
norm_discrete = mcolors.BoundaryNorm(bounds_discrete, cmap_discrete.N)

cmap = "magma"

# Create a figure with three subplots
fig = plt.figure(figsize=(12, 6), dpi=300)
gs = gridspec.GridSpec(2, 5, height_ratios=[10, 0.01], hspace=0.0, wspace=0.0)

# Function to adjust colorbar position and width
def add_colorbar(ax, im, width_shrink=0.75, pad=0.05, orientation='horizontal'):
    # Calculate new width and position
    ax_pos = ax.get_position()
    width = ax_pos.width * width_shrink
    x_position = ax_pos.x0 + (ax_pos.width - width) / 2  # Center the smaller colorbar
    cax = fig.add_axes([x_position, ax_pos.y0 - pad - 0.02, width, 0.02])
    return plt.colorbar(im, cax=cax, orientation=orientation)

# Add subplots
for i, data in enumerate([density, phase, C1, C2, C3]):
    ax = fig.add_subplot(gs[0, i])
    if i == 0:
        im = ax.imshow(data, cmap=cmap, norm=LogNorm(vmin=1e-4, vmax=1e0))
        cbar = add_colorbar(ax, im, width_shrink=0.7)
        ax.set_title(r'(a) Density $\rho$')
    elif i == 1:
        im = ax.imshow(data, cmap=cmap, vmin=-np.pi, vmax=np.pi)
        cbar = add_colorbar(ax, im, width_shrink=0.7)
        ax.set_title(r'(b) Phase $S/\hbar$')
        cbar.ax.set_xticks([-np.pi, 0, np.pi])
        cbar.ax.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
    elif i == 2:
        im = ax.imshow(data, cmap=cmap, norm=LogNorm(vmin=1e-4, vmax=1e0))
        cbar = add_colorbar(ax, im, width_shrink=0.7)
        ax.set_title(r'(c) $\mathcal{C}_1$')
    elif i == 3:
        im = ax.imshow(data, cmap=cmap, norm=LogNorm(vmin=1e-4, vmax=1e0))
        cbar = add_colorbar(ax, im, width_shrink=0.7)
        ax.set_title(r'(d) $\mathcal{C}_2$')
    elif i == 4:
        im = ax.imshow(data, cmap=cmap_discrete, norm=norm_discrete)
        cbar = add_colorbar(ax, im, width_shrink=0.7)
        ax.set_title('(e) Refinement')
        cbar.ax.set_xticks([0.20, 0.8])
        cbar.ax.set_xticklabels(["Don't refine", "Refine"])

    ax.set_xticks([])
    ax.set_yticks([])

plt.savefig("figures/figure_7.pdf", bbox_inches='tight')
plt.close()
