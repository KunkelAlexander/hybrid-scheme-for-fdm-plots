import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.patheffects import Stroke, Normal
from matplotlib import patheffects

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

import numpy as np
import matplotlib.pyplot as plt
c1, c2, c3, c4, c5, c6 = plt.cm.magma(np.array(np.linspace(0, 1, 9)[1:7]))

def draw_block(ax, center, text, shadow=True, fill_color='#4f81bd'):
    """ Draws a single block with text centered inside, optionally with shadow and fill color. """
    # Create the rectangle with sharp corners
    rect = patches.Rectangle((center[0]-1.5, center[1]-0.3), 3, 0.6, linewidth=1, edgecolor='black', facecolor=fill_color, zorder=3)
    ax.add_patch(rect)

    # Optionally add shadow
    if shadow:
        shadow_rect = patches.Rectangle((center[0]-1.45, center[1]-0.35), 3, 0.6, linewidth=0, facecolor='gray', alpha=0.5, zorder=2)
        ax.add_patch(shadow_rect)

    # Add text with outlined effect
    text_effects = [patheffects.withStroke(linewidth=3, foreground='black')]
    ax.text(center[0], center[1], text, ha='center', va='center', fontweight='bold', fontsize=10, color='white', path_effects=text_effects)

def draw_arrow(ax, start, end, text=None, style='->', horizontalalignment='center', verticalalignment='center'):
    """ Draws arrows between blocks and includes text annotations with fancy arrow style. """
    arrow_style = patches.ArrowStyle.Fancy(head_length=2, head_width=2, tail_width=2.5)
    ax.annotate('', xy=end, xycoords='data', xytext=start, textcoords='data',
                arrowprops=dict(arrowstyle=arrow_style, lw=1.5, color=c1))
    if text:
        if horizontalalignment=="center":
            x = (start[0] + end[0]) / 2
        elif horizontalalignment=="left":
            x = end[0] + .2
        elif horizontalalignment=="right":
            x = start[0] - .2
        mid = (x, (start[1] + end[1]) / 2)
        text_effects = [patheffects.withStroke(linewidth=3, foreground='black')]
        ax.text(mid[0], mid[1], text, ha=horizontalalignment, va=verticalalignment, fontsize=9, color='white', fontweight='bold', path_effects=text_effects)

fig, ax = plt.subplots(figsize=(plot_width, plot_height_double), dpi = plot_dpi)

# Define positions for the blocks and labels for AMR levels
levels = ['Level 0', 'Level 1', 'Level 2', 'Level 3']
positions = [(2, 9), (2, 7), (2, 5), (2, 3)]  # x, y positions for each block

# Draw blocks for wave function solvers
solver_labels = [f'{level}: Fluid + Gravity' if i < 2 else f'{level}: Wave + Gravity' for i, level in enumerate(levels)]
for i, (pos, label) in enumerate(zip(positions, solver_labels)):
    draw_block(ax, pos, label, fill_color=c3 if i < 2 else c6)

# Draw arrows for refinement and boundary conditions
refinement_texts = ["Refinement +\nInterpolation for BC",
                    "Refinement +\nInterpolation for BC",
                    "Refinement +\nInterpolation for BC",
                    "Continue to finer levels"]
for i in range(len(positions)-1):
    draw_arrow(ax, (1.5, positions[i][1] - 0.4), (1.5, positions[i+1][1] + 0.4), refinement_texts[i], horizontalalignment='right')

restriction_texts = ["Average data",
                     "Average data",
                     "Reverse phase matching +\nAverage data",
                     "Average data",
                     "Continue to finer levels"]
for i in range(1, len(positions)):
    draw_arrow(ax, (2.5, positions[i][1] + 0.4), (2.5, positions[i-1][1] - 0.4), restriction_texts[i], style='<-', verticalalignment='bottom', horizontalalignment='left')

# Set axis limits and turn off axis visibility
ax.set_xlim(0, 4)
ax.set_ylim(2.5, 9.5)
ax.axis('off')

plt.savefig("figures/figure_9.pdf", bbox_inches='tight')
plt.close()
