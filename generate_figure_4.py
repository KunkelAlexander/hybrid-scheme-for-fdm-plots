import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import pandas as pd

# === Global Plotting Parameters ===
plot_dpi = 600
plot_width = 12
plot_height_single = 3
plot_height_double = 6
font_size = 11

# Update global matplotlib settings
plt.rcParams.update({
    'font.size': font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    'legend.fontsize': 9,
    'figure.titlesize': font_size
})

# === Utility: Color Cycle Setter ===
def set_cycle(num_colors=10, start=0):
    """
    Set the matplotlib color cycle using the magma colormap.

    Parameters:
    - num_colors (int): Total number of colors to use from colormap.
    - start (int): Index to start sampling the colormap.

    Returns:
    - colors (array): Array of RGBA color values.
    """
    colors = plt.cm.magma(np.linspace(0, 1, num_colors)[start:])
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors)
    return colors

set_cycle()  # Apply custom color cycle

# === Parameters ===
time_step = 1
ghostBoundarySize = 8
totalN = 32 + 2 * ghostBoundarySize
nDelta = 14

# === Simulation Data ===
original_psi = np.array([-0.38402867+0.92332117j, -0.56203076+0.82711633j,
        -0.71702325+0.69704925j, -0.84266073+0.53844489j,
        -0.93379957+0.35779654j, -0.98670855+0.16249995j,
        -0.99922157-0.03944941j, -0.97082634-0.2397837j ,
        -0.90268536-0.43030122j, -0.79758834-0.60320215j,
        -0.65983796-0.75140792j, -0.49507374-0.86885096j,
        -0.31004113-0.95072314j, -0.11231541-0.99367261j,
         0.09000852-0.995941j  ,  0.28864748-0.95743544j,
         0.47546919-0.87973237j,  0.64282513-0.76601295j,
         0.78386374-0.62093288j,  0.89281087-0.45043173j,
         0.96520622-0.26148986j,  0.99808592-0.06184256j,
         0.99010386+0.14033657j,  0.94158683+0.33677032j,
         0.85452112+0.51941664j,  0.73247122+0.68079799j,
         0.58043387+0.81430739j,  0.40463348+0.91447895j,
         0.21226735+0.97721163j,  0.01121097+0.99993716j,
        -0.19030439+0.98172513j, -0.38402867+0.92332117j,
        -0.56203076+0.82711633j, -0.71702325+0.69704925j,
        -0.84266073+0.53844489j, -0.93379957+0.35779654j,
        -0.98670855+0.16249995j, -0.99922157-0.03944941j,
        -0.97082634-0.2397837j , -0.90268536-0.43030122j,
        -0.79758834-0.60320215j, -0.65983796-0.75140792j,
        -0.49507374-0.86885096j, -0.31004113-0.95072314j,
        -0.11231541-0.99367261j,  0.09000852-0.995941j  ,
         0.28864748-0.95743544j,  0.47546919-0.87973237j])

fdp1 = np.array([-0.38402867+0.92332117j, -0.56203076+0.82711633j,
        -0.71702325+0.69704925j, -0.84266073+0.53844489j,
        -0.93379957+0.35779654j, -0.98670855+0.16249995j,
        -0.99922157-0.03944941j, -0.97082634-0.2397837j ,
        -0.90268536-0.43030122j, -0.79758834-0.60320215j,
        -0.65983796-0.75140792j, -0.49507374-0.86885096j,
        -0.31004113-0.95072314j, -0.11231541-0.99367261j,
         0.09000852-0.995941j  ,  0.28864748-0.95743544j,
         0.47546919-0.87973237j,  0.64282513-0.76601295j,
         0.78386374-0.62093288j,  0.89281087-0.45043173j,
         0.96520622-0.26148986j,  0.99808592-0.06184256j,
         0.99010386+0.14033657j,  0.94158683+0.33677032j,
         0.85452112+0.51941664j,  0.73247122+0.68079799j,
         0.58043387+0.81430739j,  0.40463348+0.91447895j,
         0.21226735+0.97721163j,  0.01121097+0.99993716j,
        -0.19030439+0.98172513j, -0.38402867+0.92332117j,
        -0.56203076+0.82711633j, -0.71702325+0.69704925j,
        -0.84266073+0.53844489j, -0.93379957+0.35779654j,
        -0.98670855+0.16249995j, -0.99922157-0.03944941j,
        -0.97082634-0.2397837j , -0.90268536-0.43030122j,
        -0.79758834-0.60320215j, -0.65983796-0.75140792j,
        -0.49507374-0.86885096j, -0.31004113-0.95072314j,
        -0.11231541-0.99367261j,  0.09000852-0.995941j  ,
         0.28864748-0.95743544j,  0.47546919-0.87973237j,
         0.64282513-0.76601295j,  0.78386374-0.62093288j,
         0.89281086-0.45043175j,  0.96520616-0.26149j   ,
         0.99808593-0.06184367j,  0.9901071 +0.14032705j,
         0.94162209+0.33669917j,  0.85476269+0.51899314j,
         0.73370414+0.67879838j,  0.58541491+0.80669346j,
         0.42111297+0.89065441j,  0.25797858+0.91491994j,
         0.11947759+0.86194808j,  0.03203365+0.71980293j,
         0.01731584+0.49345965j,  0.083017  +0.21300108j,
         0.21776116-0.0696692j ,  0.39426079-0.29730598j,
         0.57935914-0.4283901j ,  0.74484332-0.4499572j ,
         0.87320407-0.3757102j ,  0.95707509-0.23369344j,
         0.99530739-0.05318254j,  0.98932668+0.14255173j,
         0.94140988+0.33722393j,  0.85448803+0.51948857j,
         0.73246586+0.68080643j,  0.58043301+0.81430807j,
         0.40463333+0.91447898j,  0.21226732+0.97721163j,
         0.01121097+0.99993715j, -0.19030439+0.98172513j])


fdp2 = np.array([-0.3657294 +0.9307212j , -0.5455957 +0.8380485j ,
        -0.70312524+0.711066j  , -0.83186877+0.55497235j,
        -0.9265555 +0.3761581j , -0.9833089 +0.18194388j,
        -0.99980557-0.01971916j, -0.97537005-0.22057489j,
        -0.91100276-0.41240025j, -0.8093389 -0.5873419j ,
        -0.67454064-0.7382377j , -0.5121266 -0.85890996j,
        -0.32874605-0.9444184j , -0.13190657-0.99126214j,
         0.07033317-0.99752355j,  0.26969346-0.96294624j,
         0.45801246-0.88894576j,  0.6275804 -0.77855176j,
         0.7714551 -0.63628376j,  0.8837463 -0.46796623j,
         0.9598569 -0.2804901j ,  0.99667084-0.08153066j,
         0.9926809 +0.12076664j,  0.94805056+0.31811973j,
         0.86460686+0.502449j  ,  0.74576604+0.6662079j ,
         0.5963935 +0.8026922j ,  0.42260456+0.9063142j ,
         0.23151413+0.97283155j,  0.03094547+0.9995211j ,
        -0.1708901 +0.9852901j , -0.3657294 +0.9307212j ,
        -0.5455957 +0.8380485j , -0.70312524+0.711066j  ,
        -0.83186877+0.55497235j, -0.9265555 +0.3761581j ,
        -0.9833089 +0.18194388j, -0.99980557-0.01971916j,
        -0.97537005-0.22057489j, -0.91100276-0.41240025j,
        -0.8093389 -0.5873419j , -0.67454064-0.7382377j ,
        -0.5121266 -0.85890996j, -0.32874605-0.9444184j ,
        -0.13190657-0.99126214j,  0.07033317-0.99752355j,
         0.26969346-0.96294624j,  0.45801246-0.88894576j,
         0.6275804 -0.77855176j,  0.7714551 -0.63628376j,
         0.8837464 -0.46796626j,  0.9598568 -0.28049013j,
         0.9966706 -0.08152936j,  0.9926889 +0.12077898j,
         0.94814485+0.31817308j,  0.865206  +0.50253844j,
         0.74844736+0.6658431j ,  0.60570115+0.7992672j ,
         0.44901457+0.89133173j,  0.29496285+0.92614454j,
         0.1634874 +0.8846926j ,  0.07501172+0.75212103j,
         0.04633756+0.5299495j ,  0.08618764+0.24599305j,
         0.19151713-0.04739938j,  0.3462932 -0.28969553j,
         0.52446645-0.4348569j ,  0.6969885 -0.46648642j,
         0.83977556-0.3973267j ,  0.93846446-0.25657755j,
         0.9881675 -0.07534789j,  0.9899763 +0.12180039j,
         0.94738704+0.31814328j,  0.8644895 +0.5024028j ,
         0.7457532 +0.6661927j ,  0.5963931 +0.8026896j ,
         0.42260465+0.9063139j ,  0.23151414+0.9728315j ,
         0.03094547+0.9995211j , -0.1708901 +0.9852901j ], dtype=np.complex64)

psi =  np.array([-0.3657294 +0.9307212j , -0.5455957 +0.8380485j ,
        -0.70312524+0.711066j  , -0.83186877+0.55497235j,
        -0.9265555 +0.3761581j , -0.9833089 +0.18194388j,
        -0.99980557-0.01971916j, -0.97537005-0.22057489j,
        -0.91100276-0.41240025j, -0.8093389 -0.5873419j ,
        -0.67454064-0.7382377j , -0.5121266 -0.85890996j,
        -0.32874605-0.9444184j , -0.13190657-0.99126214j,
         0.07033317-0.99752355j,  0.26969346-0.96294624j,
         0.45801246-0.88894576j,  0.6275804 -0.77855176j,
         0.7714551 -0.63628376j,  0.8837463 -0.46796623j,
         0.9598569 -0.2804901j ,  0.99667084-0.08153066j,
         0.9926809 +0.12076664j,  0.94805056+0.31811973j,
         0.86460686+0.502449j  ,  0.74576604+0.6662079j ,
         0.5963935 +0.8026922j ,  0.42260456+0.9063142j ,
         0.23151413+0.97283155j,  0.03094547+0.9995211j ,
        -0.1708901 +0.9852901j , -0.3657294 +0.9307212j ,
        -0.5455957 +0.8380485j , -0.70312524+0.711066j  ,
        -0.83186877+0.55497235j, -0.9265555 +0.3761581j ,
        -0.9833089 +0.18194388j, -0.99980557-0.01971916j,
        -0.97537005-0.22057489j, -0.91100276-0.41240025j,
        -0.8093389 -0.5873419j , -0.67454064-0.7382377j ,
        -0.5121266 -0.85890996j, -0.32874605-0.9444184j ,
        -0.13190657-0.99126214j,  0.07033317-0.99752355j,
         0.26969346-0.96294624j,  0.45801246-0.88894576j], dtype=np.complex64)


ghostBoundarySize = 8
totalN = 32 + 2 * ghostBoundarySize
nDelta = 14

def plot_psi(ax, psi, title, highlight, initial, original=None):
    x = np.arange(len(psi))
    c1, c2 = plt.cm.magma(np.array([0.3, 0.7]))

    if highlight == "ghost":
        ax.plot(x, np.real(psi), c=c1, label='Real' if initial else None)
        ax.plot(x, np.imag(psi), c=c2, label='Imag' if initial else None)

    elif highlight == "boundary":
        y1     = np.real(psi)
        y2     = np.imag(psi)
        y1_len = len(y1)
        y2_len = len(y2)
        y1     = np.array([y1, y1]).flatten()
        y2     = np.array([y2, y2]).flatten()
        ax.plot(x[:totalN+1], y1[:totalN+1], c=c1, label='Real' if initial else None)
        ax.plot(x[:totalN+1], y2[:totalN+1], c=c2, label='Imag' if initial else None)
        ax.plot(x[totalN+1:], y1[totalN+1:y1_len], c=c1, ls = "dashed")
        ax.plot(x[totalN+1:], y2[totalN+1:y2_len], c=c2, ls = "dashed")
        ax.plot(x[:totalN+1] + x[-1], y1[y1_len - 1:y1_len - 1 +totalN+1], c=c1)
        ax.plot(x[:totalN+1] + x[-1], y2[y2_len - 1:y2_len - 1 +totalN+1], c=c2)

    elif highlight == "evolution":
        y1     = np.roll(np.real(psi), 3)
        y2     = np.roll(np.imag(psi), 3)
        y1_len = len(y1)
        y2_len = len(y2)
        y1     = np.array([y1, y1]).flatten()
        y2     = np.array([y2, y2]).flatten()
        ax.plot(x[:totalN+1], y1[:totalN+1], c=c1)
        ax.plot(x[:totalN+1], y2[:totalN+1], c=c2)
        ax.plot(x[totalN+1:], y1[totalN+1:y1_len], c=c1, ls = "dashed")
        ax.plot(x[totalN+1:], y2[totalN+1:y2_len], c=c2, ls = "dashed")
        ax.plot(x[:totalN+1] + x[-1], y1[y1_len - 1:y1_len - 1 +totalN+1], c=c1)
        ax.plot(x[:totalN+1] + x[-1], y2[y2_len - 1:y2_len - 1 +totalN+1], c=c2)
        y1 = np.real(psi)
        y2 = np.imag(psi)
        y1_len = len(y1)
        y2_len = len(y2)
        y1     = np.array([y1, y1]).flatten()
        y2     = np.array([y2, y2]).flatten()
        ax.plot(x[:totalN+1], y1[:totalN+1], alpha=0.2, c=c1)
        ax.plot(x[:totalN+1], y2[:totalN+1], alpha=0.2, c=c2)
        ax.plot(x[totalN+1:], y1[totalN+1:y1_len], alpha=0.2, c=c1, ls = "dashed")
        ax.plot(x[totalN+1:], y2[totalN+1:y2_len], alpha=0.2, c=c2, ls = "dashed")
        ax.plot(x[:totalN+1] + x[-1], y1[y1_len - 1:y1_len - 1 +totalN+1], alpha=0.2, c=c1)
        ax.plot(x[:totalN+1] + x[-1], y2[y2_len - 1:y2_len - 1 +totalN+1], alpha=0.2, c=c2)

    else:
        ax.plot(x[ghostBoundarySize:-ghostBoundarySize], np.real(psi)[ghostBoundarySize:-ghostBoundarySize], c=c1, label='Real' if initial else None)
        ax.plot(x[ghostBoundarySize:-ghostBoundarySize], np.imag(psi)[ghostBoundarySize:-ghostBoundarySize], c=c2, label='Imag' if initial else None)


    def add_arrow_annotation(ax, l, r, ht, annotation):
        ax.annotate("", xy=(l-1, ht), xytext=(r+1, ht), textcoords=ax.transData, arrowprops=dict(arrowstyle='<->', lw=0.2))
        bbox=dict(fc="white", ec="none")
        ax.text((r-l)/2+l + 1, ht+0.2, annotation, ha="center", va="center", fontsize=7)


    if highlight == 'ghost':

        # Physical domain arrow and label
        add_arrow_annotation(ax, ghostBoundarySize, totalN - ghostBoundarySize, 1.4, r"$N$")
        add_arrow_annotation(ax, 0, ghostBoundarySize, 1.4, r"$N_{ghost}$")
        add_arrow_annotation(ax, totalN - ghostBoundarySize, totalN, 1.4, r"$N_{ghost}$")


        ax.fill_betweenx([-1.3, 1.3], 0, ghostBoundarySize, color='grey', alpha=0.3)
        ax.fill_betweenx([-1.3, 1.3], totalN - ghostBoundarySize, totalN, color='grey', alpha=0.3, label="Ghost boundary")
        ax.annotate("Physical\ndomain", xy=(0.28, 0.9), xycoords='axes fraction', fontsize=8, fontweight="bold",
                    ha='center', va='center', annotation_clip=True)
        ax.annotate("Unphysical\ndomain", xy=(0.76, 0.9), xycoords='axes fraction', fontsize=8, fontweight="bold",
                    ha='center', va='center', annotation_clip=True)
        ax.legend(loc="lower right", fontsize=8)

    elif highlight == 'boundary':

        add_arrow_annotation(ax, 0, nDelta, 1.4, r"$N_{boundary}$")
        add_arrow_annotation(ax, totalN - nDelta, totalN, 1.4, r"$N_{boundary}$")
        add_arrow_annotation(ax, totalN, len(psi)-1, 1.4, r"$N_{ext}$")

        ax.fill_betweenx([-1.3, 1.3],0, nDelta, color='orange', alpha=0.3)
        ax.fill_betweenx([-1.3, 1.3],totalN - nDelta, totalN, color='orange', alpha=0.3, label="Expansion boundary")
        ax.fill_betweenx([-1.3, 1.3],len(psi)-1, 2*totalN, color='purple', alpha=0.2, label="Periodic")
        ax.annotate("Periodic continuation \nusing Gram polynomial expansion",
                    xy=(0.5, 0.9), xycoords='axes fraction', fontsize=8, fontweight="bold",
                    ha='center', va='center', annotation_clip=True)
        ax.legend(loc="lower right", fontsize=8)
    elif highlight == 'evolution':
        ax.annotate(r"DFT $\times$ Filter $\times \sum_n (C \cdot k^{2})^{n}/n! \times$ IDFT",
                    xy=(0.5, 0.9), xycoords='axes fraction', fontsize=8, fontweight="bold",
                    ha='center', va='center')

    elif highlight == "discard":
        ax.annotate("Discard unphysical domain\n and ghost boundary", xy=(0.5, 0.9), xycoords='axes fraction', fontsize=8, fontweight="bold",
                    ha='center', va='center')
        ax.fill_betweenx([-1.3, 1.3], 0, ghostBoundarySize, color='red', alpha=0.3)
        ax.fill_betweenx([-1.3, 1.3], totalN - ghostBoundarySize, 86, color='red', alpha=0.3, label="Discard")
        ax.legend(loc="lower right", fontsize=8)

    ax.set_title(title, fontsize=12)

fig, axs = plt.subplots(1, 4, figsize=(plot_width, plot_height_single), dpi=plot_dpi, sharey=True)

for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-1, 87])
    ax.set_ylim([-2, 2.5])

plt.subplots_adjust(wspace=0)  # Remove horizontal padding between plots

# Plot initial state
plot_psi(axs[0], original_psi, title=r'(a) $\psi(t)$', highlight='ghost', initial=True)
plot_psi(axs[1], fdp1, title='(b) FC-Gram', highlight='boundary', initial=False)

# Plot post-evolution state
plot_psi(axs[2], fdp2, title='(c) Time evolution', highlight='evolution', initial=False, original=fdp1)

# Plot next time step state
plot_psi(axs[3], psi, title=r'(d) $\psi(t+\Delta t)$', highlight='discard', initial=False)

plt.savefig("figures/figure_4.pdf", bbox_inches='tight')
plt.close()
