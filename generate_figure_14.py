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

# Load the combined data from the CSV file
data_path = 'data/figure_14/performance_test.csv' # Update this path
data = pd.read_csv(data_path)

set_cycle(9, 1)

# Define your folder to label mapping here
folder_to_label = {
    'gramfe_single_matmul_gpu': 'FC-Gram' + '\nMatrix - GPU',
    'gramfe_fft_gpu': 'FC-Gram' + '\nFFT - GPU',
    'fluid_gpu': 'Madelung - GPU',
    'fd2_gpu': '4th-order\nFD - GPU',
    'fd4_gpu': '6th-order\nFD - GPU',
    'gramfe_single_matmul_cpu': 'FC-Gram' + '\nMatrix - CPU',
    'gramfe_fft_cpu': 'FC-Gram' + '\nFFT - CPU',
    'fluid_cpu': 'Madelung - CPU',
    'fd2_cpu': '4th-order\nFD - CPU',
    'fd4_cpu': '6th-order\nFD - CPU',
}

# Map the folder names to labels
data['Label'] = data['FolderName'].map(folder_to_label)
# Calculate the average Perf_PerRank for each label
average_perf_per_label = data.groupby('Label', as_index=False)['Perf_Overall'].mean()

# Normalize to the performance of the 6th order FD and convert to percent
sixth_order_FD_avg_perf = average_perf_per_label.loc[3]["Perf_Overall"]
normalized_perf          = average_perf_per_label.copy()
#normalized_perf["Perf_Overall"] = (average_perf_per_label["Perf_Overall"] / fourth_order_FD_avg_perf) * 100





font_size = 12
# Set the font size globally
plt.rcParams.update({
    'font.size': 10,       # Set the base font size
    'axes.titlesize': font_size,  # Set the font size of the axes title
    'axes.labelsize': 10,  # Set the font size of the axes labels
    'xtick.labelsize': font_size, # Set the font size of the x tick labels
    'ytick.labelsize': font_size, # Set the font size of the y tick labels
    'legend.fontsize': 10, # Set the font size of the legend
    'figure.titlesize': font_size # Set the font size of the figure title
})

# Set the plot size
fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize=(plot_width, plot_height_single), dpi=plot_dpi)  # High-resolution plot

# Get the default color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


# Generate the bar plot manually to control colors
bars = []

for i, (label, perf) in enumerate(zip(normalized_perf['Label'], normalized_perf['Perf_Overall'])):
    if i%2 == 0:
        x_shift = -0.3
        alpha = 0.7
    else:
        x_shift = 0.3
        alpha = 1.0
    x = int(i/2) * 1.5
    perf /= 1e6
    bar = ax1.bar(x  + x_shift, height= perf, width=0.6, color=colors[int(i/2) % len(colors)], alpha=alpha)    # Add text above the bar
    bars.append(bar)
    if i%2 == 1:
        ax1.text(x=x, y=perf + 10, s=label.rstrip("- GPU"),
                ha='center', va='bottom', weight='bold')


# Add text inside the bar
for bar_container in bars:
    for bar in bar_container:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, height/2, f'{(height * 1e6 / sixth_order_FD_avg_perf) * 100:.0f}%',
                ha='center', va='center', color='white', weight='bold', fontsize=6)

ax1.set_xticks([])  # Hide x-ticks as before

# Configure primary y-axis
ax1.set_ylabel(r'$10^6$ cell updates / sec')
ax1.set_ylim(0, 850)
ax1.set_xlim(-1.2, 7.2)

ax1.tick_params(axis='x', which='both', direction='in', top=True)  # Enable top ticks
ax1.tick_params(axis='y', which='both', direction='in', right=True)  # Enable right ticks
#ax1.set_xticks(rotation=45)


# Load the combined data
data_path = 'data/figure_14/strong_scaling_gramfe.csv'  # Update this path
data = pd.read_csv(data_path)

set_cycle(5, 2)
# Define your folder to label mapping here
folder_to_label = {
    'VortexPairLinear': 'Wave-only',
    'VortexPairLinear_Hybrid': 'Hybrid'
}

# Map the folder names to labels
data['Label'] = data['Scheme'].map(folder_to_label)

# Calculate the average Perf_Overall for each FolderAttribute and Rank
df = data.groupby(['Label', 'Rank'])['Perf_Overall'].mean().reset_index()

# Step 1: Find the Perf_Overall value for "Wave-only" at Rank 1
baseline_value = df[(df['Label'] == 'Wave-only') & (df['Rank'] == 1)]['Perf_Overall'].iloc[0]

# Step 2: Normalize all Perf_Overall values by this baseline
df['Normalized_Perf'] = df['Perf_Overall'] / baseline_value

# Step 3: Optionally, express as percentage
df['Normalized_Perf_Percent'] = df['Normalized_Perf']


# Plot the average Perf_Overall for each FolderAttribute
for folder_attr, group_data in df.groupby('Label'):
    if folder_attr == "Wave-only":
        plt.plot(group_data['Rank'], group_data['Normalized_Perf_Percent'], c =plt.cm.magma(0.15), marker='o', fillstyle='none', linestyle='-', label=folder_attr)
    else:
        plt.plot(group_data['Rank'], group_data['Normalized_Perf_Percent'], c =plt.cm.magma(0.6), marker='o', linestyle='-', label=folder_attr)

x = np.arange(1, data['Rank'].max()+1)

ax2.plot(x, x, label = "Ideal scaling", c="k", ls="--", alpha =0.5)
ax2.set_xlabel('# Computing nodes')
ax2.set_ylabel('Performance')
ax2.set_ylim(0, 30)

# Set y-axis labels with '%' suffix
current_yticks = plt.gca().get_yticks()
current_yticks[0] = 1
ax2.set_yticks(current_yticks, [f'{y:.0f}' for y in current_yticks])

ax2.legend()
ax2.set_xticks([1, 2, 4, 8, 16, 24, 28])  # Assuming Rank is sequential and starts at 1
ax2.tick_params(axis='both', direction='in', top=True, right=True)

plt.savefig("figures/figure_14.pdf", bbox_inches='tight')
plt.close()
