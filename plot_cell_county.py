import matplotlib.pyplot as plt
import numpy as np
import re
import matplotlib.gridspec as gridspec


def parse_patch_file(filename):
    data = {}
    regex_pattern = re.compile(r'\d+\(')  # Matches numbers followed by a '('
    with open(filename, 'r') as file:
        current_time = None
        level_aggregate = None

        for line in file:
            line = line.strip()
            if line.startswith('Time'):
                # Save the aggregated data from the previous timestamp
                if current_time is not None:
                    data[current_time] = level_aggregate

                # Prepare for a new time block
                current_time = float(line.split('=')[1].split(',')[0].strip())
                level_aggregate = {f'Level {i}': 0 for i in range(10)}  # Adjust for 10 levels
            elif line.startswith('Rank'):
                continue  # Skip the header line
            elif 'Sum:' in line or 'Ave:' in line or 'Imb:' in line or 'Weighted' in line:
                continue  # Skip summary lines
            elif line and (line[0].isdigit() or ' ' in line):
                # Extract only the patch counts before the parenthesis using regex
                matches = regex_pattern.findall(line)
                patch_counts = [int(match[:-1]) for match in matches]  # Strip the trailing '(' and convert to int
                for i, count in enumerate(patch_counts):
                    level_aggregate[f'Level {i}'] += count

        # Don't forget to save the last set of data
        if current_time is not None:
            data[current_time] = level_aggregate

    return sorted(data.items())


def parse_timing_file(filename):
    data = []
    with open(filename, 'r') as file:
        current_time = None
        categories = None
        reading_levels = False
        level_data = {}

        for line in file:
            if 'Time :' in line:
                # Extract time from the line
                time_match = re.search(r"(\d+\.\d+e[+-]?\d+)", line)
                if time_match:
                    current_time = float(time_match.group(1))

            elif 'Integration Loop' in line:
                # Start reading the levels after the next line of dashes
                _ = next(file)  # This skips the line of dashes immediately after 'Integration Loop'
                categories = next(file).strip().split()[2:]  # Read the categories from the following line
                reading_levels = True

            elif reading_levels:
                if '------' in line:
                    # This line of dashes signifies the end of the section
                    if level_data:
                        data.append((current_time, level_data))
                        level_data = {}
                    reading_levels = False
                    categories = None
                else:
                    parts = re.split(r'\s+', line.strip())
                    if parts[0].isdigit():  # This is a level data line
                        level = int(parts[0])
                        #print(parts)
                        values = list(map(float, parts[2:2+len(categories)]))  # Get values matching to the categories
                        #print(values)
                        level_data[level] = dict(zip(categories, values))

        # Check to append the last captured data if the file ends without another set of dashes
        if level_data:
            data.append((current_time, level_data))

    return sorted(data, key=lambda x: x[0])
import numpy as np

def parse_record_time_step(file_path):
    levels = 9  # Assuming levels 0 to 7
    data = {}

    with open(file_path, 'r') as file:
        header = file.readline().strip().split()  # Read and split the header line
        lines = file.readlines()

        for line in lines:
            parts = line.split()
            if len(parts) < 16:  # Skip incomplete lines
                continue
            lv, step, counter = int(parts[0]), int(parts[1]), int(parts[2])
            time_old, dtime = float(parts[3]), float(parts[5])

            if step not in data:
                data[step] = {
                    'TimeOld': time_old,
                    'dTime': np.zeros(levels),
                    'Counts': np.zeros(levels)
                }

            data[step]['dTime'][lv] += dtime
            data[step]['Counts'][lv] += 1

    # Calculate the average dTime for each level in each step
    for step, values in data.items():
        for lv in range(levels):
            if values['Counts'][lv] > 0:
                values['dTime'][lv] /= values['Counts'][lv]
            else:
                values['dTime'][lv] = 0.0

    return data

# Example usage
file_path = 'Record__TimeStep'
time_step_data = parse_record_time_step(file_path)

# Example file path
file_path = 'Record__PatchCount'
data1 = parse_patch_file(file_path)


# Assuming the file is named 'Record__Timing'
file_path = 'Record__Timing'
data2 = parse_timing_file(file_path)

# Assuming data1 and data have been defined as described
# data1 is a list of tuples with time steps and dictionaries of cell counts
# data is the dictionary from the parse_record_time_step function

# Create data3 list
data3 = []

# Iterate over data1 and match by index with data
for i, (timestep, cell_counts) in enumerate(data1):
    if i >= len(time_step_data):
        break  # Ignore if data1 has more timesteps than data

    step = sorted(time_step_data.keys())[i]  # Match by index

    # Initialize the dictionary for this time step
    level_dict = {}

    # Iterate over levels in cell_counts
    for level, count in cell_counts.items():
        level_index = int(level.split()[-1])  # Extract the level index from 'Level X'
        if level_index > 7:
            timestep_count = 0
        else:
            timestep_count = time_step_data[step]['Counts'][level_index]

        # Calculate the product and store it in the level dictionary
        level_dict[level] = int(count * timestep_count)

    # Append the tuple (timestep, level_dict) to data3
    data3.append((timestep, level_dict))


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

def plot_data(data1, data2):
    # Set up the figure and gridspec
    fig = plt.figure(figsize=(12, 6), dpi=600)
    gs = gridspec.GridSpec(2, 2)

    cells_per_patch = 8**3
    ax1 = plt.subplot(gs[0])
    colors1 = plt.get_cmap('magma')(np.linspace(0.1, 1 - 0.2, 8))  # Only 8 colors for levels 0 to 7


    def plot_cell_counts(ax, data, plot_separation=True):
        # Extract times and level data for the first plot
        times1, level_data = zip(*data)

        # Aggregate cells data for stacking
        cell_counts = {level: [round((info[level] * cells_per_patch)) for info in level_data] for level in level_data[0] if int(level.split()[1]) <= 7}
        previous_values = np.zeros(len(times1))

        # Stack area for each level, but only up to level 7
        for level in sorted(cell_counts, key=lambda x: int(x.split()[1])):  # Ensure correct level order
            current_values = np.array(cell_counts[level])
            non_zero_mask = current_values != 0  # Mask to identify non-zero points
            if np.any(non_zero_mask):  # Check if there are any non-zero points
                label = int(level.lstrip("Level "))
                if label < 4:
                    label = str(label) + ": Fluid"
                else:
                    label = str(label) + ": Wave"
                # Plot only where current_values are non-zero
                ax.fill_between(times1,
                                previous_values + current_values * non_zero_mask,
                                previous_values,
                                where=non_zero_mask,
                                color=colors1[int(level.split()[1])],
                                label=label)
            # Plot a fine white separation line
            if plot_separation:
                ax.plot(np.array(times1)[non_zero_mask], previous_values[non_zero_mask], color='white', linewidth=0.1)
            previous_values += current_values

    plot_cell_counts(ax1, data1)
    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('Number of Cells (Logarithmic)')
    #ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_yticks([128**3, 256**3, 512**3, 1024**3, 2048**3, 4096**3], [r"$128^3$", r"$256^3$", r"$512^3$", r"$1024^3$", r"$2048^3$", r"$4096^3$"])
    redshifts = np.array([100, 10, 6, 3, 2, 1])
    ax1.set_xticks(1/(redshifts+1), ["100", "10", "6", "3", "2", "1"])
    ax1.tick_params(direction='in', which='both', top=True, right=True)
    ax1.legend(title='Levels', loc='upper left')

    ax2= plt.subplot(gs[1])

    plot_cell_counts(ax2, data1)
    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('Number of Cells (Linear)')
    #ax1.set_xscale("log")

    #ax2.set_yscale("log")
    ax2.set_yticks([0, 1024**3, 2048**3], ["0", r"$1024^3$", r"$2048^3$"])
    redshifts = np.array([100, 10, 6, 3, 2, 1])
    ax2.set_xticks(1/(redshifts+1), ["100", "10", "6", "3", "2", "1"])
    ax2.tick_params(direction='in', which='both', top=True, right=True)
    ax2.legend(title='Levels', loc='upper left')


    # Mapping of original categories to new simpler categories
    category_mapping = {
        'Wave/Fluid Solver': ['Flu_Adv'],
        'Gravity Solver': ['Gra_Adv'],
        'Refinement': ['Refine', 'Flag'],
        'Other': ['-MPI', 'Buf_Ref', 'Buf_Pot', 'Buf_Flu1', 'Buf_Flu2', 'Buf_Flux', 'Buf_Rho', 'FixUp', '-MPI_FaSib', '-MPI_Real', '-MPI_Sib', 'Buf_Che', 'Che_Adv', 'FB_Adv', 'Par_2Sib', 'Par_2Son', 'Par_Coll', 'Par_K', 'Par_K-1', 'Par_KD', 'SF', 'Src_Adv', 'dt', 'Buf_Res']
    }

    # Extract times and category data for the second plot
    times2, category_dicts = zip(*data2)

    # Initialize a dictionary to hold the aggregated category data
    aggregated_data = []

    # Aggregate the data according to the new category mapping
    for time_dict in category_dicts:
        aggregated_dict = {key: 0 for key in category_mapping}
        for sub_dict in time_dict.values():
            for category, time in sub_dict.items():
                for new_category, original_categories in category_mapping.items():
                    if category in original_categories:
                        aggregated_dict[new_category] += time
                        break
        aggregated_data.append(aggregated_dict)


    # Determine all unique categories
    categories = sorted({cat for d in aggregated_data for cat in d.keys()})
    ax2 = plt.subplot(gs[2])
    color_map = plt.get_cmap('magma')
    base_colors = color_map(np.linspace(0, 1-0.2, len(categories)))


    plot_cell_counts(ax2, data3, plot_separation=False)
    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('Cum. cell updates per timestep')
    #ax1.set_xscale("log")

    #ax2.set_yscale("log")
    redshifts = np.array([100, 10, 6, 3, 2, 1])
    ax2.set_xticks(1/(redshifts+1), ["100", "10", "6", "3", "2", "1"])
    ax2.tick_params(direction='in', which='both', top=True, right=True)
    ax2.legend(title='Levels', loc='upper left')



    # Determine all unique categories
    categories = sorted({cat for d in aggregated_data for cat in d.keys()})
    ax2 = plt.subplot(gs[3])
    color_map = plt.get_cmap('magma')
    base_colors = color_map(np.linspace(0, 1-0.2, len(categories)))

    # Prepare plot data for the second plot
    values_stack = np.zeros((len(times2), len(categories)))
    for i, cat in enumerate(categories):
        values_stack[:, i] = np.array([d.get(cat, 0) for d in aggregated_data])

    print(values_stack)
    prev_values = np.zeros(len(times2))
    for idx, category in enumerate(categories):
        current_values = values_stack[:, idx]
        non_zero_mask = current_values != 0  # Mask to identify non-zero points
        if np.any(non_zero_mask):  # Check if there are any non-zero points
            # Plot the main filled area
            ax2.fill_between(times2,
                            prev_values + current_values,
                            prev_values,
                            color=base_colors[idx],
                            label=category)
            # Plot a fine white separation line
            #ax2.plot(np.array(times2)[non_zero_mask], prev_values[non_zero_mask], color='white', linewidth=0.1)
        prev_values += current_values

    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('Cum. runtime per timestep [s]')
    redshifts = np.array([100, 10, 6, 3, 2, 1])
    ax2.set_xticks(1/(redshifts+1), ["100", "10", "6", "3", "2", "1"])
    ax2.tick_params(direction='in', which='both', top=True, right=True)
    ax2.legend(title='Operations', loc='upper left')

    plt.tight_layout()
    plt.savefig("runtime.pdf", bbox_inches='tight')
    plt.show()


# Assuming data1 and data2 are predefined somewhere in your code
plot_data(data1, data2)
