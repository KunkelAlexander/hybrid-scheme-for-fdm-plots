import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd



def parse_performance_file(filename):
    df = pd.read_csv(filename, delim_whitespace=True)

    return df

def parse_patch_count_file(filename):
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

    # Convert the data dictionary into a pandas DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')

    # Reset index to add 'Time' as the first column and rename the index
    df.index.name = 'Time'
    df.reset_index(inplace=True)

    return df


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
                _ = next(file)  # Skip the line of dashes immediately after 'Integration Loop'
                categories = next(file).strip().split()[2:]  # Read the categories from the following line
                reading_levels = True

            elif reading_levels:
                if '------' in line:
                    # Line of dashes signifies the end of the section
                    if level_data:
                        for category in categories:
                            # For each category, save a dictionary of levels and their corresponding values
                            data.append({
                                'Time': current_time,
                                'Operation': category,
                                **{level: level_data[level][category] for level in level_data}
                            })
                        level_data = {}
                    reading_levels = False
                    categories = None
                else:
                    parts = re.split(r'\s+', line.strip())
                    if parts[0].isdigit():  # This is a level data line
                        level = f'Level {parts[0]}'
                        values = list(map(float, parts[2:2+len(categories)]))  # Get values for the categories
                        level_data[level] = dict(zip(categories, values))

        # Append the last captured data if the file ends without another set of dashes
        if level_data:
            for category in categories:
                data.append({
                    'Time': current_time,
                    'Operation': category,
                    **{level: level_data[level][category] for level in level_data}
                })

    # Convert the list of dictionaries to a DataFrame
    timing_per_level = pd.DataFrame(data)

    # Set 'Time' and 'Operation' as the index for a multi-index DataFrame
    timing_per_level.set_index(['Time'], inplace=True)


    # Assuming timing_per_level is your original DataFrame
    result_list = []  # To store the sums for each time step
    times = []  # To store the time values

    for time, df in timing_per_level.groupby("Time"):
        df.set_index("Operation", inplace=True)
        df = df.T
        # Define categories with distinctions for Fluid and Wave
        category_mapping = {
            'Fluid/Wave Solver': ['Flu_Adv'],
            'Gravity Solver': ['Gra_Adv'],
            'Refinement': ['Refine', 'Flag'],
            'Other': ['-MPI', 'Buf_Ref', 'Buf_Pot', 'Buf_Flu1', 'Buf_Flu2',
                    'Buf_Flux', 'Buf_Rho', 'FixUp', '-MPI_FaSib',
                    '-MPI_Real', '-MPI_Sib', 'Buf_Che', 'Che_Adv',
                    'FB_Adv', 'Par_2Sib', 'Par_2Son', 'Par_Coll',
                    'Par_K', 'Par_K-1', 'Par_KD', 'SF', 'Src_Adv', 'dt', 'Buf_Res']
        }

        for key, value in category_mapping.items():
            df[key] = df[value].sum(axis=1)
            df = df.drop(columns=value)

        df["Fluid Solver"] = df["Fluid/Wave Solver"].copy()
        df["Wave Solver"]  = df["Fluid/Wave Solver"].copy()
        df["Fluid Solver"].iloc[4:] = 0
        df["Wave Solver"].iloc[:4] = 0
        df = df.drop(columns=["Fluid/Wave Solver", "Sum"])
        sum = df.sum(axis=0)
        # Append the sum and time to the list
        result_list.append(sum)
        times.append(time)

    # Create a new DataFrame from the result_list and times
    aggregated_timing = pd.DataFrame(result_list, index=times)
    aggregated_timing = aggregated_timing.rename_axis("Time")

    return aggregated_timing


# Example file path
file_path = 'data/figure_18/Record__PatchCount'
patch_counts = parse_patch_count_file(file_path)


# Assuming the file is named 'Record__Timing'
file_path = 'data/figure_18/Record__Timing'
timing_per_level = parse_timing_file(file_path)


# Assuming the file is named 'Record__Timing'
file_path = 'data/figure_18/Record__Performance'
updates_per_level = parse_performance_file(file_path)

cell_counts = patch_counts.copy()

cells_per_patch = 8**3
for i in range(10):
    cell_counts[f"Level {i}"] *= cells_per_patch


for i in range(10):
    updates_per_level[f"CellUpdates_Lv{i}"] = cell_counts[f"Level {i}"] * updates_per_level[f"NUpdate_Lv{i}"]



import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

font_size = 12
plt.rcParams.update({
    'font.size': font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    'legend.fontsize': 10,
    'figure.titlesize': font_size
})

fig = plt.figure(figsize=(12, 6), dpi=600)
gs = gridspec.GridSpec(2, 2)
redshifts = np.array([100, 10, 6, 3, 2, 1])

ax1 = plt.subplot(gs[0])
stacks = ax1.stackplot(cell_counts["Time"], *[cell_counts[f"Level {i}"] for i in range(8)],
                labels=[f'Level {i} ({"Fluid" if i < 4 else "Wave"})' for i in range(8)],
                colors=plt.get_cmap('magma')(np.linspace(0.1, 1 - 0.2, 8)),
                edgecolor='white', linewidth=0.05)  # Add white lines between stacks)

ax1.set_xlabel('Redshift z')
ax1.set_ylabel('Number of Cells (Logarithmic)')
ax1.set_yscale("log")
ax1.set_yticks([128**3, 256**3, 512**3, 1024**3, 2048**3],
                [r"$128^3$", r"$256^3$", r"$512^3$", r"$1024^3$", r"$2048^3$"])


redshifts = np.array([100, 10, 6, 3, 2, 1])
ax1.set_xticks(1 / (redshifts + 1), ["100", "10", "6", "3", "2", "1"])
ax1.tick_params(direction='in', which='both', top=True, right=True)
ax1.legend(title='Levels', loc='upper left')

ax2 = plt.subplot(gs[1])
ax2.stackplot(cell_counts["Time"], *[cell_counts[f"Level {i}"] for i in range(8)],
                labels=[f'Level {i} ({"Fluid" if i < 4 else "Wave"})' for i in range(8)],
                colors=plt.get_cmap('magma')(np.linspace(0.1, 1 - 0.2, 8)),
                edgecolor='white', linewidth=0.05)  # Add white lines between stacks)
ax2.set_xlabel(r'Redshift $z$')
ax2.set_ylabel('Number of cells (Linear)')
ax2.set_yticks([0, 1024**3], ["0", r"$1024^3$"])
ax2.set_xticks(1 / (redshifts + 1), ["100", "10", "6", "3", "2", "1"])
ax2.tick_params(direction='in', which='both', top=True, right=True)
ax2.legend(title='Levels', loc='upper left')

# Plotting for updates_per_level
ax3 = plt.subplot(gs[2])
ax3.stackplot(updates_per_level["Time"], *[updates_per_level[f"CellUpdates_Lv{i}"] for i in range(8)],
                labels=[f'Level {i} ({"Fluid" if i < 4 else "Wave"})' for i in range(8)],
                colors=plt.get_cmap('magma')(np.linspace(0.1, 1 - 0.2, 8)),
                edgecolor='white', linewidth=0.05)  # Add white lines between stacks)

ax3.set_xlabel(r'Redshift $z$')
ax3.set_ylabel('Cell updates per timestep')
ax3.set_xticks(1 / (redshifts + 1), ["100", "10", "6", "3", "2", "1"])
ax3.tick_params(direction='in', which='both', top=True, right=True)
ax3.legend(title='Levels', loc='upper left')


ax4 = plt.subplot(gs[3])
ax4.stackplot(timing_per_level.index, timing_per_level["Other"],timing_per_level["Gravity Solver"], timing_per_level["Refinement"], timing_per_level["Fluid Solver"], timing_per_level["Wave Solver"],
                labels=['Other', 'Gravity Solver', 'Refinement', 'Fluid Solver', 'Wave Solver'],
                colors=plt.get_cmap('magma')(np.linspace(0.1, 1 - 0.2, 5)),
                edgecolor='white', linewidth=0.00)  # Add white lines between stacks)
ax4.set_xlabel(r'Redshift $z$')
ax4.set_ylabel('Runtime per timestep [s]')
ax4.set_xticks(1 / (redshifts + 1), ["100", "10", "6", "3", "2", "1"])
ax4.tick_params(direction='in', which='both', top=True, right=True)
ax4.legend(title='Operations', loc='upper left')


plt.tight_layout()
plt.savefig("figures/figure_18.pdf", bbox_inches='tight')
plt.close()