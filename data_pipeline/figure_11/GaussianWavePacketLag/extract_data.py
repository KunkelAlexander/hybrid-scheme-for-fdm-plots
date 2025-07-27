import pandas as pd
import os
import glob
import sys

# Check for command line arguments
if len(sys.argv) != 3:
    print("Usage: script.py <start_index> <end_index>")
    sys.exit(1)

# Start and end indices from command line arguments
start_index = int(sys.argv[1])
end_index = int(sys.argv[2])

# Base directory containing the subdirectories
base_directory = './'

# Prepare an empty DataFrame to hold all data
combined_df = pd.DataFrame()

# List all subdirectories in the base directory
subdirectories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

# Iterate through each subdirectory and read the corresponding density files across the index range
for subdir in subdirectories:
    subdir_path = os.path.join(base_directory, subdir)
    for index in range(start_index, end_index + 1):
        file_index = f'{index:06}'  # Format the index as a zero-padded string
        filename_pattern = f'Gaussian_Dens_{file_index}'
        # Find the density files that match the pattern
        density_files = glob.glob(os.path.join(subdir_path, f'*{filename_pattern}*'))
        for file in density_files:
            # Read the file into a DataFrame
            df = pd.read_csv(file, delim_whitespace=True, skiprows=1,
                             names=['Coord.', 'Numerical', 'Analytical', 'Error'])
            # Add a column to identify the source directory or file
            df['Source'] = subdir
            # Add the index as an additional column
            df['Index'] = file_index
            # Append this DataFrame to the master DataFrame
            combined_df = pd.concat([combined_df, df], ignore_index=True)

# Store the combined DataFrame to disk
output_filename = f'combined_density_{start_index}_{end_index}.csv'
combined_df.to_csv(output_filename, index=False)
print(f"Combined DataFrame stored to {output_filename}")

