import pandas as pd
import os
import glob

# Base directory containing the subdirectories
base_directory = './'
# Pattern to match the density files, assuming 'index' is replaced by the file index you're interested in
file_index = '000001'  # Example index
filename_pattern = f'Gaussian_Dens_{file_index}'

# Prepare an empty DataFrame to hold all data
combined_df = pd.DataFrame()

# List all subdirectories in the base directory
subdirectories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

# Iterate through each subdirectory and read the corresponding density file
for subdir in subdirectories:
    subdir_path = os.path.join(base_directory, subdir)
    # Find the density file that matches the pattern
    density_files = glob.glob(os.path.join(subdir_path, f'*{filename_pattern}*'))
    for file in density_files:
        # Read the file into a DataFrame
        df = pd.read_csv(file, delim_whitespace=True, skiprows=1, 
                         names=['Coord.', 'Numerical', 'Analytical', 'Error'])
        # Optionally, add a column to identify the source directory or file
        df['Source'] = subdir
        # Append this DataFrame to the master DataFrame
        combined_df = pd.concat([combined_df, df], ignore_index=True)

# Store the combined DataFrame to disk
output_filename = f'combined_density_{file_index}.csv'
combined_df.to_csv(output_filename, index=False)
print(f"Combined DataFrame stored to {output_filename}")

