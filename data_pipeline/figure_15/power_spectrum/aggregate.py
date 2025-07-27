# save_data.py
import argparse
import pandas as pd
import os
import re
import sys

# Setup command line argument parsing
parser = argparse.ArgumentParser(description='Aggregate power spectrum data into a single DataFrame')
parser.add_argument('power_spec_id', type=int, help='Highest ID of the power spectrum to aggregate')
parser.add_argument('output_file', type=str, help='Filename for the output DataFrame')
args = parser.parse_args()

# Base directories for each method
base_dirs = {
    'fluid': 'fluid',
    'gramfe': 'gramfe',
    'finite_difference': 'finite_difference',
    'spectral': 'spectral'
}

# Initialize an empty list to hold the data
data = []

# Loop through each method and its subdirectories to read power spectrum files
for method, base_dir in base_dirs.items():
    resolution_dirs = [d for d in os.listdir(base_dir) if re.match(r'n_\d+', d) and os.path.isdir(os.path.join(base_dir, d))]
    print(f"Identified resolution directories for {method}: {resolution_dirs}")
    for resolution_dir in resolution_dirs:
        for spec_id in range(args.power_spec_id + 1):
            spec_path = os.path.join(base_dir, resolution_dir, f'PowerSpec_{spec_id:06d}')
            print(f"Identified path: {spec_path}")
            if os.path.isfile(spec_path):
                # Read the power spectrum data
                df = pd.read_csv(spec_path, delim_whitespace=True, comment='#', names=['k', 'Power'])
                # Add additional information to the DataFrame
                df['Method'] = method
                df['Resolution'] = resolution_dir
                df['Spec ID'] = spec_id
                data.append(df)

# Concatenate all data into a single DataFrame
combined_df = pd.concat(data, ignore_index=True)

# Save the combined DataFrame to disk
combined_df.to_csv(args.output_file, index=False)
print(f"Data aggregated and saved to {args.output_file}")
