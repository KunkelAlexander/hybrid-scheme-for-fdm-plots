import pandas as pd
import os
from functools import reduce
import numpy as np 
import ast
import subprocess
import sys

if len(sys.argv) != 3:
    print("Usage: python script_name.py base_directory csv_output_path")
    sys.exit(1)

base_dir = sys.argv[1]
csv_output_path = sys.argv[2]


def parse_folder_name(folder_name):
    """
    Parse the folder name to extract parameters and their values.
    
    Assumes the folder name structure is a series of key-value pairs separated by underscores.
    """
    # Attempt to parse the folder name as a dictionary
    try:
        # Removing leading and trailing characters that may interfere with parsing
        clean_name = folder_name.replace("'", "\"")
        params = ast.literal_eval(clean_name)
    except ValueError as e:
        print(f"Error parsing folder name: {folder_name}. Error: {e}")
        return {}
    
    return params


summary_data = []

for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    params = parse_folder_name(folder_name)  # Parse parameters from the folder name

    output_file = "temp_error.txt"
    subprocess.run(["python3", "compute_errors.py", base_dir, folder_name, output_file])

    with open(output_file, 'r') as file:
        mean_l1_error = float(file.read().strip())

    os.remove(output_file)  # Clean up temporary file

    # Calculate the mean L1 error
    summary_row = {param: value for param, value in params.items()}
    summary_row['Mean_L1_Error'] = mean_l1_error
    summary_data.append(summary_row)

# Convert summary data to a DataFrame and save to CSV
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(csv_output_path, index=False)
print(f"Error summary saved to {csv_output_path}")
    
