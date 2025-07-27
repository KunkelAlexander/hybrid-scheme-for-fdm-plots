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

def calculate_mean_l1_error(df_analytical, df_numerical):
    """
    Calculate the mean L1 error between two DataFrames' Numerical and Analytical columns.
    """
    real_error = np.abs(df_analytical['Real']["Analytical"] - df_numerical['Real']["Numerical"]).mean()
    imag_error = np.abs(df_analytical['Imag']["Analytical"] - df_numerical['Imag']["Numerical"]).mean()
    return 0.5 * (real_error + imag_error)

def load_and_process_file(file_path, prefix, resolution_suffix):
    """
    Load a PlaneWave file, setting the prefix and resolution_suffix for column names.
    """
    data = pd.read_csv(file_path, delim_whitespace=True, skiprows=1,
                       names=[f'Coord', f'Numerical', f'Analytical', f'Error'])
    return data


summary_data = []

for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue

    params = parse_folder_name(folder_name)  # Parse parameters from the folder name
    
    dfs_analytical = {}
    dfs_numerical = {}

    print(folder_name, params)
    output_file = "temp_error.txt"
    subprocess.run(["python3", "compute_errors.py", base_dir, folder_name, output_file])

    with open(output_file, 'r') as file:
        mean_l1_error = float(file.read().strip())

    os.remove(output_file)  # Clean up temporary file

    #for file_name in os.listdir(folder_path):
        #if file_name.startswith('VortexPairLinear'):
        #    print(file_name)
        #    file_path = os.path.join(folder_path, file_name)
        #    if '000000_hr' in file_name:
        #        prefix = ''
        #        data_type = 'Imag' if 'Imag' in file_name else 'Real'
        #        dfs_analytical[data_type] = load_and_process_file(file_path, prefix, '_hr_lr')
        #    elif file_name.endswith('000001'):
        #        prefix = ''
        #        data_type = 'Imag' if 'Imag' in file_name else 'Real'
        #        dfs_numerical[data_type] = load_and_process_file(file_path, prefix, '')

    # Calculate the mean L1 error
    #mean_l1_error = calculate_mean_l1_error(dfs_analytical, dfs_numerical)
    if mean_l1_error is not None:
        summary_row = {param: value for param, value in params.items()}
        summary_row['Mean_L1_Error'] = mean_l1_error
        summary_data.append(summary_row)

# Convert summary data to a DataFrame and save to CSV
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(csv_output_path, index=False)
print(f"Error summary saved to {csv_output_path}")
