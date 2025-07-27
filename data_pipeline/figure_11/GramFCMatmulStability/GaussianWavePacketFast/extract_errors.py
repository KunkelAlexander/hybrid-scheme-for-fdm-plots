import os
import pandas as pd
import re
import sys


# Function to process directories and extract data
def process_directory(directory_path):
    data = []
    # Iterate over all runs in the main directory
    for run in os.listdir(directory_path):
        run_path = os.path.join(directory_path, run)
        if os.path.isdir(run_path):
            # Iterate over all subruns within each run
            for subrun in os.listdir(run_path):
                subrun_path = os.path.join(run_path, subrun)
                if os.path.isdir(subrun_path):
                    # Extract the velocity value from the subrun directory name
                    match = re.search(r"'Gau_v0': ([\d.]+)", subrun)
                    if match:
                        velocity = float(match.group(1))
                        file_path = os.path.join(subrun_path, 'Record__L1Err')
                        if os.path.isfile(file_path):
                            # Read the file, skipping the first two lines and using the manually specified column names
                            df = pd.read_csv(file_path, delim_whitespace=True, skiprows=2, names= ['NGrid', 'Time', 'Error(Dens)', 'Error(Real)', 'Error(Imag)'])

                            # Get the first row after the initial time step
                            first_row_after_initial = df[df['Time'] > 0].iloc[0]
                            errors = first_row_after_initial[['Error(Dens)', 'Error(Real)', 'Error(Imag)']].tolist()

                            # Append the run name, velocity, and the data to the list
                            data.append([run] + [velocity] + errors)

    return data

# Check if the directory path is given as a command-line argument
if len(sys.argv) != 2:
    print("Usage: python script.py <path_to_your_directory>")
    sys.exit(1)

directory_path = sys.argv[1]

# Manually specify the column names, adding 'RunName' and 'Velocity' for the new columns
column_names = ['RunName', 'Velocity', 'Error(Dens)', 'Error(Real)', 'Error(Imag)']

# Process the directory and collect data
data = process_directory(directory_path)

# Create a dataframe from the results
results_df = pd.DataFrame(data, columns=column_names)

# Save the results to a single CSV file
results_filename = "results.csv"
results_df.to_csv(os.path.join(directory_path, results_filename), index=False)

print(f"Results saved to {os.path.join(directory_path, results_filename)}")

