import os
import pandas as pd
import re

# Set the current directory as the directory to process
directory_path = "./results/"
data = []

# Iterate over top-level folders in the current directory
for top_folder in os.listdir(directory_path):
    top_folder_path = os.path.join(directory_path, top_folder)
    if not os.path.isdir(top_folder_path):
        continue
    for item in os.listdir(top_folder_path):
        item_path = os.path.join(top_folder_path, item)
        if os.path.isdir(item_path):
            match = re.search(r"'PWave_NWavelength': (\d+)", item)
            if match:
                wavelength = int(match.group(1))
                file_path = os.path.join(item_path, 'Record__L1Err')
                print(wavelength, file_path)
                if os.path.isfile(file_path):

                    with open(file_path, 'r') as file:
                        header = file.readline().strip()
                    
                    # Determine columns based on header
                    if "Error(Imag)" in header:
                        columns_to_use = ['NGrid', 'Time', 'Error(Dens)', 'Error(Real)', 'Error(Imag)']
                        error_cols = ['Error(Real)', 'Error(Imag)']
                    elif "Error(Phas)" in header:
                        columns_to_use = ['NGrid', 'Time', 'Error(Dens)', 'Error(Phas)', 'Stub']
                        error_cols = ['Error(Dens)', 'Error(Phas)']
                    else:
                        print("Gabun")
                        continue  # Skip if header is neither

                    # Read the file with appropriate column names
                    df = pd.read_csv(file_path, delim_whitespace=True, skiprows=2, names=columns_to_use)
                    print(df)
                    # Get the first row after the initial time step
                    first_row_after_initial = df[df['Time'] > 0].iloc[0]
                    errors = first_row_after_initial[error_cols]
                    data.append([top_folder, wavelength] + errors.tolist())

# Create a dataframe from the results
results_df = pd.DataFrame(data, columns=['RunName', 'Wavelength', 'Error(Dens)', 'Error(Real)'])

# Save the results to a CSV file in the current directory
results_filename = "results_planewave_wavelength.csv"
results_df.to_csv(os.path.join(directory_path, results_filename), index=False)

print(f"Results saved to {os.path.join(directory_path, results_filename)}")

