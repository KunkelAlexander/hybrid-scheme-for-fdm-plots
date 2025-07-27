import pandas as pd
import os

# List of folders containing your files
folders = ['GaussianWavePacketSlow', 'GaussianWavePacketSlowFilter', 'GaussianWavePacketMedium', 'GaussianWavePacketFast']

# Prepare an empty DataFrame to store all data
all_data = pd.DataFrame()

# Loop through each folder and read files
for folder in folders:
    # Assuming your current working directory is the parent directory of these folders
    # Adjust the path as necessary
    file_path = os.path.join(folder, 'Record__L1Err')
    if os.path.exists(file_path):
        # Read the data file into a DataFrame
        temp_df = pd.read_csv(file_path, delim_whitespace=True,  comment="#",
                              names=['NGrid', 'Time', 'Error(Dens)', 'Error(Real)', 'Error(Imag)'])
        
        # Add a column for the folder name
        temp_df['Source'] = folder
        
        # Concatenate this DataFrame with the main DataFrame
        all_data = pd.concat([all_data, temp_df], ignore_index=True)

# Once all data is concatenated, save to a single CSV file
all_data.to_csv('lt_stability_gaussian.csv', index=False)

print('Data combined and saved to combined_data.csv.')

